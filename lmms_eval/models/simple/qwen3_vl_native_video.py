"""Qwen3-VL wrapper for controlled native-resolution video experiments.

This wrapper deliberately leaves the existing ``qwen3_vl`` implementation
unchanged.  It samples a fixed number of frames itself and sends those frames
to the Hugging Face Qwen3-VL video processor, bypassing qwen-vl-utils' smaller
per-frame pixel cap.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger as eval_logger
from transformers.video_utils import VideoMetadata

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

SUPPORTED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
MAX_FRAME_BACKTRACK = 32


def _decode_frame_at_or_before(
    capture: cv2.VideoCapture,
    requested_index: int,
) -> tuple[np.ndarray, int]:
    """Read the requested frame, backing up over inaccurate container tails."""

    first_candidate = max(0, requested_index)
    last_candidate = max(-1, first_candidate - MAX_FRAME_BACKTRACK - 1)
    for candidate in range(first_candidate, last_candidate, -1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, frame_bgr = capture.read()
        if ok:
            reported_position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            actual_index = reported_position if reported_position >= 0 else candidate
            if actual_index != requested_index:
                eval_logger.warning(
                    f"Video frame {requested_index} was unavailable; "
                    f"using nearest decodable frame {actual_index}."
                )
            return frame_bgr, actual_index

    raise RuntimeError(
        f"Could not decode frame {requested_index} or any of the previous "
        f"{MAX_FRAME_BACKTRACK} frames"
    )


def decode_uniform_video(video_path: str, num_frames: int) -> tuple[torch.Tensor, VideoMetadata]:
    """Decode uniformly spaced RGB frames without resizing them.

    The returned tensor has shape ``[T, C, H, W]``.  Frame indices are based
    on the source stream, so paired videos with matching FPS/frame counts use
    the same temporal positions.
    """

    path = Path(video_path)
    if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video suffix for {path}; expected one of {sorted(SUPPORTED_VIDEO_SUFFIXES)}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    try:
        total_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if total_frames <= 0:
            raise RuntimeError(f"Could not determine frame count for {path}")
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Could not determine FPS for {path}: {fps}")

        sample_count = min(num_frames, total_frames)
        frame_indices = np.rint(np.linspace(0, total_frames - 1, sample_count)).astype(np.int64).tolist()

        # Qwen3-VL groups frames in pairs via temporal_patch_size=2.
        if len(frame_indices) % 2:
            frame_indices.append(frame_indices[-1])

        frames = []
        decoded_frame_indices = []
        for frame_index in frame_indices:
            try:
                frame_bgr, decoded_frame_index = _decode_frame_at_or_before(capture, frame_index)
            except RuntimeError as exc:
                raise RuntimeError(f"Could not decode frame {frame_index} from {path}: {exc}") from exc

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous())
            decoded_frame_indices.append(decoded_frame_index)

        video = torch.stack(frames)
        metadata = VideoMetadata(
            total_num_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            duration=total_frames / fps,
            video_backend="opencv-uniform-native",
            frames_indices=decoded_frame_indices,
        )
        return video, metadata
    finally:
        capture.release()


@register_model("qwen3_vl_native_video")
class Qwen3_VL_NativeVideo(Qwen3_VL):
    """Qwen3-VL with fixed temporal sampling and native spatial inputs.

    The defaults preserve twelve patch-aligned Full HD frames within the
    official Qwen3-VL processor's 25,165,824 total-pixel video budget.
    """

    def __init__(
        self,
        native_num_frames: int = 12,
        native_min_total_pixels: int = 4096,
        native_max_total_pixels: int = 25165824,
        **kwargs,
    ) -> None:
        if native_num_frames <= 0 or native_num_frames % 2:
            raise ValueError("native_num_frames must be a positive even integer")
        if native_min_total_pixels <= 0:
            raise ValueError("native_min_total_pixels must be positive")
        if native_max_total_pixels < native_min_total_pixels:
            raise ValueError("native_max_total_pixels must be >= native_min_total_pixels")

        batch_size = int(kwargs.get("batch_size", 1))
        if batch_size != 1:
            raise ValueError("qwen3_vl_native_video currently requires batch_size=1")

        super().__init__(max_num_frames=native_num_frames, **kwargs)
        self.native_num_frames = native_num_frames
        self.native_min_total_pixels = native_min_total_pixels
        self.native_max_total_pixels = native_max_total_pixels

    def _preprocess_chunk(self, chunk):
        """Build one request using pre-sampled, unresized source frames."""

        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        if len(contexts) != 1:
            raise ValueError("qwen3_vl_native_video expects one request per batch")

        contexts = list(contexts)
        gen_kwargs = all_gen_kwargs[0]
        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected gen_kwargs['until'] to be str or list, got {type(until)}")
        until = [item for item in until if item != "\n\n"]

        context = contexts[0].replace("<image>", "")
        if self.reasoning_prompt:
            context = context.strip() + self.reasoning_prompt
        contexts[0] = context

        visuals = doc_to_visual[0](self.task_dict[task[0]][split[0]][doc_id[0]])
        if visuals is None or len(visuals) != 1 or not isinstance(visuals[0], str):
            raise ValueError("qwen3_vl_native_video requires exactly one local video path per document")
        video_path = visuals[0]

        message = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": context},
                ],
            },
        ]
        texts = self._apply_chat_template([message])

        video, metadata = decode_uniform_video(video_path, self.native_num_frames)
        source_shape = tuple(video.shape)
        processor_size = {
            "shortest_edge": self.native_min_total_pixels,
            "longest_edge": self.native_max_total_pixels,
        }
        inputs = self.processor(
            text=texts,
            videos=[video],
            video_metadata=[metadata],
            do_sample_frames=False,
            size=processor_size,
            return_tensors="pt",
        )

        grid_thw = inputs.get("video_grid_thw")
        if grid_thw is not None:
            merge_size = self.processor.video_processor.merge_size
            visual_tokens = (grid_thw.prod(dim=1) // (merge_size**2)).tolist()
            eval_logger.info("Native video preprocessing: " f"path={video_path}, source_tensor={source_shape}, " f"grid_thw={grid_thw.tolist()}, visual_tokens={visual_tokens}, " f"frame_indices={metadata.frames_indices}")

        return inputs, contexts, gen_kwargs, until
