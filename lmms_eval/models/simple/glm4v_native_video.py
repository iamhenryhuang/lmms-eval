"""GLM-4.6V wrapper for controlled native-resolution video experiments."""

from __future__ import annotations

from typing import List

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.glm4v import GLM4V
from lmms_eval.models.simple.qwen3_vl_native_video import decode_uniform_video


@register_model("glm4v_native_video")
class GLM4VNativeVideo(GLM4V):
    """GLM-4.6V with fixed uniform frame sampling and native source frames."""

    def __init__(
        self,
        native_num_frames: int = 24,
        native_min_total_pixels: int = 4096,
        native_max_total_pixels: int = 6291456,
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        if native_num_frames <= 0 or native_num_frames % 2:
            raise ValueError("native_num_frames must be a positive even integer")
        if native_min_total_pixels <= 0:
            raise ValueError("native_min_total_pixels must be positive")
        if native_max_total_pixels < native_min_total_pixels:
            raise ValueError(
                "native_max_total_pixels must be >= native_min_total_pixels"
            )

        batch_size = int(kwargs.get("batch_size", 1))
        if batch_size != 1:
            raise ValueError("glm4v_native_video currently requires batch_size=1")

        super().__init__(**kwargs)
        self.native_num_frames = native_num_frames
        self.native_min_total_pixels = native_min_total_pixels
        self.native_max_total_pixels = native_max_total_pixels
        self.enable_thinking = enable_thinking

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []

        def _collate(request):
            tokens = self.tokenizer.encode(request[0])
            return -len(tokens), request[0]

        progress = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        reordered = utils.Collator(
            [request.args for request in requests],
            _collate,
            grouping=True,
        )

        for chunk in reordered.get_batched(n=1, batch_fn=None):
            contexts, all_gen_kwargs, doc_to_visual, doc_ids, tasks, splits = zip(*chunk)
            context = contexts[0].replace("<image>", "")
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected until to be str or list, got {type(until)}")

            visuals = doc_to_visual[0](
                self.task_dict[tasks[0]][splits[0]][doc_ids[0]]
            )
            if (
                visuals is None
                or len(visuals) != 1
                or not isinstance(visuals[0], str)
            ):
                raise ValueError(
                    "glm4v_native_video requires exactly one local video path per document"
                )
            video_path = visuals[0]

            video, metadata = decode_uniform_video(
                video_path,
                self.native_num_frames,
            )
            source_shape = tuple(video.shape)

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": context},
                    ],
                }
            )

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            inputs = self.processor(
                text=[prompt],
                videos=[video],
                video_metadata=[metadata],
                do_sample_frames=False,
                size={
                    "shortest_edge": self.native_min_total_pixels,
                    "longest_edge": self.native_max_total_pixels,
                },
                padding=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)

            grid_thw = inputs.get("video_grid_thw")
            pixel_values = inputs.get("pixel_values_videos")
            visual_tokens = None
            if grid_thw is not None:
                merge_size = self.processor.video_processor.merge_size
                visual_tokens = (
                    grid_thw.prod(dim=1) // (merge_size**2)
                ).tolist()
            eval_logger.info(
                "Native GLM-4.6V video preprocessing: "
                f"path={video_path}, source_tensor={source_shape}, "
                f"max_total_pixels={self.native_max_total_pixels}, "
                f"pixel_values={tuple(pixel_values.shape) if pixel_values is not None else None}, "
                f"grid_thw={grid_thw.tolist() if grid_thw is not None else None}, "
                f"visual_tokens={visual_tokens}, "
                f"frame_indices={metadata.frames_indices}"
            )

            inputs = inputs.to(self.device)

            defaults = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current = {**defaults, **gen_kwargs}
            do_sample = bool(current.get("do_sample", False))
            temperature = current["temperature"] if do_sample else None
            top_p = current["top_p"] if do_sample else None

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=current["num_beams"],
                    max_new_tokens=current["max_new_tokens"],
                    use_cache=self.use_cache,
                )

            generated = output_ids[:, inputs["input_ids"].shape[1] :]
            answer = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            for term in until:
                if term:
                    answer = answer.split(term)[0]

            responses.append(answer)
            self.cache_hook.add_partial(
                "generate_until",
                (context, gen_kwargs),
                answer,
            )
            progress.update(1)

        progress.close()
        return reordered.get_original(responses)
