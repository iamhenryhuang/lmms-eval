"""Qwen2.5-VL wrapper for controlled native-resolution video experiments.

Mirrors ``qwen3_vl_native_video.py``: samples a fixed number of frames itself
and sends those frames to the Hugging Face Qwen2.5-VL video processor with an
explicit pixel budget, so paired High/Low resolution videos can be compared
under the same frame count and temporal sampling as the other models in this
experiment, instead of relying on qwen-vl-utils' default per-frame pixel cap.

Unlike ``Qwen3_VL``, the existing ``Qwen2_5_VL`` simple-model class has no
``_preprocess_chunk`` hook to override, so ``generate_until`` is reimplemented
here in full; only the visual-input construction differs from the parent.
"""

from __future__ import annotations

from typing import List

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
from lmms_eval.models.simple.qwen3_vl_native_video import decode_uniform_video


@register_model("qwen2_5_vl_native_video")
class Qwen2_5_VL_NativeVideo(Qwen2_5_VL):
    """Qwen2.5-VL with fixed temporal sampling and native spatial inputs."""

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
            raise ValueError("qwen2_5_vl_native_video currently requires batch_size=1")

        super().__init__(max_num_frames=native_num_frames, **kwargs)
        self.native_num_frames = native_num_frames
        self.native_min_total_pixels = native_min_total_pixels
        self.native_max_total_pixels = native_max_total_pixels

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            if len(contexts) != 1:
                raise ValueError("qwen2_5_vl_native_video expects one request per batch")

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
                raise ValueError("qwen2_5_vl_native_video requires exactly one local video path per document")
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
            texts = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

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
                eval_logger.info(
                    "Native video preprocessing: "
                    f"path={video_path}, source_tensor={source_shape}, "
                    f"grid_thw={grid_thw.tolist()}, visual_tokens={visual_tokens}, "
                    f"frame_indices={metadata.frames_indices}"
                )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
