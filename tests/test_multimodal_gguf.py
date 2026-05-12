# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests multimodal GGUF models against unquantized HuggingFace baselines.

Downloads backbone + mmproj GGUF files, runs inference via vllm.LLM,
and compares logprobs against AutoModelForImageTextToText.
"""

import gc
import os
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from pytest import MarkDecorator
from transformers import AutoModelForImageTextToText, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import rescale_image_size

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_TOKENS = 32
NUM_LOGPROBS = 10
GPU_MEMORY_UTILIZATION = 0.8


class GGUFMMTestConfig(NamedTuple):
    original_model: str
    gguf_repo: str
    gguf_backbone: str
    gguf_mmproj: str
    prompt: list[str]
    image_names: list[str]
    max_model_len: int = 4096
    marks: list[MarkDecorator] = []
    mm_processor_kwargs: dict[str, Any] = {}

    @property
    def gguf_model(self) -> str:
        """Download backbone + mmproj; return local path to backbone."""
        repo_path = Path(self.gguf_repo)
        if repo_path.is_dir():
            mmproj_path = repo_path / self.gguf_mmproj
            backbone_path = repo_path / self.gguf_backbone
            assert mmproj_path.is_file(), f"Missing GGUF mmproj file: {mmproj_path}"
            assert backbone_path.is_file(), (
                f"Missing GGUF backbone file: {backbone_path}"
            )
            return str(backbone_path)
        hf_hub_download(self.gguf_repo, filename=self.gguf_mmproj)
        return hf_hub_download(self.gguf_repo, filename=self.gguf_backbone)


_GEMMA3_PROMPTS = [
    (
        "<bos><start_of_turn>user\n"
        "<start_of_image>What's the content in the center of the image?"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
    (
        "<bos><start_of_turn>user\n"
        "<start_of_image>What is the season?"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
]
_GEMMA3_IMAGE_NAMES = ["stop_sign", "cherry_blossom"]

GEMMA3_CONFIG = GGUFMMTestConfig(
    original_model="google/gemma-3-4b-it",
    gguf_repo="google/gemma-3-4b-it-qat-q4_0-gguf",
    gguf_backbone="gemma-3-4b-it-q4_0.gguf",
    gguf_mmproj="mmproj-model-f16-4B.gguf",
    prompt=_GEMMA3_PROMPTS,
    image_names=_GEMMA3_IMAGE_NAMES,
    max_model_len=4096,
    marks=[pytest.mark.slow],
    mm_processor_kwargs={},
)

GEMMA3_CONFIG_PAN_AND_SCAN = GGUFMMTestConfig(
    original_model="google/gemma-3-4b-it",
    gguf_repo="google/gemma-3-4b-it-qat-q4_0-gguf",
    gguf_backbone="gemma-3-4b-it-q4_0.gguf",
    gguf_mmproj="mmproj-model-f16-4B.gguf",
    prompt=_GEMMA3_PROMPTS,
    image_names=_GEMMA3_IMAGE_NAMES,
    max_model_len=4096,
    marks=[pytest.mark.slow],
    mm_processor_kwargs={"do_pan_and_scan": True},
)

MODELS_TO_TEST = [GEMMA3_CONFIG, GEMMA3_CONFIG_PAN_AND_SCAN]


def _vllm_generate_greedy_logprobs(
    model_path: str,
    tokenizer_name: str,
    prompts: list[str],
    images: list,
    max_tokens: int,
    num_logprobs: int,
    dtype: str,
    max_model_len: int,
    mm_processor_kwargs: dict[str, Any],
) -> list[tuple[list[int], str, list[dict[int, float] | None]]]:
    """Run inference via vllm.LLM and return (token_ids, text, logprobs)."""
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=max_model_len,
        mm_processor_kwargs=mm_processor_kwargs or None,
    )
    try:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
        )
        inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": image}}
            for prompt, image in zip(prompts, images)
        ]
        outputs = llm.generate(inputs, sampling_params)

        results = []
        for req_output in outputs:
            sample = req_output.outputs[0]
            token_ids = list(sample.token_ids)
            text = sample.text
            logprobs_list: list[dict[int, float] | None] = []
            if sample.logprobs:
                for lp in sample.logprobs:
                    if lp is not None:
                        logprobs_list.append(
                            {tok_id: info.logprob for tok_id, info in lp.items()}
                        )
                    else:
                        logprobs_list.append(None)
            results.append((token_ids, text, logprobs_list))
        return results
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _hf_generate_greedy_logprobs(
    model_name: str,
    prompts: list[str],
    images: list,
    max_tokens: int,
    num_logprobs: int,
    dtype: str,
) -> list[tuple[list[int], str, list[dict[int, float]]]]:
    """Run inference via HuggingFace and return (token_ids, text, logprobs)."""
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name)
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch_dtype
    ).to(device)
    hf_model.eval()

    results = []
    for prompt, image in zip(prompts, images):
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            output = hf_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0][prompt_len:]
        text = processor.decode(generated_ids, skip_special_tokens=True)

        logprobs_list: list[dict[int, float]] = []
        for score in output.scores:
            lp = F.log_softmax(score[0].float(), dim=-1)
            topk = torch.topk(lp, num_logprobs)
            logprobs_list.append(
                {
                    tid.item(): lp_val.item()
                    for tid, lp_val in zip(topk.indices, topk.values)
                }
            )
        results.append((generated_ids.tolist(), text, logprobs_list))

    del hf_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return results


def check_logprobs_close(
    outputs_0_lst: list[tuple[list[int], str, list]],
    outputs_1_lst: list[tuple[list[int], str, list]],
    name_0: str,
    name_1: str,
) -> None:
    """Compare two model output logprob sequences for approximate equality."""
    import warnings

    assert len(outputs_0_lst) == len(outputs_1_lst)

    for prompt_idx, (out_0, out_1) in enumerate(zip(outputs_0_lst, outputs_1_lst)):
        ids_0, text_0, lps_0 = out_0
        ids_1, text_1, lps_1 = out_1

        if lps_0 is None:
            lps_0 = [None] * len(ids_0)
        if lps_1 is None:
            lps_1 = [None] * len(ids_1)

        for idx, (tok_0, tok_1) in enumerate(zip(ids_0, ids_1)):
            if tok_0 != tok_1:
                lp_0 = lps_0[idx] if idx < len(lps_0) else None
                lp_1 = lps_1[idx] if idx < len(lps_1) else None

                fail_msg = (
                    f"Test {prompt_idx}, token {idx}:"
                    f"\nMatched tokens: {ids_0[:idx]}"
                    f"\n{name_0}: {text_0!r}  token={tok_0}  logprobs={lp_0}"
                    f"\n{name_1}: {text_1!r}  token={tok_1}  logprobs={lp_1}"
                )

                assert lp_0 is not None, fail_msg
                assert lp_1 is not None, fail_msg
                assert tok_0 in lp_1, fail_msg
                assert tok_1 in lp_0, fail_msg

                warnings.warn(fail_msg, stacklevel=2)
                break


def run_multimodal_gguf_test(
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    images = [ImageAsset(name).pil_image for name in model.image_names]
    size_factors = [0.25, 0.5, 1.0]
    inputs_per_image = [
        (
            [prompt for _ in size_factors],
            [rescale_image_size(image, factor) for factor in size_factors],
        )
        for image, prompt in zip(images, model.prompt)
    ]

    # Run vLLM GGUF first to keep CUDA context clean before loading HF model.
    gguf_outputs_per_case = [
        _vllm_generate_greedy_logprobs(
            model_path=model.gguf_model,
            tokenizer_name=model.original_model,
            prompts=prompts,
            images=scaled_images,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            dtype=dtype,
            max_model_len=model.max_model_len,
            mm_processor_kwargs=model.mm_processor_kwargs,
        )
        for prompts, scaled_images in inputs_per_image
    ]

    hf_outputs_per_case = [
        _hf_generate_greedy_logprobs(
            model_name=model.original_model,
            prompts=prompts,
            images=scaled_images,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            dtype=dtype,
        )
        for prompts, scaled_images in inputs_per_image
    ]

    for hf_outputs, gguf_outputs in zip(hf_outputs_per_case, gguf_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=gguf_outputs,
            name_0="hf",
            name_1="gguf",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for multimodal GGUF tests.",
)
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(test_config, marks=test_config.marks)
        for test_config in MODELS_TO_TEST
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [MAX_TOKENS])
@pytest.mark.parametrize("num_logprobs", [NUM_LOGPROBS])
def test_gemma3_mm_gguf(
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    run_multimodal_gguf_test(model, dtype, max_tokens, num_logprobs)
