# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests multimodal GGUF models against unquantized HuggingFace baselines.

Downloads backbone + mmproj GGUF files, runs inference via vllm.LLM,
and compares logprobs against AutoModelForImageTextToText.
"""

import gc
import os
from typing import Any, NamedTuple

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import rescale_image_size

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_TOKENS = 32
NUM_LOGPROBS = 10
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.8


class GGUFMMTestConfig(NamedTuple):
    original_model: str
    gguf_model_path: str
    prompts: list[str]
    image_names: list[str]
    mm_processor_kwargs: dict[str, Any] | None = None


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
    gguf_model_path="google/gemma-3-4b-it-qat-q4_0-gguf:Q4_0",
    prompts=_GEMMA3_PROMPTS,
    image_names=_GEMMA3_IMAGE_NAMES,
)

GEMMA3_CONFIG_PAN_AND_SCAN = GGUFMMTestConfig(
    original_model="google/gemma-3-4b-it",
    gguf_model_path="google/gemma-3-4b-it-qat-q4_0-gguf:Q4_0",
    prompts=_GEMMA3_PROMPTS,
    image_names=_GEMMA3_IMAGE_NAMES,
    mm_processor_kwargs={"do_pan_and_scan": True},
)

_QWEN35_PROMPTS = [
    (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "What's the content in the center of the image?"
        "<|im_end|>\n<|im_start|>assistant\n"
    ),
    (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "What is the season?"
        "<|im_end|>\n<|im_start|>assistant\n"
    ),
]
_QWEN35_IMAGE_NAMES = ["stop_sign", "cherry_blossom"]

QWEN35_CONFIG = GGUFMMTestConfig(
    original_model="Qwen/Qwen3.5-0.8B",
    gguf_model_path="unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    prompts=_QWEN35_PROMPTS,
    image_names=_QWEN35_IMAGE_NAMES,
)

QWEN35_MOE_CONFIG = GGUFMMTestConfig(
    original_model="Qwen/Qwen3.5-35B-A3B",
    gguf_model_path="unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
    prompts=_QWEN35_PROMPTS,
    image_names=_QWEN35_IMAGE_NAMES,
)

GEMMA3_MODELS_TO_TEST = [
    pytest.param(GEMMA3_CONFIG, marks=pytest.mark.slow),
    pytest.param(GEMMA3_CONFIG_PAN_AND_SCAN, marks=pytest.mark.slow),
]
QWEN35_MODELS_TO_TEST = [
    QWEN35_CONFIG,
    pytest.param(QWEN35_MOE_CONFIG, marks=pytest.mark.slow),
]


def _vllm_generate_greedy_logprobs(
    model_path: str,
    tokenizer_name: str,
    prompts: list[str],
    images: list,
    max_tokens: int,
    num_logprobs: int,
    dtype: str,
    mm_processor_kwargs: dict[str, Any] | None,
) -> list[tuple[list[int], str, list[dict[int, float] | None]]]:
    """Run inference via vllm.LLM and return (token_ids, text, logprobs)."""
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        mm_processor_kwargs=mm_processor_kwargs,
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
        for image, prompt in zip(images, model.prompts)
    ]

    # Run vLLM GGUF first to keep CUDA context clean before loading HF model.
    gguf_outputs_per_case = [
        _vllm_generate_greedy_logprobs(
            model_path=model.gguf_model_path,
            tokenizer_name=model.original_model,
            prompts=prompts,
            images=scaled_images,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            dtype=dtype,
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
    GEMMA3_MODELS_TO_TEST,
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for multimodal GGUF tests.",
)
@pytest.mark.parametrize(
    "model",
    QWEN35_MODELS_TO_TEST,
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [MAX_TOKENS])
@pytest.mark.parametrize("num_logprobs", [NUM_LOGPROBS])
def test_qwen35_mm_gguf(
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    run_multimodal_gguf_test(model, dtype, max_tokens, num_logprobs)
