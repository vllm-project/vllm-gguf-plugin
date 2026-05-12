# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests GGUF models against unquantized models generations.
Uses vllm.LLM directly with check_logprobs_close comparison.
"""

import os
import warnings
from typing import NamedTuple

import pytest
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024


class GGUFTestConfig(NamedTuple):
    original_model: str
    gguf_model_path: str  # Full path to .gguf file


QWEN2_CONFIG = GGUFTestConfig(
    original_model="Qwen/Qwen2.5-1.5B-Instruct",
    gguf_model_path="Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q6_K",
)

QWEN3_CONFIG = GGUFTestConfig(
    original_model="Qwen/Qwen3-0.6B",
    gguf_model_path="unsloth/Qwen3-0.6B-GGUF:Q8_0",
)

PHI3_CONFIG = GGUFTestConfig(
    original_model="microsoft/Phi-3.5-mini-instruct",
    gguf_model_path="bartowski/Phi-3.5-mini-instruct-GGUF:IQ4_XS",
)

GPT2_CONFIG = GGUFTestConfig(
    original_model="openai-community/gpt2-large",
    gguf_model_path="QuantFactory/gpt2-large-GGUF:Q4_K_M",
)

STABLELM_CONFIG = GGUFTestConfig(
    original_model="stabilityai/stablelm-3b-4e1t",
    gguf_model_path="afrideva/stablelm-3b-4e1t-GGUF:Q4_K_M",
)

DOLPHIN_CONFIG = GGUFTestConfig(
    # Test VocabParallelEmbedding sharding issue.
    original_model="cognitivecomputations/TinyDolphin-2.8-1.1b",
    gguf_model_path="tsunemoto/TinyDolphin-2.8-1.1b-GGUF:Q6_K",
)

GEMMA3_CONFIG = GGUFTestConfig(
    original_model="google/gemma-3-270m-it",
    gguf_model_path="ggml-org/gemma-3-270m-it-qat-GGUF:Q4_0",
)

OLMOE_CONFIG = GGUFTestConfig(
    original_model="allenai/OLMoE-1B-7B-0125",
    gguf_model_path="allenai/OLMoE-1B-7B-0125-GGUF:Q4_0",
)

MODELS = [
    # LLAMA_CONFIG,  # broken: https://github.com/vllm-project/vllm/issues/19458
    QWEN2_CONFIG,
    QWEN3_CONFIG,
    PHI3_CONFIG,
    GPT2_CONFIG,
    STABLELM_CONFIG,
    # DOLPHIN_CONFIG,   # FIXME(Isotr0py): Investigate later
    GEMMA3_CONFIG,
    OLMOE_CONFIG,
    # STARCODER_CONFIG,  # broken
]


def _generate_greedy_logprobs(
    model_path: str,
    prompts: list[str],
    max_tokens: int,
    num_logprobs: int,
    tokenizer_name: str | None = None,
    quantization: str | None = None,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
) -> list[tuple[list[int], str, list[dict[int, float] | None]]]:
    """Generate greedy outputs with logprobs using vllm.LLM.

    Returns list of (token_ids, text, logprobs_list) tuples.
    Each logprobs element maps token_id -> logprob value.
    """
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        quantization=quantization,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=num_logprobs,
    )
    outputs = llm.generate(prompts, sampling_params)

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


def check_logprobs_close(
    outputs_0_lst: list[tuple[list[int], str, list]],
    outputs_1_lst: list[tuple[list[int], str, list]],
    name_0: str,
    name_1: str,
) -> None:
    """Compare logprobs of two model outputs.

    For each generated token position:
    - If tokens match, continue
    - If tokens differ, verify each model's selected token appears in
      the other model's top-k logprobs, then break (sequences diverge)
    """
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
                # Each model's selected token must be in the other's top-k
                assert tok_0 in lp_1, fail_msg
                assert tok_1 in lp_0, fail_msg

                warnings.warn(fail_msg, stacklevel=2)
                break  # sequences diverge from here


def check_model_outputs(
    prompts: list[str],
    model: GGUFTestConfig,
    max_tokens: int,
    num_logprobs: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model.original_model)
    if tokenizer.chat_template is not None:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    gguf_outputs = _generate_greedy_logprobs(
        model_path=model.gguf_model_path,
        prompts=prompts[:-1],
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tokenizer_name=model.original_model,
        quantization="gguf",
    )

    original_outputs = _generate_greedy_logprobs(
        model_path=model.original_model,
        prompts=prompts[:-1],
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
    )

    check_logprobs_close(
        outputs_0_lst=original_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="original",
        name_1="gguf",
    )


@pytest.mark.parametrize(
    "model",
    MODELS,
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    example_prompts: list[str],
    model: GGUFTestConfig,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    check_model_outputs(example_prompts, model, max_tokens, num_logprobs)


def _multi_gpu_test(num_gpus: int):
    """Skip if not enough GPUs available."""
    return pytest.mark.skipif(
        torch.cuda.device_count() < num_gpus,
        reason=f"Need {num_gpus} GPUs, found {torch.cuda.device_count()}",
    )


@pytest.mark.parametrize("model", [QWEN3_CONFIG])
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tp_size", [2])
@_multi_gpu_test(num_gpus=2)
def test_distributed(
    example_prompts: list[str],
    model: GGUFTestConfig,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
) -> None:
    """Test GGUF model with tensor parallelism across 2 GPUs."""
    tokenizer = AutoTokenizer.from_pretrained(model.original_model)
    prompts = example_prompts
    if tokenizer.chat_template is not None:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    gguf_outputs = _generate_greedy_logprobs(
        model_path=model.gguf_model_path,
        prompts=prompts[:-1],
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tokenizer_name=model.original_model,
        quantization="gguf",
        tensor_parallel_size=tp_size,
        dtype="half",
    )

    original_outputs = _generate_greedy_logprobs(
        model_path=model.original_model,
        prompts=prompts[:-1],
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
        dtype="half",
    )

    check_logprobs_close(
        outputs_0_lst=original_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="original",
        name_1="gguf",
    )
