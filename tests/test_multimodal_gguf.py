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
    # Per-model, because a model whose image processor emits more patch tokens
    # than MAX_MODEL_LEN leaves would have its prompt truncated -- and the
    # comparison against HF would then be measuring the truncation.
    max_model_len: int = MAX_MODEL_LEN
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

# No ``<|begin_of_text|>``: unlike Gemma 3, this tokenizer's post-processor
# prepends it, so writing it here would produce two.
_MUSE_GLIMMER_PROMPTS = [
    (
        "<|start|>user<|message|><|patch|>"
        "What's the content in the center of the image?"
        "<|eot|><|start|>assistant"
    ),
    ("<|start|>user<|message|><|patch|>What is the season?<|eot|><|start|>assistant"),
]

# The GGUF repo ships neither ``config.json`` nor a tokenizer, so the config has
# to come from the original repo.  Naming it as the tokenizer is enough: the
# plugin falls back to the tokenizer path when resolving the GGUF config source.
MUSE_GLIMMER_CONFIG = GGUFMMTestConfig(
    original_model="meta-models/Muse-Glimmer-30B",
    gguf_model_path="meta-models/Muse-Glimmer-30B-GGUF:Q4_K_XL",
    prompts=_MUSE_GLIMMER_PROMPTS,
    image_names=_GEMMA3_IMAGE_NAMES,
    # The image processor emits up to 4096 patch tokens on its own, so leave
    # room for that plus the prompt.
    max_model_len=8192,
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
    max_model_len: int,
    mm_processor_kwargs: dict[str, Any] | None,
) -> list[tuple[list[int], str, list[dict[int, float] | None]]]:
    """Run inference via vllm.LLM and return (token_ids, text, logprobs)."""
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=max_model_len,
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


# Q4_K noise has no direction, so a shift that does is the signature of a
# conversion the adapter got wrong.  Measured across the shipped prompts and
# three image scales, the shared prefix drifts by at most 0.20; a norm that never
# had its offset removed shifts every logprob together and lands far outside
# this.
MAX_LOGPROB_BIAS = 0.5


def check_logprobs_unbiased(
    outputs_0_lst: list[tuple[list[int], str, list]],
    outputs_1_lst: list[tuple[list[int], str, list]],
    name_0: str,
    name_1: str,
) -> None:
    """Compare the two runs quantitatively wherever that is meaningful.

    Stricter than :func:`check_logprobs_close`, which compares nothing at all
    while the tokens agree, tolerates the first divergence with a warning, and
    zips the two sequences without checking their lengths -- so an empty run, a
    truncated run, and a shared token sequence whose distribution has drifted all
    pass it.  None of those show up as a load error for Muse Glimmer, whose four
    conversions each produce fluent output when they are undone wrongly.

    Comparison stops at the first divergence, and that boundary is the whole
    point rather than a shortcut.  Greedy decoding conditions each step on what it
    already emitted, so up to and including the divergence both runs share a
    context and a logprob difference is attributable; past it they are completing
    different sentences, and position-wise differences measure that instead.  The
    gap is not subtle -- the same outputs drift by 0.03 to 0.20 across the shared
    prefix and by up to 1.09 once the contexts have parted.

    So the agreeing prefix, which the older comparison skips, is exactly where a
    systematic shift is visible, and a bound there is what this adds.
    """
    assert len(outputs_0_lst) == len(outputs_1_lst)

    for prompt_idx, (out_0, out_1) in enumerate(zip(outputs_0_lst, outputs_1_lst)):
        ids_0, text_0, lps_0 = out_0
        ids_1, text_1, lps_1 = out_1
        context = f"Test {prompt_idx}:\n{name_0}: {text_0!r}\n{name_1}: {text_1!r}"

        assert len(ids_0) == len(ids_1), (
            f"{context}\ngenerated {len(ids_0)} and {len(ids_1)} tokens; a "
            "comparison over the shorter of the two would pass on a run that "
            "stopped early"
        )
        assert ids_0, f"{context}\nboth runs generated nothing"

        diverged = next(
            (idx for idx, (a, b) in enumerate(zip(ids_0, ids_1)) if a != b), None
        )
        if diverged is None:
            shared = len(ids_0)
        else:
            shared = diverged + 1
            divergence = (
                f"{context}\nfirst differ at token {diverged}: "
                f"{name_0} chose {ids_0[diverged]}, {name_1} chose "
                f"{ids_1[diverged]}"
            )
            # Each side's choice has to at least be a candidate for the other.
            # A row permutation left undone picks tokens the reference would
            # never rank, which is what this catches.
            assert ids_0[diverged] in lps_1[diverged], divergence
            assert ids_1[diverged] in lps_0[diverged], divergence

        differences = [
            lps_1[idx][tok] - lps_0[idx][tok]
            for idx, tok in enumerate(ids_0[:shared])
            if lps_0 and lps_1 and tok in lps_0[idx] and tok in lps_1[idx]
        ]
        assert differences, f"{context}\nno position was comparable"

        bias = sum(differences) / len(differences)
        assert abs(bias) <= MAX_LOGPROB_BIAS, (
            f"{context}\nmean logprob difference {bias:+.3f} over "
            f"{len(differences)} shared positions exceeds {MAX_LOGPROB_BIAS}; "
            "quantization noise has no direction, so a drift this one-sided "
            "points at a norm offset or a discarded tensor"
        )


def run_multimodal_gguf_test(
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    compare=check_logprobs_close,
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
        compare(
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for multimodal GGUF tests.",
)
@pytest.mark.parametrize(
    "model",
    [pytest.param(MUSE_GLIMMER_CONFIG, marks=pytest.mark.slow)],
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [MAX_TOKENS])
@pytest.mark.parametrize("num_logprobs", [NUM_LOGPROBS])
def test_muse_glimmer_mm_gguf(
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Images only, compared quantitatively rather than for fluency.

    Video is out of scope here rather than covered: the converter keeps only the
    sum of the patch embedding's per-time-step blocks, so it is refused during
    input validation instead, and ``test_plugin.py`` is what holds that gate in
    place.

    The comparison is :func:`check_logprobs_unbiased` because every conversion
    this adapter undoes -- the Q/K row permutation, the norm offset, the
    discarded synthetic Q/K norms, the patch embedding split -- loads cleanly
    when it is wrong and produces fluent, incorrect text.  A comparison that
    tolerates the first divergence cannot tell that apart from quantization
    noise.
    """
    run_multimodal_gguf_test(
        model, dtype, max_tokens, num_logprobs, compare=check_logprobs_unbiased
    )


def _outputs(tokens: list[int], logprob: float = -0.5, top: float = -0.1):
    """One prompt's worth of output, with *tokens* chosen at *logprob*."""
    return [
        (
            tokens,
            "".join(chr(97 + tok % 26) for tok in tokens),
            [{tok: logprob, -1: top} for tok in tokens],
        )
    ]


def test_unbiased_check_accepts_an_identical_run():
    reference = _outputs([1, 2, 3, 4])

    check_logprobs_unbiased(reference, reference, "a", "b")


def test_unbiased_check_accepts_noise_without_a_direction():
    """Q4_K noise is what this comparison has to tolerate."""
    left = _outputs([1, 2, 3, 4], logprob=-0.5)
    right = [
        (
            left[0][0],
            left[0][1],
            [
                {tok: lp, -1: -0.1}
                for tok, lp in zip(left[0][0], (-0.4, -0.6, -0.45, -0.55))
            ],
        )
    ]

    check_logprobs_unbiased(left, right, "a", "b")


def test_unbiased_check_rejects_a_run_that_stopped_early():
    """The condition that let a truncated run pass: zip over the shorter side."""
    with pytest.raises(AssertionError, match="stopped early"):
        check_logprobs_unbiased(_outputs([1, 2, 3, 4]), _outputs([1, 2]), "a", "b")


def test_unbiased_check_rejects_an_empty_run():
    with pytest.raises(AssertionError, match="generated nothing"):
        check_logprobs_unbiased(_outputs([]), _outputs([]), "a", "b")


def test_unbiased_check_rejects_a_one_sided_logprob_shift():
    """The condition that mattered most: same tokens, drifted distribution.

    A norm whose offset was never removed looks exactly like this -- every
    logprob pushed the same way, with the argmax often unchanged.
    """
    left = _outputs([1, 2, 3, 4], logprob=-0.5)
    right = _outputs([1, 2, 3, 4], logprob=-2.0)

    with pytest.raises(AssertionError, match="one-sided"):
        check_logprobs_unbiased(left, right, "a", "b")


def test_unbiased_check_rejects_a_choice_the_reference_would_never_rank():
    """What an undone Q/K row permutation produces at the divergence."""
    left = [([1, 2], "a", [{1: -0.1}, {2: -0.5}])]
    right = [([1, 9], "b", [{1: -0.1}, {9: -0.5}])]

    with pytest.raises(AssertionError, match="first differ at token"):
        check_logprobs_unbiased(left, right, "a", "b")


def test_unbiased_check_ignores_drift_once_the_contexts_have_parted():
    """Pins the boundary, which is the part easiest to "strengthen" wrongly.

    Greedy decoding conditions on what it already emitted, so past the divergence
    the two runs are completing different sentences.  Comparing there reported a
    bias of -1.0 on outputs that were both correct descriptions of the image,
    which is measuring the divergence rather than the weights.
    """
    left = [([1, 2, 3, 4], "a", [{1: -0.1}, {2: -0.5, 9: -0.6}, {3: -0.5}, {4: -0.5}])]
    right = [
        (
            [1, 9, 8, 7],
            "b",
            [{1: -0.1}, {9: -0.5, 2: -0.6}, {8: -0.5, 3: -9.0}, {7: -0.5, 4: -9.0}],
        )
    ]

    check_logprobs_unbiased(left, right, "a", "b")
