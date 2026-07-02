# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for GGUF quantization on diffusion models.

Validates that GGUF-quantized diffusion models generate valid images and
use less peak GPU memory than BF16 baseline.

Requires vllm-omni to be installed alongside the plugin.

Usage:
    pytest tests/diffusion/test_gguf_memory.py -v
"""

from __future__ import annotations

import gc

import pytest
import torch

vllm_omni = pytest.importorskip("vllm_omni")

from vllm_omni.entrypoints.omni import Omni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402
from vllm_omni.outputs import OmniRequestOutput  # noqa: E402
from vllm_omni.platforms import current_omni_platform  # noqa: E402

GGUF_REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF_FILENAME = "z-image-turbo-Q4_0.gguf"
GGUF_MODEL_REF = f"{GGUF_REPO}/{GGUF_FILENAME}"
HF_MODEL = "Tongyi-MAI/Z-Image-Turbo"


def _generate_single_stage_image(
    model: str,
    quantization: str | None = None,
    load_format: str = "auto",
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 2,
    seed: int = 42,
    **extra_kwargs,
) -> tuple[list, float]:
    """Generate an image with a single-stage diffusion model.

    Returns (images, peak_memory_gib).
    """
    omni_kwargs = dict(extra_kwargs)
    if quantization:
        omni_kwargs["quantization"] = quantization
    omni_kwargs["load_format"] = load_format

    omni = Omni(model, **omni_kwargs)
    torch.accelerator.reset_peak_memory_stats()

    generator = torch.Generator(
        device=current_omni_platform.device_type,
    ).manual_seed(seed)
    outputs = omni.generate(
        "a photo of a cat sitting on a laptop keyboard",
        OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
        ),
    )

    peak_mem = torch.accelerator.max_memory_allocated() / (1024**3)

    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if hasattr(first_output, "images") and first_output.images:
        images = first_output.images
    else:
        assert hasattr(first_output, "request_output") and first_output.request_output
        request_output = first_output.request_output
        if isinstance(request_output, list):
            req_out = request_output[0]
        else:
            req_out = request_output
        assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
        images = req_out.images
    assert len(images) >= 1
    assert images[0].width == width
    assert images[0].height == height

    omni.shutdown()
    del omni
    gc.collect()
    torch.accelerator.empty_cache()

    return images, peak_mem


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.slow
def test_single_stage_zimage_gguf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Z-Image-Turbo GGUF generates valid images and uses less memory than BF16."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # BF16 baseline
    _, mem_bf16 = _generate_single_stage_image(
        model=HF_MODEL,
    )

    # GGUF
    images, mem_gguf = _generate_single_stage_image(
        model=GGUF_MODEL_REF,
        quantization="gguf",
        load_format="gguf",
    )

    assert len(images) >= 1
    images[0].save("test_zimage_gguf.png")

    print(f"Z-Image BF16 peak memory: {mem_bf16:.2f} GiB")
    print(f"Z-Image GGUF peak memory: {mem_gguf:.2f} GiB")
    reduction = (mem_bf16 - mem_gguf) / mem_bf16 * 100
    print(f"Memory reduction: {reduction:.1f}%")
    assert mem_gguf < mem_bf16, (
        f"GGUF ({mem_gguf:.2f} GiB) should use less memory than "
        f"BF16 ({mem_bf16:.2f} GiB)"
    )
