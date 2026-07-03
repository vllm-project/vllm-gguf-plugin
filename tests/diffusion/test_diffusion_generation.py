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
import threading
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest
import torch

pynvml = pytest.importorskip("pynvml")

vllm_omni = pytest.importorskip("vllm_omni")

from vllm_omni.entrypoints.omni import Omni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402
from vllm_omni.outputs import OmniRequestOutput  # noqa: E402
from vllm_omni.platforms import current_omni_platform  # noqa: E402


class DiffusionGGUFTestConfig(NamedTuple):
    artifact_prefix: str
    gguf_model: str
    hf_model: str
    min_cosine_similarity: float = 0.95


Z_IMAGE_CONFIG = DiffusionGGUFTestConfig(
    artifact_prefix="zimage",
    hf_model="Tongyi-MAI/Z-Image-Turbo",
    gguf_model="unsloth/Z-Image-Turbo-GGUF:Q4_0",
)


FLUX_CONFIG = DiffusionGGUFTestConfig(
    artifact_prefix="flux2_klein",
    hf_model="black-forest-labs/FLUX.2-klein-4B",
    gguf_model="unsloth/FLUX.2-klein-4B-GGUF:Q8_0",
)


def _image_cosine_similarity(hf_image, gguf_image) -> float:
    hf_tensor = torch.as_tensor(
        np.array(hf_image.convert("RGB"), copy=True), dtype=torch.float32
    ).flatten()
    gguf_tensor = torch.as_tensor(
        np.array(gguf_image.convert("RGB"), copy=True), dtype=torch.float32
    ).flatten()
    return torch.nn.functional.cosine_similarity(hf_tensor, gguf_tensor, dim=0).item()


@dataclass
class _GpuMemoryStats:
    baseline_gib: float
    peak_used_gib: float

    @property
    def peak_delta_gib(self) -> float:
        return max(0.0, self.peak_used_gib - self.baseline_gib)


class _GpuMemoryMonitor:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._baseline_bytes = 0
        self._peak_bytes = 0

    def __enter__(self) -> _GpuMemoryMonitor:
        pynvml.nvmlInit()
        device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        used = pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
        self._baseline_bytes = used
        self._peak_bytes = used
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._sample_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample_once()
        pynvml.nvmlShutdown()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        if self._handle is None:
            return
        used = pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
        self._peak_bytes = max(self._peak_bytes, used)

    def stats(self) -> _GpuMemoryStats:
        gib = 1024**3
        return _GpuMemoryStats(
            baseline_gib=self._baseline_bytes / gib,
            peak_used_gib=self._peak_bytes / gib,
        )


def _generate_single_stage_image(
    model: str,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 20,
    seed: int = 42,
    **extra_kwargs,
) -> tuple[list, float]:
    """Generate an image with a single-stage diffusion model.

    Returns (images, peak_memory_gib).
    """
    omni_kwargs = dict(extra_kwargs)

    with _GpuMemoryMonitor() as memory_monitor:
        omni = Omni(model, **omni_kwargs)
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
    peak_mem = memory_monitor.stats().peak_delta_gib

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
@pytest.mark.parametrize(
    "model", [Z_IMAGE_CONFIG, FLUX_CONFIG], ids=["Z-Image-Turbo", "FLUX.2-klein"]
)
def test_single_stage_diffusion_gguf(
    model: DiffusionGGUFTestConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Z-Image-Turbo GGUF generates valid images and uses less memory than BF16."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # BF16 baseline
    hf_images, mem_bf16 = _generate_single_stage_image(
        model=model.hf_model,
    )

    # GGUF
    images, mem_gguf = _generate_single_stage_image(
        model=model.hf_model,
        diffusion_quantization_config={
            "method": "gguf",
            "gguf_model": model.gguf_model,
        },
    )

    hf_image_path = f"test_{model.artifact_prefix}_hf.png"
    gguf_image_path = f"test_{model.artifact_prefix}_gguf.png"
    assert len(hf_images) >= 1
    hf_images[0].save(hf_image_path)
    assert len(images) >= 1
    images[0].save(gguf_image_path)
    print(f"Saved HF image: {hf_image_path}")
    print(f"Saved GGUF image: {gguf_image_path}")

    image_cosine_similarity = _image_cosine_similarity(hf_images[0], images[0])
    print(
        f"{model.artifact_prefix} image cosine similarity: "
        f"{image_cosine_similarity:.4f}"
    )
    assert image_cosine_similarity >= model.min_cosine_similarity, (
        f"GGUF image cosine similarity ({image_cosine_similarity:.4f}) should be "
        f">= {model.min_cosine_similarity:.4f}"
    )

    print(f"{model.artifact_prefix} BF16 peak VRAM delta: {mem_bf16:.2f} GiB")
    print(f"{model.artifact_prefix} GGUF peak VRAM delta: {mem_gguf:.2f} GiB")
    reduction = (mem_bf16 - mem_gguf) / mem_bf16 * 100
    print(f"VRAM reduction: {reduction:.1f}%")
    assert mem_gguf < mem_bf16, (
        f"GGUF ({mem_gguf:.2f} GiB) should use less VRAM than BF16 ({mem_bf16:.2f} GiB)"
    )
