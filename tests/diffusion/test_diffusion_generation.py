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

import pytest
import torch

pynvml = pytest.importorskip("pynvml")

vllm_omni = pytest.importorskip("vllm_omni")

from vllm_omni.entrypoints.omni import Omni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402
from vllm_omni.outputs import OmniRequestOutput  # noqa: E402
from vllm_omni.platforms import current_omni_platform  # noqa: E402

GGUF_REPO = "/mnt/data0/LLM"
GGUF_FILENAME = "z-image-turbo-Q4_0.gguf"
GGUF_MODEL_REF = f"{GGUF_REPO}/{GGUF_FILENAME}"
HF_MODEL = "/mnt/data0/LLM/Z-Image-Turbo"


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

    def __enter__(self) -> "_GpuMemoryMonitor":
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
def test_single_stage_zimage_gguf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Z-Image-Turbo GGUF generates valid images and uses less memory than BF16."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # BF16 baseline
    hf_images, mem_bf16 = _generate_single_stage_image(
        model=HF_MODEL,
    )

    # GGUF
    images, mem_gguf = _generate_single_stage_image(
        model=HF_MODEL,
        diffusion_quantization_config={
            "method": "gguf",
            "gguf_model": GGUF_MODEL_REF,
        },
    )

    assert len(hf_images) >= 1
    hf_images[0].save("test_zimage_hf.png")
    assert len(images) >= 1
    images[0].save("test_zimage_gguf.png")
    print("Saved HF image: test_zimage_hf.png")
    print("Saved GGUF image: test_zimage_gguf.png")

    print(f"Z-Image BF16 peak VRAM delta: {mem_bf16:.2f} GiB")
    print(f"Z-Image GGUF peak VRAM delta: {mem_gguf:.2f} GiB")
    reduction = (mem_bf16 - mem_gguf) / mem_bf16 * 100
    print(f"VRAM reduction: {reduction:.1f}%")
    assert mem_gguf < mem_bf16, (
        f"GGUF ({mem_gguf:.2f} GiB) should use less VRAM than "
        f"BF16 ({mem_bf16:.2f} GiB)"
    )
