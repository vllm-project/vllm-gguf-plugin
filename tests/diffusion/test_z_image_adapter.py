# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Z-Image diffusion GGUF adapter."""

from __future__ import annotations

import pytest
import torch

from vllm_gguf_plugin.weights_adapter.diffusion import (
    ZImageDiffusionGGUFAdapter,
    get_diffusion_gguf_adapter,
)

pytestmark = [pytest.mark.cpu]


def test_z_image_adapter_selected_for_z_image_family():
    adapter = get_diffusion_gguf_adapter(
        "dummy.gguf",
        model_class_name="ZImagePipeline",
        model_type="z_image",
    )

    assert isinstance(adapter, ZImageDiffusionGGUFAdapter)


def test_z_image_adapter_declares_hf_text_encoder_modules_unquantized():
    assert ZImageDiffusionGGUFAdapter.unquantized_modules == ("model", "lm_head")


def test_z_image_adapter_matches_model_type_variants():
    for model_type in ("z_image", "zimage", "z-image"):
        assert ZImageDiffusionGGUFAdapter.is_compatible(
            model_class_name="OtherPipeline",
            model_type=model_type,
        )


def test_z_image_adapter_renames_known_gguf_tensor_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_gguf_plugin.weights_adapter.diffusion.z_image as z_image_module

    monkeypatch.setattr(
        z_image_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("model.diffusion_model.final_layer.qweight", torch.ones((1, 1))),
                ("model.diffusion_model.x_embedder.qweight_type", torch.tensor(1)),
                ("transformer_blocks.0.attention.out.weight", torch.full((1, 1), 2.0)),
                ("transformer_blocks.0.attention.qkv.qweight", torch.full((1, 1), 3.0)),
                ("context_refiner.0.feed_forward.w1.weight", torch.full((1, 2), 4.0)),
                ("context_refiner.0.feed_forward.w3.weight", torch.full((1, 2), 5.0)),
            ]
        ),
    )

    adapter = ZImageDiffusionGGUFAdapter("dummy.gguf")

    weights = list(adapter.weights_iterator())
    names = [name for name, _ in weights]

    assert "all_final_layer.2-1.qweight" in names
    assert "all_x_embedder.2-1.qweight_type" in names
    assert "transformer_blocks.0.attention.to_out.0.weight" in names
    assert "transformer_blocks.0.attention.to_qkv.qweight" in names
    assert "context_refiner.0.feed_forward.w1.weight" in names
    assert "context_refiner.0.feed_forward.w3.weight" in names
