# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Qwen-Image diffusion GGUF adapter."""

from __future__ import annotations

import pytest

from vllm_gguf_plugin.weights_adapter.diffusion import (
    QwenImageDiffusionGGUFAdapter,
    get_diffusion_gguf_adapter,
)

pytestmark = [pytest.mark.cpu]


def test_qwen_adapter_selected_for_qwen_image_family():
    adapter = get_diffusion_gguf_adapter(
        "dummy.gguf",
        model_class_name="QwenImagePipeline",
        model_type="qwen_image",
    )
    assert isinstance(adapter, QwenImageDiffusionGGUFAdapter)


def test_qwen_adapter_matches_multiple_pipeline_variants():
    for model_class_name in (
        "QwenImagePipeline",
        "QwenImageEditPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageLayeredPipeline",
    ):
        assert QwenImageDiffusionGGUFAdapter.is_compatible(
            model_class_name=model_class_name,
            model_type=None,
        )


def test_qwen_adapter_preserves_split_projection_names(monkeypatch: pytest.MonkeyPatch):
    import vllm_gguf_plugin.weights_adapter.diffusion.qwen_image as qwen_image_module

    monkeypatch.setattr(
        qwen_image_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("transformer_blocks.0.attn.to_q.qweight_type", 1),
                ("transformer_blocks.0.attn.to_q.qweight", 2),
                ("transformer_blocks.0.attn.to_k.qweight_type", 3),
                ("transformer_blocks.0.attn.to_k.qweight", 4),
                ("transformer_blocks.0.attn.to_out.0.qweight_type", 5),
                ("transformer_blocks.0.attn.to_out.0.qweight", 6),
            ]
        ),
    )

    adapter = QwenImageDiffusionGGUFAdapter("dummy.gguf")

    weights = list(adapter.weights_iterator())

    assert ("transformer_blocks.0.attn.to_q.qweight_type", 1) in weights
    assert ("transformer_blocks.0.attn.to_q.qweight", 2) in weights
    assert ("transformer_blocks.0.attn.to_k.qweight_type", 3) in weights
    assert ("transformer_blocks.0.attn.to_k.qweight", 4) in weights
    assert ("transformer_blocks.0.attn.to_out.0.qweight_type", 5) in weights
    assert ("transformer_blocks.0.attn.to_out.0.qweight", 6) in weights


def test_qwen_adapter_keeps_top_level_quantized_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_gguf_plugin.weights_adapter.diffusion.qwen_image as qwen_image_module

    monkeypatch.setattr(
        qwen_image_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("img_in.qweight_type", 1),
                ("img_in.qweight", 2),
                ("time_text_embed.timestep_embedder.linear_1.qweight_type", 3),
                ("time_text_embed.timestep_embedder.linear_1.qweight", 4),
            ]
        ),
    )

    adapter = QwenImageDiffusionGGUFAdapter("dummy.gguf")

    weights = list(adapter.weights_iterator())

    assert ("img_in.qweight_type", 1) in weights
    assert ("img_in.qweight", 2) in weights
    assert ("time_text_embed.timestep_embedder.linear_1.qweight_type", 3) in weights
    assert ("time_text_embed.timestep_embedder.linear_1.qweight", 4) in weights


def test_qwen_adapter_keeps_modulation_quantized_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_gguf_plugin.weights_adapter.diffusion.qwen_image as qwen_image_module

    monkeypatch.setattr(
        qwen_image_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("transformer_blocks.0.img_mod.1.qweight_type", 1),
                ("transformer_blocks.0.img_mod.1.qweight", 2),
                ("transformer_blocks.0.txt_mod.1.qweight_type", 3),
                ("transformer_blocks.0.txt_mod.1.qweight", 4),
            ]
        ),
    )

    adapter = QwenImageDiffusionGGUFAdapter("dummy.gguf")

    weights = list(adapter.weights_iterator())

    assert ("transformer_blocks.0.img_mod.1.qweight_type", 1) in weights
    assert ("transformer_blocks.0.img_mod.1.qweight", 2) in weights
    assert ("transformer_blocks.0.txt_mod.1.qweight_type", 3) in weights
    assert ("transformer_blocks.0.txt_mod.1.qweight", 4) in weights
