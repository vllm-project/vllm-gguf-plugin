# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniMax-H3 diffusion GGUF adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gguf_plugin.weights_adapter.diffusion import (
    MiniMaxH3DiffusionGGUFAdapter,
    get_diffusion_gguf_adapter,
)

pytestmark = [pytest.mark.cpu]


def test_minimax_h3_adapter_selected_for_pipeline():
    adapter = get_diffusion_gguf_adapter(
        "dummy.gguf",
        model_class_name="MiniMaxH3Pipeline",
        model_type=None,
    )
    assert isinstance(adapter, MiniMaxH3DiffusionGGUFAdapter)
    assert adapter.unquantized_modules == ("text_encoder",)


@pytest.mark.parametrize("model_type", ["minimax_h3", "minimax-h3", "minimaxh3"])
def test_minimax_h3_adapter_selected_for_model_type(model_type: str):
    assert MiniMaxH3DiffusionGGUFAdapter.is_compatible(
        model_class_name=None,
        model_type=model_type,
    )


def test_quantized_qkv_keeps_converter_fused_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_gguf_plugin.weights_adapter.diffusion.minimax_h3 as h3_module

    qweight = torch.arange(12).reshape(3, 4)
    monkeypatch.setattr(
        h3_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("blocks.0.attn.qkv_proj.qweight_type", torch.tensor(10)),
                ("blocks.0.attn.qkv_proj.qweight", qweight),
            ]
        ),
    )

    weights = dict(MiniMaxH3DiffusionGGUFAdapter("dummy.gguf").weights_iterator())

    assert weights["blocks.0.attn.qkv_proj.qweight_type"].item() == 10
    assert weights["blocks.0.attn.qkv_proj.qweight"] is qweight


def test_dense_qkv_is_restored_to_grouped_checkpoint_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_gguf_plugin.weights_adapter.diffusion.minimax_h3 as h3_module

    projection_rows = 56 * 128
    fused = torch.arange(3 * projection_rows).reshape(-1, 1)
    monkeypatch.setattr(
        h3_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter([("blocks.0.attn.qkv_proj.weight", fused)]),
    )

    weights = dict(MiniMaxH3DiffusionGGUFAdapter("dummy.gguf").weights_iterator())
    grouped = weights["blocks.0.attn.qkv_proj.weight"].reshape(56, 3 * 128)

    assert torch.equal(grouped[0, :128], fused[:128, 0])
    assert torch.equal(
        grouped[0, 128:256], fused[projection_rows : projection_rows + 128, 0]
    )
    assert torch.equal(
        grouped[0, 256:], fused[2 * projection_rows : 2 * projection_rows + 128, 0]
    )


def test_pruned_adaln_layout_is_rejected_early(monkeypatch: pytest.MonkeyPatch):
    import vllm_gguf_plugin.weights_adapter.diffusion.minimax_h3 as h3_module

    reader = SimpleNamespace(tensors=[SimpleNamespace(name="adaln_t_table")])
    monkeypatch.setattr(h3_module.gguf, "GGUFReader", lambda _path: reader)

    adapter = MiniMaxH3DiffusionGGUFAdapter("pruned.gguf")
    with pytest.raises(ValueError, match="Unsupported MiniMax-H3 GGUF schema"):
        adapter.unquantized_module_names()


def test_incomplete_time_embedder_layout_is_rejected_early(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_gguf_plugin.weights_adapter.diffusion.minimax_h3 as h3_module

    reader = SimpleNamespace(
        tensors=[
            SimpleNamespace(
                name="time_embedder.proj_in.weight",
                tensor_type=SimpleNamespace(name="F32"),
            )
        ]
    )
    monkeypatch.setattr(h3_module.gguf, "GGUFReader", lambda _path: reader)

    adapter = MiniMaxH3DiffusionGGUFAdapter("incomplete.gguf")
    with pytest.raises(ValueError, match="Unsupported MiniMax-H3 GGUF schema"):
        adapter.unquantized_module_names()
