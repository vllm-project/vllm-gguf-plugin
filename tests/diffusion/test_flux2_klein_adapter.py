# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Flux2-Klein diffusion GGUF adapter."""

from __future__ import annotations

import pytest
import torch

from vllm_gguf_plugin.weights_adapter.diffusion import (
    Flux2KleinDiffusionGGUFAdapter,
    get_diffusion_gguf_adapter,
)

pytestmark = [pytest.mark.cpu]


def test_flux2_klein_adapter_selected_for_flux_family():
    adapter = get_diffusion_gguf_adapter(
        "dummy.gguf",
        model_class_name="Flux2Pipeline",
        model_type="flux",
    )

    assert isinstance(adapter, Flux2KleinDiffusionGGUFAdapter)


def test_flux2_klein_adapter_matches_flux_model_type():
    assert Flux2KleinDiffusionGGUFAdapter.is_compatible(
        model_class_name="OtherPipeline",
        model_type="flux-dev",
    )


def test_flux2_klein_adapter_renames_core_projection_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_gguf_plugin.weights_adapter.diffusion.flux2_klein as flux_module

    monkeypatch.setattr(
        flux_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                ("img_in.qweight", torch.ones((1, 1))),
                ("time_in.in_layer.qweight_type", torch.tensor(1)),
                (
                    "double_blocks.0.img_attn.norm.query_norm.weight",
                    torch.full((1, 1), 2.0),
                ),
                ("final_layer.linear.qweight", torch.full((1, 1), 3.0)),
            ]
        ),
    )

    adapter = Flux2KleinDiffusionGGUFAdapter("dummy.gguf")

    weights = list(adapter.weights_iterator())
    names = [name for name, _ in weights]

    assert "x_embedder.qweight" in names
    assert "time_guidance_embed.timestep_embedder.linear_1.qweight_type" in names
    assert "transformer_blocks.0.attn.norm_q.weight" in names
    assert "proj_out.qweight" in names


def test_flux2_klein_adapter_swaps_final_adaln_shift_and_scale(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_gguf_plugin.weights_adapter.diffusion.flux2_klein as flux_module

    monkeypatch.setattr(
        flux_module,
        "gguf_quant_weights_iterator",
        lambda _path: iter(
            [
                (
                    "final_layer.adaLN_modulation.1.scale",
                    torch.tensor([1.0, 2.0, 3.0, 4.0]),
                )
            ]
        ),
    )

    adapter = Flux2KleinDiffusionGGUFAdapter("dummy.gguf")

    weights = list(adapter.weights_iterator())

    assert len(weights) == 1
    assert weights[0][0] == "norm_out.linear.weight"
    assert torch.equal(weights[0][1], torch.tensor([3.0, 4.0, 1.0, 2.0]))
