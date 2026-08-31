# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_gguf_plugin.weights_adapter.gemma4 import Gemma4GGUFAdapter


def test_gemma4_name_mapping():
    assert Gemma4GGUFAdapter.map_name("blk.2.ffn_gate_inp.scale") == (
        "model.language_model.layers.2.router.scale"
    )
    assert Gemma4GGUFAdapter.map_name("blk.2.layer_output_scale.weight") == (
        "model.language_model.layers.2.layer_scalar"
    )
    assert Gemma4GGUFAdapter.map_name("v.blk.3.attn_q.weight") == (
        "model.vision_tower.encoder.layers.3.self_attn.q_proj.linear.weight"
    )
    assert Gemma4GGUFAdapter.map_name("v.position_embd.weight") == (
        "model.vision_tower.patch_embedder.position_embedding_table"
    )
    assert Gemma4GGUFAdapter.map_name("v.std_bias") == ("model.vision_tower.std_bias")
    assert Gemma4GGUFAdapter.map_name("v.std_scale") == ("model.vision_tower.std_scale")
    assert Gemma4GGUFAdapter.map_name("mm.input_projection.weight") == (
        "model.embed_vision.embedding_projection.weight"
    )
    assert Gemma4GGUFAdapter.map_name("rope_freqs.weight") is None


def test_gemma4_splits_packed_gate_up_experts():
    weight = torch.arange(16).reshape(2, 4, 2)
    name = "model.language_model.layers.0.experts.0.gate_up_proj.qweight"

    mapped = list(Gemma4GGUFAdapter._split_expert_weights(name, weight))

    assert [item[0] for item in mapped] == [
        "model.language_model.layers.0.experts.0.gate_proj.qweight",
        "model.language_model.layers.0.experts.0.up_proj.qweight",
        "model.language_model.layers.0.experts.1.gate_proj.qweight",
        "model.language_model.layers.0.experts.1.up_proj.qweight",
    ]
    assert torch.equal(mapped[0][1], weight[0, :2])
    assert torch.equal(mapped[1][1], weight[0, 2:])
