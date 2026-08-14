# SPDX-License-Identifier: Apache-2.0

import regex
from transformers import PretrainedConfig

from vllm_gguf_plugin.weights_adapter import (
    OLMoEGGUFAdapter,
    get_weights_adapter,
)


def test_olmoe_adapter():
    config = PretrainedConfig(
        model_type="olmoe",
        num_hidden_layers=1,
    )
    adapter = get_weights_adapter(config)
    assert isinstance(adapter, OLMoEGGUFAdapter)

    gguf_to_hf_name_map, sideload_params = adapter._get_model_specific_mapping(
        config
    )

    assert gguf_to_hf_name_map["blk.0.ffn_down_exps.weight"] == (
        "model.layers.0.mlp.experts.0.down_proj.weight"
    )
    assert len(gguf_to_hf_name_map) == 3
    assert any(
        regex.fullmatch(
            pattern,
            "model.layers.0.mlp.experts.1.down_proj.weight",
        )
        for pattern in sideload_params
    )
