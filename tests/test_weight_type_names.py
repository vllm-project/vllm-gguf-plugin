# SPDX-License-Identifier: Apache-2.0

"""Names of the synthetic ``weight_type`` companions the GGUF iterators yield.

The iterators yield a ``<base>.weight_type`` tensor for every quantized
``<base>.weight``. Only the parameter-name suffix may change: a module whose
own name contains ``weight`` must keep it. Qwen3.8-Flash-Next's hyper-connection
projections are named ``input_mix_weight_down.weight``, and replacing every
``weight`` produced ``input_mix_weight_type_down.weight_type``, which no module
owns (loading its GGUF failed on exactly that name).
"""

import gguf
import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFWriter

from vllm_gguf_plugin.weight_utils import (
    gguf_quant_weights_iterator_multi,
    gguf_weight_type_name,
)

_INNER_GGUF = "output_hc_down.weight"
_INNER_NATIVE = "model.hyper_connection_mixer.input_mix_weight_down.weight"
_PLAIN_GGUF = "blk.0.attn_q.weight"
_PLAIN_NATIVE = "model.layers.0.self_attn.q_proj.weight"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight_type",
        ),
        (_INNER_NATIVE, _INNER_NATIVE + "_type"),
        (
            "model.layers.0.attn_hyper_connection.block_inject_weight.weight",
            "model.layers.0.attn_hyper_connection.block_inject_weight.weight_type",
        ),
        (
            "model.layers.0.mlp.experts.w13_weight",
            "model.layers.0.mlp.experts.w13_weight_type",
        ),
        (
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_q.weight_type",
        ),
    ],
)
def test_weight_type_name_changes_only_the_last_weight(name, expected):
    assert gguf_weight_type_name(name) == expected


def test_weight_type_name_rejects_names_without_weight():
    with pytest.raises(ValueError, match="weight"):
        gguf_weight_type_name("model.layers.0.linear_attn.dt_bias")


def _write_q8_gguf(tmp_path, names: list[str]) -> str:
    path = tmp_path / "model.gguf"
    writer = GGUFWriter(str(path), "llama")
    values = np.linspace(-2.0, 2.0, 2 * 32, dtype=np.float32).reshape(2, 32)
    packed = gguf.quants.quantize(values, GGMLQuantizationType.Q8_0)
    for name in names:
        writer.add_tensor(name, packed, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


def test_iterator_names_companions_after_the_mapped_parameter(tmp_path):
    path = _write_q8_gguf(tmp_path, [_INNER_GGUF, _PLAIN_GGUF])
    names = [
        name
        for name, _ in gguf_quant_weights_iterator_multi(
            [path], {_INNER_GGUF: _INNER_NATIVE, _PLAIN_GGUF: _PLAIN_NATIVE}
        )
    ]
    assert names == [
        _INNER_NATIVE + "_type",
        _INNER_NATIVE,
        _PLAIN_NATIVE + "_type",
        _PLAIN_NATIVE,
    ]


def test_diffusion_iterator_names_companions_after_the_parameter(tmp_path):
    from vllm_gguf_plugin.weights_adapter.diffusion.base import (
        gguf_quant_weights_iterator,
    )

    inner = "transformer_blocks.0.attn.gate_weight_proj.weight"
    plain = "transformer_blocks.0.attn.to_q.weight"
    path = _write_q8_gguf(tmp_path, [inner, plain])
    names = [name for name, _ in gguf_quant_weights_iterator(path)]

    assert [name for name in names if name.endswith("_type")] == [
        inner + "_type",
        plain + "_type",
    ]
    assert inner in names and plain in names
