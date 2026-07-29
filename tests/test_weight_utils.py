# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from gguf.quants import GGMLQuantizationType

from vllm_gguf_plugin import weight_utils


def _fake_tensor(name: str, tensor_type=GGMLQuantizationType.Q4_0):
    return SimpleNamespace(
        name=name,
        tensor_type=tensor_type,
        data=np.zeros(4, dtype=np.uint8),
    )


def test_dequant_suffix_matches_whole_path_segments():
    """``output.weight`` must not drag every ``attn_output.weight`` along: a
    dequantized tensor for a GGUF-quantized module loads as a plain param the
    module has no slot for."""
    reader = SimpleNamespace(
        byte_order="L",
        tensors=[
            _fake_tensor("output.weight"),
            _fake_tensor("blk.3.attn_output.weight"),
        ],
    )
    with (
        patch.object(weight_utils.gguf, "GGUFReader", return_value=reader),
        patch.object(
            weight_utils.gguf.quants,
            "dequantize",
            return_value=np.zeros(4, dtype=np.float32),
        ),
    ):
        names = [
            name
            for name, _ in weight_utils.gguf_quant_weights_iterator_multi(
                ["model.gguf"], dequant_suffixes=("output.weight",)
            )
        ]

    assert names == [
        "output.weight",
        "blk.3.attn_output.qweight_type",
        "blk.3.attn_output.qweight",
    ]
