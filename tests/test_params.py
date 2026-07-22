# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from vllm.model_executor.utils import set_weight_attrs

from vllm_gguf_plugin.quantization.params import (
    GGUFUninitializedWeightTypeParameter,
    _resolve_gguf_weight_type_loader,
)


class _Layer:
    """Layer stub without weight_loader_v2."""


def _make_qweight_type(num_elements: int):
    param = GGUFUninitializedWeightTypeParameter(requires_grad=False)
    set_weight_attrs(
        param,
        {
            "weight_type": 0,
            "shard_weight_type": {},
            "num_elements": num_elements,
            "ignore_warning": True,
        },
    )
    return param


def _fail_base_loader(*args, **kwargs):
    pytest.fail("base weight loader should not be called")


def test_weight_type_loader_stores_scalar_without_shard_id():
    loader = _resolve_gguf_weight_type_loader(_Layer(), _fail_base_loader)
    param = _make_qweight_type(num_elements=1)

    loader(param, torch.tensor(7))

    assert param.weight_type == 7


def test_weight_type_loader_stores_tuple_shard_ids():
    """Fused GGUF weight type covering multiple shards (e.g. Qwen3.5 GDN
    in_proj_qkv → in_proj_qkvz shards (0, 1, 2))."""
    loader = _resolve_gguf_weight_type_loader(_Layer(), _fail_base_loader)
    param = _make_qweight_type(num_elements=4)

    loader(param, torch.tensor(8), (0, 1, 2))

    assert param.shard_weight_type == {0: 8, 1: 8, 2: 8}
    assert param.weight_type == 8
