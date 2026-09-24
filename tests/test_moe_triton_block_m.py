# SPDX-License-Identifier: Apache-2.0

"""TRITON_MOE_BLOCK_M_BY_TYPE must govern the launch grid, not just the padding.

``ops.ggml_moe_get_block_size`` reads this table to pad ``sorted_token_ids``
whenever a quant type has no CUDA MMQ kernel. If the launch grid does not use
the same value, ``_validate_args`` sees more token ids than ``expert_ids *
block_m`` and rejects the call -- so a table documented as a per-type override
is usable only by types whose wrapper also passes ``block_m`` by hand, which
today is exactly one of the twenty-three.

No GPU is needed: the resolved value is captured where the runner first uses
it, before any tensor is touched.
"""

import pytest

from vllm_gguf_plugin.triton.fused_moe import utils
from vllm_gguf_plugin.triton.gemm.utils import GGML_TYPE_Q4_0

_UNLISTED = max(utils.TRITON_MOE_BLOCK_M_BY_TYPE) + 1000


class _Stop(Exception):
    """Raised by the stub so the runner returns before allocating anything."""


def test_table_entry_is_reported_by_the_accessor():
    assert (
        utils.get_triton_moe_block_m(GGML_TYPE_Q4_0)
        == utils.TRITON_MOE_BLOCK_M_BY_TYPE[GGML_TYPE_Q4_0]
    )


def test_unlisted_type_falls_back_to_the_module_default():
    assert _UNLISTED not in utils.TRITON_MOE_BLOCK_M_BY_TYPE
    assert utils.get_triton_moe_block_m(_UNLISTED) == utils.TRITON_FUSED_MOE_BLOCK_M


@pytest.mark.parametrize(
    ("quant_type", "expected"),
    [
        (GGML_TYPE_Q4_0, utils.TRITON_MOE_BLOCK_M_BY_TYPE[GGML_TYPE_Q4_0]),
        (_UNLISTED, utils.TRITON_FUSED_MOE_BLOCK_M),
    ],
)
def test_runner_resolves_block_m_from_the_table(monkeypatch, quant_type, expected):
    seen = {}

    def fake_validate(*args):
        seen["block_m"] = args[-1]
        raise _Stop

    monkeypatch.setattr(utils, "_validate_args", fake_validate)
    with pytest.raises(_Stop):
        utils.run_triton_fused_moe_kernel(
            None, None, None, None, None, None, 0, 1, 1, quant_type
        )
    assert seen["block_m"] == expected


def test_explicit_block_m_still_wins(monkeypatch):
    seen = {}

    def fake_validate(*args):
        seen["block_m"] = args[-1]
        raise _Stop

    monkeypatch.setattr(utils, "_validate_args", fake_validate)
    with pytest.raises(_Stop):
        utils.run_triton_fused_moe_kernel(
            None, None, None, None, None, None, 0, 1, 1, GGML_TYPE_Q4_0, block_m=16
        )
    assert seen["block_m"] == 16
