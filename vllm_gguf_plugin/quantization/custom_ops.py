# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def register_or_get_vllm_custom_op(
    *,
    op_name: str,
    op_func: Callable[..., Any],
    fake_impl: Callable[..., Any],
) -> Any:
    """Register a vLLM custom op, or reuse an identical pre-existing op.

    vLLM builds that still carry in-tree GGUF may have already registered the
    same `torch.ops.vllm` operator names. Reusing that operator keeps explicit
    override-mode smoke tests possible without weakening the default safe no-op
    behavior on in-tree builds.
    """
    try:
        direct_register_custom_op(
            op_name=op_name,
            op_func=op_func,
            fake_impl=fake_impl,
        )
    except RuntimeError as exc:
        if (
            "same name and overload name" not in str(exc)
            and "already" not in str(exc).lower()
        ) or not hasattr(torch.ops.vllm, op_name):
            raise
    return getattr(torch.ops.vllm, op_name)
