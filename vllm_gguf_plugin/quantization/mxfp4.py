# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

try:
    from vllm.model_executor.layers.quantization.mxfp4 import (
        GptOssMxfp4MoEMethod,
        Mxfp4MoEMethod,
    )
except ImportError:  # pragma: no cover - only for older vLLM builds.
    GptOssMxfp4MoEMethod = None
    Mxfp4MoEMethod = None

GGUF_MXFP4_BLOCK_SIZE = 32
GGUF_MXFP4_BLOCK_BYTES = 17
GGUF_MXFP4_SCALE_BYTES = 1


class _GptOssMxfp4LayerQuantConfig:
    def __init__(self, gguf_quant_config):
        self.gguf_quant_config = gguf_quant_config

    def get_name(self):
        return "gpt_oss_mxfp4"

    def __getattr__(self, name):
        return getattr(self.gguf_quant_config, name)


def _as_mxfp4_blocks(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.dtype != torch.uint8:
        qweight = qweight.view(torch.uint8)
    if qweight.ndim >= 3 and qweight.shape[-1] == GGUF_MXFP4_BLOCK_BYTES:
        return qweight.contiguous()
    if qweight.ndim < 2:
        raise ValueError(
            "Expected a row-packed or block-packed GGUF MXFP4 tensor, "
            f"got {qweight.ndim} dimensions."
        )
    if qweight.shape[-1] % GGUF_MXFP4_BLOCK_BYTES != 0:
        raise ValueError(
            "Expected GGUF MXFP4 row byte width to be divisible by "
            f"{GGUF_MXFP4_BLOCK_BYTES}, got {qweight.shape[-1]}."
        )
    blocks_per_row = qweight.shape[-1] // GGUF_MXFP4_BLOCK_BYTES
    return qweight.reshape(*qweight.shape[:-1], blocks_per_row, GGUF_MXFP4_BLOCK_BYTES)


def split_gguf_mxfp4_weight(qweight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one GGUF MXFP4 tensor into vLLM native weight and scale tensors."""
    blocks = _as_mxfp4_blocks(qweight)
    leading_shape = blocks.shape[:-2]
    blocks_per_row = blocks.shape[-2]
    scale_bytes = blocks[..., :GGUF_MXFP4_SCALE_BYTES]
    packed_values = blocks[..., GGUF_MXFP4_SCALE_BYTES:]
    weight = packed_values.reshape(
        *leading_shape, blocks_per_row * (GGUF_MXFP4_BLOCK_SIZE // 2)
    )
    weight_scale = scale_bytes.reshape(
        *leading_shape, blocks_per_row * GGUF_MXFP4_SCALE_BYTES
    )
    return weight.contiguous(), weight_scale.contiguous()


def split_gguf_mxfp4_moe_weight(
    qweight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a GGUF MXFP4 expert tensor while preserving expert/row dimensions."""
    if qweight.ndim < 3:
        return split_gguf_mxfp4_weight(qweight)
    if qweight.ndim >= 4 and qweight.shape[-1] == GGUF_MXFP4_BLOCK_BYTES:
        leading_shape = qweight.shape[:-2]
        flat_qweight = qweight.reshape(-1, qweight.shape[-2], GGUF_MXFP4_BLOCK_BYTES)
        weight, weight_scale = split_gguf_mxfp4_weight(flat_qweight)
        return (
            weight.reshape(*leading_shape, weight.shape[-1]).contiguous(),
            weight_scale.reshape(*leading_shape, weight_scale.shape[-1]).contiguous(),
        )
    if qweight.shape[-1] % GGUF_MXFP4_BLOCK_BYTES != 0:
        return split_gguf_mxfp4_weight(qweight)

    leading_shape = qweight.shape[:-1]
    flat_qweight = qweight.reshape(-1, qweight.shape[-1])
    weight, weight_scale = split_gguf_mxfp4_weight(flat_qweight)
    return (
        weight.reshape(*leading_shape, weight.shape[-1]).contiguous(),
        weight_scale.reshape(*leading_shape, weight_scale.shape[-1]).contiguous(),
    )


def iter_gguf_mxfp4_native_moe_weights(
    module_name: str,
    qweight: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    weight, weight_scale = split_gguf_mxfp4_moe_weight(qweight)
    return [
        (f"{module_name}.weight", weight),
        (f"{module_name}.weight_scale", weight_scale),
    ]


if Mxfp4MoEMethod is None:

    class GGUFMxfp4FusedMoE:  # pragma: no cover - only for older vLLM builds.
        """Native vLLM MXFP4 MoE method for GGUF MXFP4 expert tensors."""

        def __init__(self, gguf_quant_config, moe_config):
            del gguf_quant_config, moe_config
            raise RuntimeError("This vLLM build does not provide Mxfp4MoEMethod.")


else:

    class GGUFMxfp4FusedMoE(Mxfp4MoEMethod):
        """Native vLLM MXFP4 MoE method for GGUF MXFP4 expert tensors."""

        def __init__(self, gguf_quant_config, moe_config):
            del gguf_quant_config
            super().__init__(moe_config)


if GptOssMxfp4MoEMethod is None:

    class GGUFGptOssMxfp4FusedMoE:  # pragma: no cover
        """Native vLLM GPT-OSS MXFP4 MoE method for GGUF expert tensors."""

        def __init__(self, gguf_quant_config, moe_config):
            del gguf_quant_config, moe_config
            raise RuntimeError("This vLLM build does not provide GptOssMxfp4MoEMethod.")


else:

    class GGUFGptOssMxfp4FusedMoE(GptOssMxfp4MoEMethod):
        """Native vLLM GPT-OSS MXFP4 MoE method for GGUF expert tensors."""

        def __init__(self, gguf_quant_config, moe_config):
            self.gguf_quant_config = gguf_quant_config
            del gguf_quant_config
            super().__init__(moe_config)

        def create_weights(self, layer, *args, **kwargs):
            super().create_weights(layer, *args, **kwargs)
            layer.quant_config = _GptOssMxfp4LayerQuantConfig(self.gguf_quant_config)
