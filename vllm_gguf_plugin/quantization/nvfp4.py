# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

GGUF_NVFP4_BLOCK_SIZE = 64
GGUF_NVFP4_BLOCK_BYTES = 36
GGUF_NVFP4_SCALE_BYTES = 4
GGUF_NVFP4_GROUP_SIZE = 16


def _as_nvfp4_blocks(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.dtype != torch.uint8:
        qweight = qweight.view(torch.uint8)
    if qweight.ndim >= 3 and qweight.shape[-1] == GGUF_NVFP4_BLOCK_BYTES:
        return qweight.contiguous()
    if qweight.ndim < 2:
        raise ValueError(
            "Expected a row-packed or block-packed GGUF NVFP4 tensor, "
            f"got {qweight.ndim} dimensions."
        )
    if qweight.shape[-1] % GGUF_NVFP4_BLOCK_BYTES != 0:
        raise ValueError(
            "Expected GGUF NVFP4 row byte width to be divisible by "
            f"{GGUF_NVFP4_BLOCK_BYTES}, got {qweight.shape[-1]}."
        )
    blocks_per_row = qweight.shape[-1] // GGUF_NVFP4_BLOCK_BYTES
    return qweight.reshape(*qweight.shape[:-1], blocks_per_row, GGUF_NVFP4_BLOCK_BYTES)


def gguf_ue4m3_to_fp8_e4m3fn(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode GGUF UE4M3 scales and re-encode as torch FP8 E4M3FN."""
    scale_bytes = scale_bytes.to(torch.uint8)
    exp = ((scale_bytes >> 3) & 0x0F).to(torch.int32)
    man = (scale_bytes & 0x07).to(torch.float32)
    raw = torch.where(
        exp == 0,
        man * (2.0**-9),
        (1.0 + man / 8.0) * torch.pow(2.0, exp.to(torch.float32) - 7.0),
    )
    scale = torch.where(
        (scale_bytes == 0) | (scale_bytes == 0x7F),
        torch.zeros_like(raw),
        raw,
    )
    return scale.to(torch.float8_e4m3fn)


def split_gguf_nvfp4_weight(qweight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one GGUF NVFP4 tensor into vLLM native weight and scale tensors."""
    blocks = _as_nvfp4_blocks(qweight)
    leading_shape = blocks.shape[:-2]
    blocks_per_row = blocks.shape[-2]
    scale_bytes = blocks[..., :GGUF_NVFP4_SCALE_BYTES]
    packed_values = blocks[..., GGUF_NVFP4_SCALE_BYTES:].reshape(
        *leading_shape, blocks_per_row, GGUF_NVFP4_SCALE_BYTES, 8
    )
    low = packed_values & 0x0F
    high = packed_values >> 4
    values = torch.cat((low, high), dim=-1)
    packed_values = values[..., 0::2] | (values[..., 1::2] << 4)

    weight = packed_values.reshape(
        *leading_shape, blocks_per_row * (GGUF_NVFP4_BLOCK_SIZE // 2)
    )
    weight_scale = gguf_ue4m3_to_fp8_e4m3fn(
        scale_bytes.reshape(*leading_shape, blocks_per_row * GGUF_NVFP4_SCALE_BYTES)
    )
    return weight.contiguous(), weight_scale.contiguous()


def split_gguf_nvfp4_moe_weight(
    qweight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a GGUF NVFP4 expert tensor while preserving expert/row dimensions."""
    if qweight.ndim < 3:
        return split_gguf_nvfp4_weight(qweight)
    if qweight.ndim >= 4 and qweight.shape[-1] == GGUF_NVFP4_BLOCK_BYTES:
        leading_shape = qweight.shape[:-2]
        flat_qweight = qweight.reshape(-1, qweight.shape[-2], GGUF_NVFP4_BLOCK_BYTES)
        weight, weight_scale = split_gguf_nvfp4_weight(flat_qweight)
        return (
            weight.reshape(*leading_shape, weight.shape[-1]).contiguous(),
            weight_scale.reshape(*leading_shape, weight_scale.shape[-1]).contiguous(),
        )
    if qweight.shape[-1] % GGUF_NVFP4_BLOCK_BYTES != 0:
        return split_gguf_nvfp4_weight(qweight)

    leading_shape = qweight.shape[:-1]
    flat_qweight = qweight.reshape(-1, qweight.shape[-1])
    weight, weight_scale = split_gguf_nvfp4_weight(flat_qweight)
    return (
        weight.reshape(*leading_shape, weight.shape[-1]).contiguous(),
        weight_scale.reshape(*leading_shape, weight_scale.shape[-1]).contiguous(),
    )


def iter_gguf_nvfp4_native_weights(
    module_name: str,
    qweight: torch.Tensor,
    include_weight_scale_2: bool = True,
) -> list[tuple[str, torch.Tensor]]:
    weight, weight_scale = split_gguf_nvfp4_weight(qweight)
    native_weights: list[tuple[str, torch.Tensor]] = [
        (f"{module_name}.weight", weight),
        (f"{module_name}.weight_scale", weight_scale),
    ]
    if include_weight_scale_2:
        native_weights.append(
            (f"{module_name}.weight_scale_2", torch.tensor(1.0, dtype=torch.float32))
        )
    return native_weights


def _moe_expert_module_name(module_name: str, expert_id: int) -> str:
    marker = ".experts.0."
    if marker not in module_name:
        return module_name
    return module_name.replace(marker, f".experts.{expert_id}.", 1)


def iter_gguf_nvfp4_native_moe_weights(
    module_name: str,
    qweight: torch.Tensor,
    default_sidecar_suffixes: Collection[str] = ("weight_scale_2", "input_scale"),
) -> list[tuple[str, torch.Tensor]]:
    weight, weight_scale = split_gguf_nvfp4_moe_weight(qweight)
    native_weights: list[tuple[str, torch.Tensor]] = [
        (f"{module_name}.weight", weight),
        (f"{module_name}.weight_scale", weight_scale),
    ]

    num_experts = qweight.shape[0] if qweight.ndim >= 3 else 1
    for expert_id in range(num_experts):
        expert_module_name = _moe_expert_module_name(module_name, expert_id)
        for suffix in ("weight_scale_2", "input_scale"):
            if suffix in default_sidecar_suffixes:
                native_weights.append(
                    (
                        f"{expert_module_name}.{suffix}",
                        torch.tensor(1.0, dtype=torch.float32),
                    )
                )
    return native_weights


def iter_gguf_nvfp4_native_moe_sidecar_weights(
    module_name: str,
    suffix: str,
    values: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    """Expand per-expert GGUF NVFP4 sidecar vectors into native scalar loads."""
    values = values.reshape(-1).to(torch.float32)
    return [
        (
            f"{_moe_expert_module_name(module_name, expert_id)}.{suffix}",
            value.reshape(()),
        )
        for expert_id, value in enumerate(values)
    ]


class GGUFModelOptNvFp4FusedMoE(ModelOptNvFp4FusedMoE):
    """Native vLLM NVFP4 W4A16 MoE method for GGUF NVFP4 expert tensors."""

    def __init__(self, gguf_quant_config, moe_config):
        del gguf_quant_config
        modelopt_config = ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            group_size=GGUF_NVFP4_GROUP_SIZE,
        )
        super().__init__(quant_config=modelopt_config, moe_config=moe_config)


@register_weight_loader_v2_supported_method
class GGUFNvFp4LinearMethod(LinearMethodBase):
    """Native vLLM NVFP4 W4A16 linear method for GGUF NVFP4 tensors."""

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.group_size = GGUF_NVFP4_GROUP_SIZE
        self.kernel = init_nvfp4_linear_kernel(use_a16=True)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                "GGUF NVFP4 input feature size must be divisible by "
                f"{self.group_size}, got {input_size_per_partition}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = (
            layer.weight_loader_v2
            if hasattr(layer, "weight_loader_v2")
            else fallback_weight_loader
        )
        assert weight_loader is not None

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(layer, "input_scale"):
            del layer.input_scale
        layer.weight_global_scale = Parameter(
            layer.weight_scale_2.max().to(torch.float32),
            requires_grad=False,
        )
        del layer.weight_scale_2
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
