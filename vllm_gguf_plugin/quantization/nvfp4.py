# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
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
    if qweight.ndim == 3:
        if qweight.shape[-1] != GGUF_NVFP4_BLOCK_BYTES:
            raise ValueError(
                "Expected GGUF NVFP4 block dimension to be "
                f"{GGUF_NVFP4_BLOCK_BYTES}, got {qweight.shape[-1]}."
            )
        return qweight.contiguous()
    if qweight.ndim != 2:
        raise ValueError(
            "Expected a 2-D row-packed or 3-D block-packed GGUF NVFP4 tensor, "
            f"got {qweight.ndim} dimensions."
        )
    if qweight.shape[1] % GGUF_NVFP4_BLOCK_BYTES != 0:
        raise ValueError(
            "Expected GGUF NVFP4 row byte width to be divisible by "
            f"{GGUF_NVFP4_BLOCK_BYTES}, got {qweight.shape[1]}."
        )
    blocks_per_row = qweight.shape[1] // GGUF_NVFP4_BLOCK_BYTES
    return qweight.reshape(qweight.shape[0], blocks_per_row, GGUF_NVFP4_BLOCK_BYTES)


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
    rows, blocks_per_row, _ = blocks.shape
    scale_bytes = blocks[..., :GGUF_NVFP4_SCALE_BYTES]
    packed_values = blocks[..., GGUF_NVFP4_SCALE_BYTES:].reshape(
        rows, blocks_per_row, GGUF_NVFP4_SCALE_BYTES, 8
    )
    low = packed_values & 0x0F
    high = packed_values >> 4
    values = torch.cat((low, high), dim=-1)
    packed_values = values[..., 0::2] | (values[..., 1::2] << 4)

    weight = packed_values.reshape(rows, blocks_per_row * (GGUF_NVFP4_BLOCK_SIZE // 2))
    weight_scale = gguf_ue4m3_to_fp8_e4m3fn(
        scale_bytes.reshape(rows, blocks_per_row * GGUF_NVFP4_SCALE_BYTES)
    )
    return weight.contiguous(), weight_scale.contiguous()


def iter_gguf_nvfp4_native_weights(
    module_name: str,
    qweight: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    weight, weight_scale = split_gguf_nvfp4_weight(qweight)
    return [
        (f"{module_name}.weight", weight),
        (f"{module_name}.weight_scale", weight_scale),
        (f"{module_name}.weight_scale_2", torch.tensor(1.0, dtype=torch.float32)),
    ]


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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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
