# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .layout import GGUFLinearLayout
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    _gguf_ordered_shard_ids,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
    _resolve_gguf_weight_loader,
    _resolve_gguf_weight_type_loader,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)


def _fused_mul_mat_gguf(
    x: torch.Tensor, weight: torch.Tensor, weight_type: int
) -> torch.Tensor:
    if weight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if weight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if weight.shape[0] > 5120 else 6
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    if weight_type in UNQUANTIZED_TYPES:
        return x @ weight.T
    if x.shape[0] <= mmvq_safe and weight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(weight, x, weight_type, weight.shape[0])
    elif weight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(weight, x, weight_type, weight.shape[0])
    elif weight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
        shape = (weight.shape[0], weight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(weight, weight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        weight_type = WeightType(weight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {weight_type}")
    return y


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf
except AttributeError as error:
    raise error


@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    def __init__(
        self,
        quant_config,
        layout: GGUFLinearLayout | None = None,
    ) -> None:
        self.quant_config = quant_config
        self.layout = layout

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
        del output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            weight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            weight_type,
            {
                "weight_loader": weight_loader_type,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "ignore_warning": True,
            },
        )
        set_weight_attrs(weight_type, extra_weight_attrs)
        layer.register_parameter("weight_type", weight_type)

        if self.layout is not None:
            set_weight_attrs(
                weight,
                {
                    "gguf_layout": self.layout,
                    "gguf_logical_input_size": input_size,
                    "gguf_weight_type_parameter": weight_type,
                },
            )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        weight_type = layer.weight_type.weight_type
        if not (weight_type in UNQUANTIZED_TYPES or weight_type in DEQUANT_TYPES):
            weight_type = WeightType(weight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {weight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_weight(layer)
        self._materialize_weight_type(layer)

    def _materialize_weight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "weight")

    def _materialize_weight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "weight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        weight = layer.weight
        shard_id_map = weight.shard_id_map
        shard_id = weight.shard_id
        if len(data_container := weight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=weight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            current_offset = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=weight.weight_loader,
                input_dim=weight.input_dim,
                output_dim=weight.output_dim,
                tensor_shape=weight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(weight.shard_id_map)
            if hasattr(weight, "ignore_warning"):
                padded_param.ignore_warning = weight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            weight.data_container.clear()
            weight.shard_id.clear()
            weight.shard_id_map.clear()
            if weight.data.numel() > 0:
                weight.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
            layer.register_parameter("weight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        if self.layout is not None:
            x = self.layout.input_to_gguf(x)

        shard_id = layer.weight.shard_id
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            weight = layer.weight
            fallback_wtype = layer.weight_type.weight_type
            shard_weight_types = [
                layer.weight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = fused_mul_mat_gguf_op(x, weight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            result = []
            for idx in shard_id:
                start, end, offset = layer.weight.shard_offset_map[idx]
                weight_type = layer.weight_type.shard_weight_type.get(
                    idx, fallback_wtype
                )
                result.append(
                    fused_mul_mat_gguf_op(
                        x, weight[start:end, :offset].contiguous(), weight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            weight = layer.weight
            weight_type = layer.weight_type.weight_type
            out = fused_mul_mat_gguf_op(x, weight, weight_type)
        if bias is not None:
            out.add_(bias)
        return out
