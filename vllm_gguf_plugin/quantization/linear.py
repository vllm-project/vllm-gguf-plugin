# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from ..kernel_support import (
    QuantizationBackend,
    supports_moe,
    upstream_storage_padding_bytes,
)
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
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)


def _fused_mul_mat_gguf(
    x: torch.Tensor, weight: torch.Tensor, weight_type: int
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    mode = ops.cuda_dense_kernel_mode()
    if mode in {"auto", "upstream"}:
        return ops.ggml_dense(weight, x, weight_type, weight.shape[0])
    if weight_type in UNQUANTIZED_TYPES:
        return x @ weight.T

    if weight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if weight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if weight.shape[0] > 5120 else 6
    use_mmvq = x.shape[0] <= mmvq_safe

    if use_mmvq and weight_type in MMVQ_QUANT_TYPES:
        return ops.ggml_mul_mat_vec_a8(weight, x, weight_type, weight.shape[0])
    if weight_type in DEQUANT_TYPES:
        return ops.ggml_mul_mat_a8(weight, x, weight_type, weight.shape[0])
    weight_type = WeightType(weight_type)
    raise NotImplementedError(f"Unsupported GGUF quantization type: {weight_type}")


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

        set_weight_attrs(weight, {"gguf_weight_type_parameter": weight_type})
        if self.layout is not None:
            set_weight_attrs(
                weight,
                {
                    "gguf_layout": self.layout,
                    "gguf_logical_input_size": input_size,
                },
            )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        weight_type = layer.weight_type.weight_type
        if not (
            weight_type in UNQUANTIZED_TYPES
            or weight_type in DEQUANT_TYPES
            or supports_moe(weight_type, QuantizationBackend.UPSTREAM)
        ):
            weight_type = WeightType(weight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {weight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)
        self._materialize_upstream_storage_padding(layer)

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_weight(layer)
        self._materialize_weight_type(layer)

    def _materialize_weight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "weight")

    def _materialize_weight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "weight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Materialize merged GGUF shards in their execution layout."""
        weight = layer.weight
        data_container = weight.data_container
        if len(data_container) <= 1:
            return

        dtypes = {data.dtype for data in data_container}
        assert len(dtypes) == 1, ValueError(
            f"Data container has mixed dtypes: {dtypes}"
        )
        dtype = next(iter(dtypes))
        ordered_shard_ids = _gguf_ordered_shard_ids(weight.shard_id)
        fallback_wtype = layer.weight_type.weight_type
        shard_weight_types = {
            shard_id: layer.weight_type.shard_weight_type.get(shard_id, fallback_wtype)
            for shard_id in ordered_shard_ids
        }
        mixed_types = len(set(shard_weight_types.values())) > 1
        shard_offset_map: dict[int | str, tuple[int, int, int]] = {}
        shard_storage_map: dict[int | str, tuple[int, int, int]] = {}

        if mixed_types:
            storage_size = 0
            current_row = 0
            for shard_id in ordered_shard_ids:
                data = data_container[weight.shard_id_map[shard_id]]
                rows, packed_row_size = data.shape
                shard_offset_map[shard_id] = (
                    current_row,
                    current_row + rows,
                    packed_row_size,
                )
                shard_storage_map[shard_id] = (
                    storage_size,
                    rows,
                    packed_row_size,
                )
                storage_size += data.numel()
                if dtype == torch.uint8 and torch.version.hip is None:
                    storage_size += upstream_storage_padding_bytes(
                        shard_weight_types[shard_id], packed_row_size
                    )
                current_row += rows

            padded_data = torch.zeros(storage_size, dtype=dtype, device=weight.device)
            for shard_id in ordered_shard_ids:
                data = data_container[weight.shard_id_map[shard_id]]
                storage_offset, rows, packed_row_size = shard_storage_map[shard_id]
                shard_view = padded_data.narrow(
                    0, storage_offset, rows * packed_row_size
                ).view(rows, packed_row_size)
                shard_view.copy_(data)
        else:
            padded_side = max(data.size(1) for data in data_container)
            concat_side = sum(data.size(0) for data in data_container)
            weight_type = next(iter(shard_weight_types.values()))
            padding_bytes = 0
            if dtype == torch.uint8 and torch.version.hip is None:
                padding_bytes = upstream_storage_padding_bytes(weight_type, padded_side)
            logical_numel = concat_side * padded_side
            storage = torch.zeros(
                logical_numel + padding_bytes,
                dtype=dtype,
                device=weight.device,
            )
            padded_data = storage[:logical_numel].view(concat_side, padded_side)
            current_row = 0
            for shard_id in ordered_shard_ids:
                data = data_container[weight.shard_id_map[shard_id]]
                start = current_row
                end = start + data.size(0)
                packed_row_size = data.size(1)
                padded_data[start:end, :packed_row_size] = data
                shard_offset_map[shard_id] = (
                    start,
                    end,
                    packed_row_size,
                )
                current_row = end

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
        attrs = {"shard_offset_map": shard_offset_map}
        if shard_storage_map:
            attrs["shard_storage_map"] = shard_storage_map
        set_weight_attrs(padded_param, attrs)
        weight.data_container.clear()
        weight.shard_id.clear()
        weight.shard_id_map.clear()
        if weight.data.numel() > 0:
            weight.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
        layer.register_parameter("weight", padded_param)

    def _materialize_upstream_storage_padding(self, layer: torch.nn.Module) -> None:
        """Add the upstream storage tail when the loader could not preallocate it."""
        weight = layer.weight
        if (
            torch.version.hip is not None
            or weight.dtype != torch.uint8
            or weight.ndim != 2
            or not weight.is_contiguous()
        ):
            return

        weight_type = layer.weight_type.weight_type
        padding_bytes = upstream_storage_padding_bytes(weight_type, weight.shape[1])
        if padding_bytes == 0:
            return
        logical_numel = weight.numel()
        storage_bytes = weight.untyped_storage().nbytes()
        offset_bytes = weight.storage_offset() * weight.element_size()
        if storage_bytes >= offset_bytes + logical_numel + padding_bytes:
            return
        storage = torch.empty(
            logical_numel + padding_bytes,
            dtype=weight.dtype,
            device=weight.device,
        )
        storage[:logical_numel].copy_(weight.reshape(-1))
        storage[logical_numel:].zero_()
        weight.data = storage[:logical_numel].view_as(weight)

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
                shard_storage_map = getattr(weight, "shard_storage_map", None)
                if shard_storage_map is None:
                    shard_weight = weight[start:end, :offset].contiguous()
                else:
                    storage_offset, rows, packed_row_size = shard_storage_map[idx]
                    shard_weight = weight.narrow(
                        0, storage_offset, rows * packed_row_size
                    ).view(rows, packed_row_size)
                result.append(fused_mul_mat_gguf_op(x, shard_weight, weight_type))
            out = torch.cat(result, axis=1)
        else:
            weight = layer.weight
            weight_type = layer.weight_type.weight_type
            out = fused_mul_mat_gguf_op(x, weight, weight_type)
        if bias is not None:
            out.add_(bias)
        return out
