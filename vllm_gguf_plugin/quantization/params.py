# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parameter import BasevLLMParameter


def _clone_loaded_weight(loaded_weight: torch.Tensor) -> torch.Tensor:
    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)
    return loaded_weight.detach().clone()


def _resolve_gguf_weight_loader(
    layer: torch.nn.Module,
    fallback_weight_loader=None,
):
    return (
        layer.weight_loader_v2
        if hasattr(layer, "weight_loader_v2")
        else fallback_weight_loader
    )


def _resolve_gguf_weight_type_loader(
    layer: torch.nn.Module,
    fallback_weight_loader=None,
):
    """Weight loader for GGUF weight-type parameters."""
    base_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    if base_loader is None:
        return fallback_weight_loader

    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):
        if loaded_shard_id is None and hasattr(param, "_store"):
            param._store(loaded_weight)
            return
        base_loader(param, loaded_weight, loaded_shard_id)

    return _gguf_weight_type_loader_v2


def _materialize_parameter_data(
    param: Parameter | UninitializedParameter,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if isinstance(param, UninitializedParameter):
        param.materialize(shape, device=param.device, dtype=dtype)


def _gguf_shard_id_as_int(shard_id: int | str) -> int:
    if isinstance(shard_id, int):
        return shard_id
    qkv_idxs = {"q": 0, "k": 1, "v": 2}
    return qkv_idxs[shard_id]


def _gguf_ordered_shard_ids(shard_ids: list[int | str]) -> list[int | str]:
    return sorted(shard_ids, key=_gguf_shard_id_as_int)


def _store_gguf_loaded_weight(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    loaded_weight = _clone_loaded_weight(loaded_weight).to(device=param.device)
    if shard_id is None:
        _materialize_parameter_data(
            param, tuple(loaded_weight.shape), loaded_weight.dtype
        )
        param.data.copy_(loaded_weight)
        return

    if shard_id not in param.shard_id_map:
        param.shard_id_map[shard_id] = len(param.data_container)
        param.data_container.append(loaded_weight)
        param.shard_id.append(shard_id)
    else:
        param.data_container[param.shard_id_map[shard_id]] = loaded_weight
    if not isinstance(param, UninitializedParameter) and param.data.numel() == 0:
        param.data = loaded_weight


def _store_gguf_weight_type(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    loaded_weight = _clone_loaded_weight(loaded_weight).to(
        device=param.device, dtype=torch.uint8
    )
    weight_type = int(loaded_weight.item())
    num_elements = getattr(param, "num_elements", 1)
    if shard_id is None:
        _materialize_parameter_data(param, (num_elements,), torch.uint8)
        param.weight_type = weight_type
        if param.data.numel() == 1:
            param.data.fill_(weight_type)
        else:
            param.data.zero_()
            param.data[0] = weight_type
        return

    param.shard_weight_type[shard_id] = weight_type
    if len(param.shard_weight_type) == 1:
        param.weight_type = weight_type
    if not isinstance(param, UninitializedParameter):
        if param.data.numel() == 0:
            param.data = torch.empty(
                num_elements, dtype=torch.uint8, device=loaded_weight.device
            )
        param.data[_gguf_shard_id_as_int(shard_id)] = weight_type


def _gguf_embedding_weight_loader(
    layer: VocabParallelEmbedding,
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
) -> None:
    loaded_weight = _clone_loaded_weight(loaded_weight).to(device=param.device)
    start_idx = layer.shard_indices.org_vocab_start_index
    shard_size = layer.shard_indices.org_vocab_end_index - start_idx
    loaded_weight = loaded_weight.narrow(param.output_dim, start_idx, shard_size)

    padded_shape = list(loaded_weight.shape)
    padded_shape[param.output_dim] = param.tensor_shape[param.output_dim]
    _materialize_parameter_data(param, tuple(padded_shape), loaded_weight.dtype)
    param.data.zero_()
    param.data.narrow(param.output_dim, 0, loaded_weight.shape[param.output_dim]).copy_(
        loaded_weight
    )


def _gguf_embedding_weight_type_loader(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
) -> None:
    _store_gguf_weight_type(param, loaded_weight)


def _materialize_gguf_moe_param(
    layer: FusedMoE,
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
) -> None:
    if not isinstance(param, UninitializedParameter):
        return

    shard_dim = {"w1": 0, "w2": 1, "w3": 0}[shard_id]
    if getattr(param, "is_transposed", False):
        shard_dim = int(not shard_dim)

    if len(loaded_weight.shape) != 3:
        return

    shard_dim += 1
    final_shape = list(loaded_weight.shape)
    if shard_id in {"w1", "w3"}:
        final_shape[1] *= 2
    final_shape[shard_dim] = final_shape[shard_dim] // layer.tp_size
    param.materialize(tuple(final_shape), dtype=loaded_weight.dtype)


def _gguf_moe_weight_loader(
    layer: FusedMoE,
    base_weight_loader,
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    _materialize_gguf_moe_param(layer, param, loaded_weight, shard_id)
    return base_weight_loader(
        param,
        loaded_weight,
        weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=return_success,
    )


def _gguf_moe_weight_type_loader(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    del weight_name, expert_id
    _store_gguf_weight_type(param, loaded_weight, shard_id)
    return True if return_success else None


class _GGUFParamLoadMixin:
    """Mixin providing GGUF parameter weight loading methods."""

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and loaded_weight.ndim >= 1:
            shard_size = loaded_weight.shape[0] // tp_size
            if shard_size > 0:
                loaded_weight = loaded_weight.narrow(
                    0, tp_rank * shard_size, shard_size
                )
        self._store(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and loaded_weight.ndim >= 2:
            shard_size = loaded_weight.shape[1] // tp_size
            if shard_size > 0:
                loaded_weight = loaded_weight.narrow(
                    1, tp_rank * shard_size, shard_size
                )
        self._store(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        tp_rank = kwargs.get("tp_rank", 0)
        shard_size = kwargs.get("shard_size")
        if (
            shard_size is not None
            and loaded_weight.ndim >= 1
            and shard_size > 0
            and shard_size < loaded_weight.shape[0]
        ):
            loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)
        self._store(loaded_weight, shard_id=shard_id)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        tp_rank = kwargs.get("tp_rank", 0)
        shard_size = kwargs.get("shard_size")
        num_kv_head_replicas = kwargs.get("num_heads", 1)
        if (
            shard_size is not None
            and loaded_weight.ndim >= 1
            and shard_size > 0
            and shard_size < loaded_weight.shape[0]
        ):
            effective_tp_rank = (
                tp_rank // num_kv_head_replicas if shard_id in ("k", "v") else tp_rank
            )
            loaded_weight = loaded_weight.narrow(
                0, effective_tp_rank * shard_size, shard_size
            )
        self._store(loaded_weight, shard_id=shard_id)


class GGUFWeightParameter(_GGUFParamLoadMixin, BasevLLMParameter):
    def __init__(
        self,
        *,
        data: torch.Tensor,
        weight_loader,
        input_dim: int,
        output_dim: int,
        tensor_shape: tuple[int, ...],
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.tensor_shape = tensor_shape
        self.data_container: list[torch.Tensor] = []
        self.shard_id: list[int | str] = []
        self.shard_id_map: dict[int | str, int] = {}
        super().__init__(data=data, weight_loader=weight_loader)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _store(
        self,
        loaded_weight: torch.Tensor,
        shard_id: int | str | None = None,
    ) -> None:
        _store_gguf_loaded_weight(self, loaded_weight, shard_id)


class GGUFWeightTypeParameter(_GGUFParamLoadMixin, BasevLLMParameter):
    def __init__(self, *, data: torch.Tensor, weight_loader):
        self.weight_type = 0
        self.shard_weight_type: dict[int | str, int] = {}
        self.num_elements = data.numel()
        super().__init__(data=data, weight_loader=weight_loader)

    def _store(
        self,
        loaded_weight: torch.Tensor,
        shard_id: int | str | None = None,
    ) -> None:
        _store_gguf_weight_type(self, loaded_weight, shard_id)


def _materialize_gguf_weight_parameter(
    layer: torch.nn.Module,
    param_name: str,
    fallback_weight_loader=None,
) -> None:
    raw_param = getattr(layer, param_name)
    if isinstance(raw_param, GGUFWeightParameter):
        return

    if fallback_weight_loader is None:
        fallback_weight_loader = getattr(raw_param, "weight_loader", None)
    weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    assert weight_loader is not None
    if isinstance(raw_param, UninitializedParameter):
        data = torch.empty(0, dtype=torch.uint8, device=raw_param.device)
    else:
        data = raw_param.data
    qweight = GGUFWeightParameter(
        data=data,
        weight_loader=weight_loader,
        input_dim=raw_param.input_dim,
        output_dim=raw_param.output_dim,
        tensor_shape=raw_param.tensor_shape,
    )
    qweight.data_container = list(raw_param.data_container)
    qweight.shard_id = list(raw_param.shard_id)
    qweight.shard_id_map = dict(raw_param.shard_id_map)
    if hasattr(raw_param, "ignore_warning"):
        qweight.ignore_warning = raw_param.ignore_warning
    layer.register_parameter(param_name, qweight)


def _materialize_gguf_weight_type_parameter(
    layer: torch.nn.Module,
    param_name: str,
    fallback_weight_loader=None,
) -> None:
    raw_param = getattr(layer, param_name)
    if isinstance(raw_param, GGUFWeightTypeParameter):
        return

    if fallback_weight_loader is None:
        fallback_weight_loader = getattr(raw_param, "weight_loader", None)
    weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    assert weight_loader is not None
    num_elements = getattr(raw_param, "num_elements", 1)
    if isinstance(raw_param, UninitializedParameter):
        data = torch.empty(num_elements, dtype=torch.uint8, device=raw_param.device)
    else:
        data = raw_param.data
    qweight_type = GGUFWeightTypeParameter(data=data, weight_loader=weight_loader)
    qweight_type.num_elements = num_elements
    qweight_type.weight_type = raw_param.weight_type
    qweight_type.shard_weight_type = dict(raw_param.shard_weight_type)
    if hasattr(raw_param, "ignore_warning"):
        qweight_type.ignore_warning = raw_param.ignore_warning
    layer.register_parameter(param_name, qweight_type)


class GGUFUninitializedParameter(_GGUFParamLoadMixin, UninitializedParameter):
    """Base class for uninitialized GGUF parameters."""

    cls_to_become = Parameter


class GGUFUninitializedWeightParameter(GGUFUninitializedParameter):
    data_container: list[torch.Tensor]

    def _store(self, loaded_weight: torch.Tensor, shard_id=None):
        _store_gguf_loaded_weight(self, loaded_weight, shard_id)


class GGUFUninitializedWeightTypeParameter(GGUFUninitializedParameter):
    def _store(self, loaded_weight: torch.Tensor, shard_id=None):
        _store_gguf_weight_type(self, loaded_weight, shard_id)
