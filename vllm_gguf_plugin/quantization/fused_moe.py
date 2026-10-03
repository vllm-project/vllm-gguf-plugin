# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from ..kernel_support import (
    QuantizationBackend,
    QuantizationOperation,
    supports,
)
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
    _materialize_upstream_moe_storage_padding,
    _store_gguf_weight_type,
)


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(activation_enum, out, inp)
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    moe_mode = ops.cuda_moe_kernel_mode()
    if moe_mode in {"upstream", "auto"}:
        num_tokens = x.size(0)
        top_k = topk_ids.size(1)
        # Routing can be shared only for identical formats. Method decisions
        # remain per projection, so W2 can still select grouped/another method.
        alignment_cache = {} if weight_type == weight_type2 else None
        out = ops.ggml_moe(
            x,
            w1,
            topk_ids,
            weight_type,
            w1.size(1),
            top_k,
            num_tokens,
            alignment_cache=alignment_cache,
        )
        out = act(out)
        out = ops.ggml_moe(
            out,
            w2,
            topk_ids.reshape(-1, 1),
            weight_type2,
            w2.size(1),
            1,
            num_tokens * top_k,
            alignment_cache=alignment_cache,
        )
        out = out.reshape(num_tokens, top_k, w2.size(1)).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
        return out_hidden_states

    backend = QuantizationBackend(moe_mode)

    def backend_supports(quant_type: int, operation: QuantizationOperation) -> bool:
        return supports(quant_type, backend, operation)

    if (
        backend_supports(weight_type2, QuantizationOperation.MMQ)
        and backend_supports(weight_type, QuantizationOperation.MMQ)
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        block_size = ops.ggml_moe_get_block_size(weight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        if weight_type != weight_type2:
            # A different format/backend can require a different route tile.
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, ops.ggml_moe_get_block_size(weight_type2), E
            )
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif backend_supports(
        weight_type2, QuantizationOperation.MMVQ
    ) and backend_supports(weight_type, QuantizationOperation.MMVQ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, weight_type, N, num_tokens)
        out = act(out)

        out = ops.ggml_moe_a8_vec(
            out, w2, topk_ids, 1, weight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    else:
        raise RuntimeError(f"{moe_mode} MoE has no kernel for the selected types")
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    del w1, w2, topk_weights, topk_ids, weight_type, weight_type2, activation
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader,
                    layer,
                    base_weight_loader,
                    params_dtype=params_dtype,
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w13_weight", w13_weight)

        w13_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_weight_type, extra_weight_attrs)
        layer.register_parameter("w13_weight_type", w13_weight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader,
                    layer,
                    base_weight_loader,
                    params_dtype=params_dtype,
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)

        w2_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_weight_type, extra_weight_attrs)
        layer.register_parameter("w2_weight_type", w2_weight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Finalize floating types and upstream storage on MoE weights.

        Align each floating type marker with the loaded tensor's dtype and
        reserve the trailing storage required by upstream MoE kernels.
        """
        float_types = {
            torch.float32: int(WeightType.F32),
            torch.float16: int(WeightType.F16),
            torch.bfloat16: int(WeightType.BF16),
        }
        for weight_name in ("w13_weight", "w2_weight"):
            weight = getattr(layer, weight_name)
            type_param = getattr(layer, f"{weight_name}_type")
            if weight.dtype in float_types:
                # The GGUF iterator skips floating weight_type entries, so
                # their default F32 marker can disagree with the tensor.
                weight_type = float_types[weight.dtype]
                _store_gguf_weight_type(
                    type_param, torch.tensor(weight_type, dtype=torch.uint8)
                )
            _materialize_upstream_moe_storage_padding(weight, type_param.weight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        return fused_moe_gguf_op(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            layer.w13_weight_type.weight_type,
            layer.w2_weight_type.weight_type,
            layer.activation.value,
        )
