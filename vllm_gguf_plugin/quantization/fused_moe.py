# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch
from gguf import GGML_QUANT_SIZES
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
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
)
from .utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES, logger

_MAX_GGUF_MOE_ROUTED_TOKENS = (1 << 16) - 1


def _moe_token_slices(num_tokens: int, top_k: int) -> list[slice]:
    """Split batches before the 16-bit routed-token launch boundary."""
    tokens_per_chunk = _MAX_GGUF_MOE_ROUTED_TOKENS // top_k
    if tokens_per_chunk <= 0:
        raise ValueError(
            f"GGUF MoE top_k={top_k} exceeds the routed-token launch limit."
        )
    return [
        slice(start, min(start + tokens_per_chunk, num_tokens))
        for start in range(0, num_tokens, tokens_per_chunk)
    ]


def _apply_gguf_moe_activation(
    inp: torch.Tensor,
    activation: str,
    activation_situ_beta: float,
    activation_situ_linear_beta: float,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)
    d = inp.shape[-1] // 2
    output_shape = inp.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    apply_moe_activation(
        activation_enum,
        out,
        inp,
        activation_situ_beta=(
            None if activation_situ_beta < 0 else activation_situ_beta
        ),
        activation_situ_linear_beta=(
            None if activation_situ_linear_beta < 0 else activation_situ_linear_beta
        ),
    )
    return out


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    activation_situ_beta: float,
    activation_situ_linear_beta: float,
) -> torch.Tensor:
    top_k = topk_ids.shape[1]
    if x.shape[0] * top_k > _MAX_GGUF_MOE_ROUTED_TOKENS:
        return torch.cat(
            [
                _fused_moe_gguf(
                    x[token_slice],
                    w1,
                    w2,
                    topk_weights[token_slice],
                    topk_ids[token_slice],
                    expert_map,
                    qweight_type,
                    qweight_type2,
                    activation,
                    activation_situ_beta,
                    activation_situ_linear_beta,
                )
                for token_slice in _moe_token_slices(x.shape[0], top_k)
            ],
            dim=0,
        )

    if expert_map.numel() != 0:
        local_topk_ids = expert_map[topk_ids.long()]
        topk_weights = topk_weights * (local_topk_ids >= 0).to(topk_weights.dtype)
        topk_ids = local_topk_ids.clamp_min(0).to(topk_ids.dtype)

    def act(inp: torch.Tensor):
        return _apply_gguf_moe_activation(
            inp,
            activation,
            activation_situ_beta,
            activation_situ_linear_beta,
        )

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        block_size = ops.ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        out = ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ops.ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    else:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                out = fused_mul_mat_gguf_op(inp, w1[ii], qweight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], qweight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    activation_situ_beta: float,
    activation_situ_linear_beta: float,
) -> torch.Tensor:
    del (
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_map,
        qweight_type,
        qweight_type2,
        activation,
        activation_situ_beta,
        activation_situ_linear_beta,
    )
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
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        parallel = layer.moe_config.moe_parallel_config
        if parallel.tp_size <= 1:
            return
        weight_type = WeightType(layer.w2_qweight_type.weight_type)
        block_size, _ = GGML_QUANT_SIZES[weight_type]
        per_rank_intermediate = layer.moe_config.intermediate_size // parallel.tp_size
        if per_rank_intermediate % block_size:
            raise ValueError(
                f"GGUF MoE {weight_type.name} blocks contain {block_size} values, "
                f"but TP{parallel.tp_size} would split intermediate_size="
                f"{layer.moe_config.intermediate_size} into "
                f"{per_rank_intermediate} values per rank. Use "
                "--enable-expert-parallel so each rank owns complete experts."
            )

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
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            (
                layer.expert_map
                if layer.expert_map is not None
                else torch.empty(0, dtype=torch.int32, device=x.device)
            ),
            layer.w13_qweight_type.weight_type,
            layer.w2_qweight_type.weight_type,
            layer.activation.value,
            (
                -1.0
                if layer.moe_config.activation_situ_beta is None
                else layer.moe_config.activation_situ_beta
            ),
            (
                -1.0
                if layer.moe_config.activation_situ_linear_beta is None
                else layer.moe_config.activation_situ_linear_beta
            ),
        )
