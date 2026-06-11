# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .linear import GGUFLinearMethod
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_embedding_weight_loader,
    _gguf_embedding_weight_type_loader,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
)
from .utils import DEQUANT_TYPES, UNQUANTIZED_TYPES


def _apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    if qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ops.ggml_dequantize(
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    qweight_type = WeightType(qweight_type)
    raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


def _apply_gguf_embedding_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    del qweight, qweight_type
    return torch.empty(*x.shape, hidden_size, dtype=dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_apply_gguf_embedding",
        op_func=_apply_gguf_embedding,
        fake_impl=_apply_gguf_embedding_fake,
    )
    apply_gguf_embedding = torch.ops.vllm._apply_gguf_embedding
except AttributeError as error:
    raise error


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF."""

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
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        extra_weight_attrs.pop("weight_loader", None)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": partial(_gguf_embedding_weight_loader, layer),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": _gguf_embedding_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(
            layer,
            "qweight",
            fallback_weight_loader=partial(_gguf_embedding_weight_loader, layer),
        )

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(
            layer,
            "qweight_type",
            fallback_weight_loader=_gguf_embedding_weight_type_loader,
        )

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        from . import apply_gguf_embedding as apply_gguf_embedding_op

        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]
        return apply_gguf_embedding_op(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: VocabParallelEmbedding):
        del layer
        return embed_tokens
