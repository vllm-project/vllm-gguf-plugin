# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import gguf
import torch

from .base import (
    UNQUANTIZED_GGUF_TYPE_NAMES,
    DiffusionGGUFAdapter,
    gguf_quant_weights_iterator,
)

_NUM_QUERY_GROUPS = 56
_HEAD_DIM = 128


def _fused_qkv_to_grouped(weight: torch.Tensor) -> torch.Tensor:
    """Restore H3's grouped checkpoint layout for its dense weight loader.

    H3 GGUF converters store QKV as contiguous Q, K, V rows. Quantized
    ``qweight`` tensors enter vLLM's QKV loader directly in that layout. Dense
    ``weight`` tensors, however, pass through MiniMaxH3DiTModel.load_weights,
    which expects the original per-head ``[q, k, v]`` checkpoint layout and
    performs the grouped-to-fused conversion itself.
    """
    rows_per_projection = _NUM_QUERY_GROUPS * _HEAD_DIM
    expected_rows = 3 * rows_per_projection
    if weight.shape[0] != expected_rows:
        raise ValueError(
            "MiniMax-H3 GGUF QKV tensor has incompatible output dimension: "
            f"got {tuple(weight.shape)}, expected first dimension {expected_rows}."
        )

    rest_shape = weight.shape[1:]
    q, k, v = weight.split(rows_per_projection, dim=0)
    return torch.cat(
        [
            q.reshape(_NUM_QUERY_GROUPS, _HEAD_DIM, *rest_shape),
            k.reshape(_NUM_QUERY_GROUPS, _HEAD_DIM, *rest_shape),
            v.reshape(_NUM_QUERY_GROUPS, _HEAD_DIM, *rest_shape),
        ],
        dim=1,
    ).reshape(expected_rows, *rest_shape)


class MiniMaxH3DiffusionGGUFAdapter(DiffusionGGUFAdapter):
    """GGUF adapter for MiniMax-H3 single-partition pipelines."""

    # H3 forwards the diffusion quantization config to its Qwen3-VL encoder.
    # The DiT GGUF does not contain encoder tensors, so keep that component on
    # its Hugging Face weights.
    unquantized_modules = ("text_encoder",)

    @staticmethod
    def is_compatible(
        model_class_name: str | None,
        model_type: str | None,
    ) -> bool:
        if model_class_name and model_class_name.startswith("MiniMaxH3"):
            return True
        if not model_type:
            return False
        normalized_model_type = model_type.lower().replace("-", "_")
        return normalized_model_type in {"minimax_h3", "minimaxh3"}

    def unquantized_weight_names(self) -> Iterable[str]:
        reader = gguf.GGUFReader(self.gguf_file)
        tensor_names = {tensor.name for tensor in reader.tensors}
        required_time_embedder_names = {
            "time_embedder.proj_in.weight",
            "time_embedder.proj_in.bias",
            "time_embedder.proj_out.weight",
            "time_embedder.proj_out.bias",
        }
        if "adaln_t_table" in tensor_names or not required_time_embedder_names.issubset(
            tensor_names
        ):
            raise ValueError(
                "Pruned MiniMax-H3 GGUF checkpoints are not supported by the "
                "current vLLM-Omni MiniMax-H3 architecture. Use a non-pruned "
                "FL2VA or Ref2VA GGUF checkpoint."
            )

        for tensor in reader.tensors:
            if tensor.tensor_type.name in UNQUANTIZED_GGUF_TYPE_NAMES:
                yield tensor.name

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in gguf_quant_weights_iterator(self.gguf_file):
            if name.endswith(".attn.qkv_proj.weight"):
                weight = _fused_qkv_to_grouped(weight)
            yield name, weight
