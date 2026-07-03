# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.models.utils import WeightsMapper

from .base import DiffusionGGUFAdapter, gguf_quant_weights_iterator

Z_IMAGE_KEYS_RENAME_DICT = {
    "final_layer.": "all_final_layer.2-1.",
    "x_embedder.": "all_x_embedder.2-1.",
    ".attention.qkv": ".attention.to_qkv",
    ".attention.k_norm": ".attention.norm_k",
    ".attention.q_norm": ".attention.norm_q",
    ".attention.out": ".attention.to_out.0",
    "model.diffusion_model.": "",
}


class ZImageDiffusionGGUFAdapter(DiffusionGGUFAdapter):
    """GGUF adapter for Z-Image models with QKV/FFN shard support."""

    @staticmethod
    def is_compatible(
        model_class_name: str | None,
        model_type: str | None,
    ) -> bool:
        if model_class_name and model_class_name.startswith("ZImage"):
            return True
        return bool(
            model_type and model_type.lower() in {"z_image", "zimage", "z-image"}
        )

    unquantized_modules = ("model", "lm_head")

    gguf_to_hf_mapper = WeightsMapper(
        orig_to_new_substr=Z_IMAGE_KEYS_RENAME_DICT,
    )

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        weights = gguf_quant_weights_iterator(self.gguf_file)
        yield from self.gguf_to_hf_mapper.apply(weights)
