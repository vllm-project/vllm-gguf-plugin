# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Generator

import torch
from vllm.model_executor.models.utils import WeightsMapper

from .base import DiffusionGGUFAdapter, gguf_quant_weights_iterator


class WanDiffusionGGUFAdapter(DiffusionGGUFAdapter):
    """Convert original Wan transformer names to the Diffusers layout."""

    gguf_to_hf_mapper = WeightsMapper(
        orig_to_new_prefix={
            "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
            "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
            "text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
            "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
            "time_projection.1.": "condition_embedder.time_proj.",
            "head.head.": "proj_out.",
        },
        orig_to_new_substr={
            "head.modulation": "scale_shift_table",
            ".self_attn.": ".attn1.",
            ".cross_attn.": ".attn2.",
            ".q.": ".to_q.",
            ".k.": ".to_k.",
            ".v.": ".to_v.",
            ".o.": ".to_out.0.",
            ".ffn.0.": ".ffn.net.0.proj.",
            ".ffn.2.": ".ffn.net.2.",
            ".norm3.": ".norm2.",
            ".modulation": ".scale_shift_table",
        },
    )

    @staticmethod
    def is_compatible(model_class_name: str | None, model_type: str | None) -> bool:
        return model_class_name == "WanPipeline" or model_type == "wan"

    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield from self.gguf_to_hf_mapper.apply(
            gguf_quant_weights_iterator(self.gguf_file)
        )
