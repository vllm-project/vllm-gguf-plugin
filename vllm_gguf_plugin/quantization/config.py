# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix

from .utils import is_layer_skipped_gguf

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from .layout import GGUFLinearLayout


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(
        self,
        unquantized_modules: list[str] | None = None,
        dense_module_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []
        #: Module paths ending in one of these stay unquantized wherever they
        #: appear.  ``unquantized_modules`` cannot express that: a fused layer
        #: is matched by asking whether a declared name *contains* the layer's
        #: full runtime path, so every declaration has to spell out a prefix and
        #: a layer index.  Those are knowable for a target model and awkward for
        #: a draft, whose layers vLLM numbers after the target's.
        self.dense_module_suffixes = dense_module_suffixes or []
        self.linear_layouts: dict[str, GGUFLinearLayout] = {}

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_name(self) -> QuantizationMethods:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        # A target model's list is filled in by the loader, which holds the same
        # object the layers were built against.  A draft's is not: its config is
        # rebuilt from scratch here, so anything the loader records reaches the
        # target's object and never the draft's.  Reading the list back out of
        # the config dict is what lets a draft declare one at all.
        return cls(
            unquantized_modules=config.get("unquantized_modules"),
            dense_module_suffixes=config.get("dense_module_suffixes"),
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> "QuantizationMethods | None":
        del hf_quant_cfg
        if user_quant == "gguf":
            return "gguf"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        from .fused_moe import GGUFMoEMethod
        from .linear import GGUFLinearMethod
        from .vocal_embeds import GGUFEmbeddingMethod

        if isinstance(layer, LinearBase):
            if prefix.endswith(tuple(self.dense_module_suffixes)) or (
                is_layer_skipped_gguf(
                    prefix, self.unquantized_modules, self.packed_modules_mapping
                )
            ):
                return UnquantizedLinearMethod()
            return GGUFLinearMethod(
                self,
                layout=self.linear_layouts.get(prefix),
            )
        if isinstance(layer, VocabParallelEmbedding):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedEmbeddingMethod()
            return GGUFEmbeddingMethod(self)
        if isinstance(layer, RoutedExperts):
            return GGUFMoEMethod(self, layer.moe_config)
        return None

    def register_linear_layouts(
        self,
        layouts: Mapping[str, "GGUFLinearLayout"],
        prefix: str = "",
    ) -> None:
        """Register GGUF linear layouts before model initialization."""
        self.linear_layouts.update(
            (maybe_prefix(prefix, module_name), layout)
            for module_name, layout in layouts.items()
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        :param hf_to_vllm_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to vllm model structure
        """
        if self.unquantized_modules is not None:
            self.unquantized_modules = hf_to_vllm_mapper.apply_list(
                self.unquantized_modules
            )
        if self.linear_layouts:
            layouts = self.linear_layouts
            mapped_names = hf_to_vllm_mapper.apply_list(list(layouts))
            self.linear_layouts = dict(zip(mapped_names, layouts.values(), strict=True))
