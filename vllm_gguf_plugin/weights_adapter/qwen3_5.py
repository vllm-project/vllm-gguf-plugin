# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download
from vllm.logger import init_logger
from vllm.transformers_utils.repo_utils import list_filtered_repo_files

from ..gguf_utils import detect_gguf_multimodal
from ..weight_utils import get_gguf_weight_type_map
from .base import GGUFLoadSpec
from .default import GGUFWeightsAdapter

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN35_MODEL_TYPES = (
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
)


class Qwen35GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for Qwen3.5 dense and MoE GGUF models (Qwen3.6 reuses the
    qwen3_5_moe architecture)."""

    _reorder: dict | None = None
    _dequant_suffixes: tuple[str, ...] = ()

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN35_MODEL_TYPES

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        # patch_hf_config may replace the config object (multimodal upgrade)
        self.config = model_config.hf_config
        mmproj_path = self._ensure_mmproj(model_path, model_config)

        gguf_to_hf_name_map = self.build_name_map(model_config)
        self.update_tie_word_embeddings(
            model_path, model_config.hf_config, gguf_to_hf_name_map
        )

        weights_source = self._get_all_gguf_files(model_path)
        if mmproj_path is not None:
            weights_source.append(str(mmproj_path))

        weight_type_map: dict[str, str] = {}
        for gguf_file in weights_source:
            weight_type_map.update(
                get_gguf_weight_type_map(gguf_file, gguf_to_hf_name_map)
            )
        unquantized_modules = self.get_unquantized_modules(weight_type_map)
        # ParallelLMHead / VocabParallelEmbedding take plain params only.
        for name in ("lm_head", "embed_tokens"):
            if name not in unquantized_modules:
                unquantized_modules.append(name)

        self.load_spec = GGUFLoadSpec(
            weights_source=weights_source,
            gguf_to_hf_name_map=gguf_to_hf_name_map,
            unquantized_modules=unquantized_modules,
        )
        return self.load_spec

    def _ensure_mmproj(self, model_path: str, model_config: ModelConfig):
        """Locate (and download if needed) the mmproj file for multimodal
        Qwen3.5 models. Returns its path, or None for text-only configs."""
        if getattr(model_config.hf_config, "vision_config", None) is None:
            return None

        mmproj_path = detect_gguf_multimodal(model_path)
        if mmproj_path is not None:
            return mmproj_path

        repo_id = self._infer_repo_id(model_config)
        if repo_id is not None:
            try:
                mmproj_files = list_filtered_repo_files(
                    repo_id,
                    allow_patterns=["*mmproj*.gguf"],
                    revision=model_config.revision,
                )
                if mmproj_files:
                    logger.info(
                        "Downloading mmproj file %s from %s",
                        mmproj_files[0],
                        repo_id,
                    )
                    # Reuse the backbone's cache root so the file lands in
                    # the same snapshot dir and concurrent TP workers
                    # serialize on the hub cache lock instead of clobbering
                    # each other.
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=mmproj_files[0],
                        revision=model_config.revision,
                        cache_dir=self._resolve_hf_cache_dir(model_path),
                    )
            except Exception as e:
                logger.warning("Failed to download mmproj from %s: %s", repo_id, e)
            mmproj_path = detect_gguf_multimodal(model_path)

        if mmproj_path is None:
            raise RuntimeError(
                "Could not find mmproj file for multimodal GGUF model. "
                "Please ensure a *mmproj*.gguf file is in the same directory "
                "as the backbone GGUF file or available in the HF repo."
            )
        return mmproj_path

    @staticmethod
    def _resolve_hf_cache_dir(model_path: str) -> str | None:
        """Return the HF cache root containing *model_path* (covers custom
        --download-dir layouts), or None for the default cache."""
        for parent in Path(model_path).parents:
            if parent.name.startswith("models--"):
                return str(parent.parent)
        return None

    @staticmethod
    def _infer_repo_id(model_config: ModelConfig) -> str | None:
        ref = str(model_config.model_weights or model_config.model)
        if os.path.exists(ref):
            return None
        if ":" in ref:
            base, _ = ref.rsplit(":", 1)
            return None if os.path.isdir(base) else base
        if ref.endswith(".gguf") and "/" in ref:
            return ref.rsplit("/", 1)[0]
        return None

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        yield from super().map_weights(self._transform_qwen35_weights(weights))

    def _transform_qwen35_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        vision_config = getattr(self.config, "vision_config", None)
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 1)
        for name, weight in weights:
            # Forced-unquantized modules keep plain params.
            if "qweight" in name and ("lm_head." in name or "embed_tokens." in name):
                continue
            if name.endswith(".A_log"):
                # GGUF stores A = -exp(A_log); recover A_log for the model.
                yield name, torch.log(-weight)
                continue
            if "conv1d.weight" in name and weight.dim() == 2:
                # depthwise Conv1d: [d, k] -> [d, 1, k]
                weight = weight.unsqueeze(1)
            elif (
                temporal_patch_size > 1
                and weight.dim() == 4
                and "patch_embed.proj.weight" in name
            ):
                # GGUF mmproj stores patch_embed as a 2D conv; the model uses
                # a 3D conv over temporal_patch_size frames. Expand to 5D.
                weight = (
                    weight.unsqueeze(2).repeat(1, 1, temporal_patch_size, 1, 1)
                    / temporal_patch_size
                )
            elif name.endswith(".weight") and weight.dim() == 1 and "norm" not in name:
                # GGUF flattens [1, hidden] linears (e.g. shared expert gate).
                weight = weight.unsqueeze(0)
            yield name, weight
