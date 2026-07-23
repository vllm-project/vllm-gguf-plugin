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
from ..weight_utils import get_gguf_weight_type_map, gguf_quant_weights_iterator_multi
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
        # Only the first Conv3d temporal slice has a gguf-py map entry.
        patch_embd = gguf_to_hf_name_map.get("v.patch_embd.weight")
        if patch_embd is not None:
            tps = getattr(self.config.vision_config, "temporal_patch_size", 1)
            for i in range(1, tps):
                gguf_to_hf_name_map[f"v.patch_embd.weight.{i}"] = f"{patch_embd}.{i}"
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

        # llama.cpp tiles GDN V heads when num_value_heads != num_key_heads.
        tc = self.config.get_text_config()
        num_k = getattr(tc, "linear_num_key_heads", 0) or 0
        num_v = getattr(tc, "linear_num_value_heads", 0) or 0
        self._reorder = None
        self._dequant_suffixes: tuple[str, ...] = ()
        if num_k and num_v and num_v % num_k == 0 and num_v // num_k > 1:
            self._reorder = {
                "num_k": num_k,
                "r": num_v // num_k,
                "head_k": tc.linear_key_head_dim,
                "head_v": tc.linear_value_head_dim,
            }
            # Row reorders work on packed rows; out_proj is a column
            # reorder, so it alone needs float.
            self._dequant_suffixes = ("linear_attn.out_proj.weight",)
            if "linear_attn.out_proj" not in unquantized_modules:
                unquantized_modules.append("linear_attn.out_proj")

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

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        weights = gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source,
            self.load_spec.gguf_to_hf_name_map,
            dequant_suffixes=("embed_tokens.weight", "lm_head.weight")
            + self._dequant_suffixes,
        )
        yield from self.map_weights(weights)

    @staticmethod
    def _inv_reorder(t, dim, num_k, r, head_dim):
        # Undo llama.cpp's grouped->tiled V-head reorder along *dim*.
        shape = list(t.shape)
        if dim < 0:
            dim += len(shape)
        t = t.reshape(*shape[:dim], r, num_k, head_dim, *shape[dim + 1 :])
        t = t.transpose(dim, dim + 1)
        return t.reshape(*shape).contiguous()

    def _reorder_gdn(self, name: str, w: torch.Tensor) -> torch.Tensor | None:
        """Undo llama.cpp's grouped->tiled V-head reorder. Row reorders apply
        to packed ``qweight`` too, since GGUF quantizes per row."""
        rc = self._reorder
        nk, r, hk, hv = rc["num_k"], rc["r"], rc["head_k"], rc["head_v"]
        inv = self._inv_reorder
        if name.endswith("qweight_type"):
            return None
        base = name.removesuffix(".qweight").removesuffix(".weight")
        if base.endswith(".A_log"):
            return inv(torch.log(-w), 0, nk, r, 1)
        if base.endswith(".dt_bias"):
            return inv(w, 0, nk, r, 1)
        if base.endswith("linear_attn.in_proj_z"):
            return inv(w, 0, nk, r, hv)
        if base.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
            return inv(w, 0, nk, r, 1)
        if base.endswith("linear_attn.in_proj_qkv"):
            qk = hk * nk * 2  # q + k rows are unchanged; only V rows reorder
            return torch.cat([w[:qk], inv(w[qk:], 0, nk, r, hv)], dim=0)
        if base.endswith("linear_attn.out_proj"):
            return inv(w, 1, nk, r, hv)  # column (input) reorder, dequantized
        if base.endswith("linear_attn.conv1d") and w.dim() == 2:
            qk = hk * nk * 2
            return torch.cat([w[:qk], inv(w[qk:], 0, nk, r, hv)], dim=0).unsqueeze(1)
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
        patch_embed_parts: dict[str, torch.Tensor] = {}
        for name, weight in weights:
            # Forced-unquantized modules keep plain params.
            if "qweight" in name and ("lm_head." in name or "embed_tokens." in name):
                continue
            if self._reorder is not None:
                # Also folds A_log's log(-a) recovery in the right order.
                out = self._reorder_gdn(name, weight)
                if out is not None:
                    yield name, out
                    continue
            if name.endswith(".A_log"):
                # GGUF stores A = -exp(A_log); recover A_log for the model.
                yield name, torch.log(-weight)
                continue
            if (
                name.endswith("norm.weight")
                and not name.endswith("linear_attn.norm.weight")
                and "visual" not in name
            ):
                # GGUF conversion bakes (w + 1) into these RMSNorm weights.
                yield name, weight - 1
                continue
            if temporal_patch_size > 1 and "patch_embed.proj.weight" in name:
                # GGUF holds one 2D conv per temporal frame; stack back to 5D.
                base, split, _ = name.rpartition(".weight.")
                key = f"{base}.weight" if split else name
                patch_embed_parts[name] = weight
                parts = [patch_embed_parts.get(key)] + [
                    patch_embed_parts.get(f"{key}.{i}")
                    for i in range(1, temporal_patch_size)
                ]
                if any(part is None for part in parts):
                    continue
                yield key, torch.stack(parts, dim=2)
                continue
            if "conv1d.weight" in name and weight.dim() == 2:
                # depthwise Conv1d: [d, k] -> [d, 1, k]
                weight = weight.unsqueeze(1)
            elif name.endswith(".weight") and weight.dim() == 1 and "norm" not in name:
                # GGUF flattens [1, hidden] linears (e.g. shared expert gate).
                weight = weight.unsqueeze(0)
            yield name, weight
