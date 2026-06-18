# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable

import torch

from ..quantization.nvfp4 import split_gguf_nvfp4_moe_weight
from .base import GGUFLoadSpec
from .default import GGUFWeightsAdapter

_QWEIGHT_SUFFIX = ".qweight"
_QWEIGHT_TYPE_SUFFIX = ".qweight_type"
_WEIGHT_SCALE_2_SUFFIX = ".weight_scale_2"
_INPUT_SCALE_SUFFIX = ".input_scale"
_GEMMA4_EXPERT_GATE_UP_SUFFIX = ".experts.gate_up_proj"
_GEMMA4_EXPERT_DOWN_SUFFIX = ".experts.down_proj"
_GEMMA4_EXPERT_SUFFIX_TO_SHARD = {
    _GEMMA4_EXPERT_GATE_UP_SUFFIX: "w13",
    _GEMMA4_EXPERT_DOWN_SUFFIX: "w2",
}


def _gemma4_native_moe_module_prefix(prefix: str) -> str:
    # This shorter FusedMoE prefix matches recent vLLM builds and also matches
    # nested routed-experts prefixes through is_layer_skipped_gguf substring
    # matching.
    return f"{prefix}.moe.experts"


def _flatten_gemma4_patch_embed_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.flatten(1) if weight.dim() == 4 else weight


def _transform_gemma4_weight_name(name: str) -> str:
    replacements = {
        ".experts.gate_up_proj.qweight_type": (
            ".moe.experts.routed_experts.w13_qweight_type"
        ),
        ".experts.gate_up_proj.qweight": (".moe.experts.routed_experts.w13_qweight"),
        ".experts.down_proj.qweight_type": (
            ".moe.experts.routed_experts.w2_qweight_type"
        ),
        ".experts.down_proj.qweight": ".moe.experts.routed_experts.w2_qweight",
    }
    for old, new in replacements.items():
        if name.endswith(old):
            return name.removesuffix(old) + new
    return name


def _gemma4_packed_expert_module(module_name: str) -> tuple[str, str] | None:
    for suffix, shard in _GEMMA4_EXPERT_SUFFIX_TO_SHARD.items():
        if module_name.endswith(suffix):
            return module_name.removesuffix(suffix), shard
    return None


def _duplicate_w13_sidecar(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(torch.float32)
    if weight.ndim == 0:
        return weight.reshape(1, 1).expand(1, 2).contiguous()
    if weight.ndim >= 2 and weight.shape[-1:] == (2,):
        return weight.reshape(-1, 2).contiguous()
    return weight.reshape(-1, 1).expand(-1, 2).contiguous()


def _default_sidecar(num_experts: int, shard: str) -> torch.Tensor:
    shape = (num_experts, 2) if shard == "w13" else (num_experts,)
    return torch.ones(shape, dtype=torch.float32)


class Gemma4GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for Gemma4 GGUF models."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._native_nvfp4_gemma4_moe_projection_modules: dict[str, str] = {}

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in ("gemma4", "gemma4_assistant", "gemma4_mtp")

    def prepare_loading(
        self,
        model_path: str,
        model_config,
    ) -> GGUFLoadSpec:
        load_spec = super().prepare_loading(model_path, model_config)
        self._promote_native_nvfp4_moe_modules(load_spec)
        return load_spec

    def _promote_native_nvfp4_moe_modules(self, load_spec: GGUFLoadSpec) -> None:
        self._native_nvfp4_gemma4_moe_projection_modules.clear()

        packed_by_prefix: dict[str, set[str]] = {}
        for module_name in list(self._native_nvfp4_modules):
            packed = _gemma4_packed_expert_module(module_name)
            if packed is None:
                continue
            prefix, shard = packed
            packed_by_prefix.setdefault(prefix, set()).add(shard)

        for prefix, shards in packed_by_prefix.items():
            for suffix in _GEMMA4_EXPERT_SUFFIX_TO_SHARD:
                module_name = f"{prefix}{suffix}"
                self._native_nvfp4_modules.discard(module_name)
                if module_name in load_spec.nvfp4_modules:
                    load_spec.nvfp4_modules.remove(module_name)

            if {"w13", "w2"}.issubset(shards):
                native_prefix = _gemma4_native_moe_module_prefix(prefix)
                self._native_nvfp4_moe_modules.add(native_prefix)
                if native_prefix not in load_spec.nvfp4_moe_modules:
                    load_spec.nvfp4_moe_modules.append(native_prefix)
                for suffix, shard in _GEMMA4_EXPERT_SUFFIX_TO_SHARD.items():
                    self._native_nvfp4_gemma4_moe_projection_modules[
                        f"{prefix}{suffix}"
                    ] = shard

        load_spec.nvfp4_modules.sort()
        load_spec.nvfp4_moe_modules.sort()

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            handled = False
            for suffix in (_WEIGHT_SCALE_2_SUFFIX, _INPUT_SCALE_SUFFIX):
                if not name.endswith(suffix):
                    continue
                module_name = name.removesuffix(suffix)
                shard = self._native_nvfp4_gemma4_moe_projection_modules.get(
                    module_name
                )
                if shard is None:
                    break
                native_prefix = _gemma4_native_moe_module_prefix(
                    module_name.rsplit(".experts.", 1)[0]
                )
                native_name = f"{native_prefix}.{shard}_{suffix.removeprefix('.')}"
                native_weight = (
                    _duplicate_w13_sidecar(weight) if shard == "w13" else weight
                )
                yield native_name, self.transform_weight(native_name, native_weight)
                handled = True
                break
            if handled:
                continue

            if name.endswith(_QWEIGHT_TYPE_SUFFIX):
                module_name = name.removesuffix(_QWEIGHT_TYPE_SUFFIX)
                if module_name in self._native_nvfp4_gemma4_moe_projection_modules:
                    continue

            if name.endswith(_QWEIGHT_SUFFIX):
                module_name = name.removesuffix(_QWEIGHT_SUFFIX)
                shard = self._native_nvfp4_gemma4_moe_projection_modules.get(
                    module_name
                )
                if shard is not None:
                    weight, weight_scale = split_gguf_nvfp4_moe_weight(weight)
                    native_prefix = _gemma4_native_moe_module_prefix(
                        module_name.rsplit(".experts.", 1)[0]
                    )
                    native_name = f"{native_prefix}.{shard}_weight"
                    yield (
                        native_name,
                        self.transform_weight(native_name, weight),
                    )
                    native_name = f"{native_prefix}.{shard}_weight_scale"
                    yield (
                        native_name,
                        self.transform_weight(native_name, weight_scale),
                    )

                    sidecars = self._native_nvfp4_sidecar_suffixes.get(
                        module_name, set()
                    )
                    num_experts = weight.shape[0] if weight.ndim >= 3 else 1
                    if "weight_scale_2" not in sidecars:
                        native_name = f"{native_prefix}.{shard}_weight_scale_2"
                        native_weight = _default_sidecar(num_experts, shard)
                        yield (
                            native_name,
                            self.transform_weight(native_name, native_weight),
                        )
                    if "input_scale" not in sidecars:
                        native_name = f"{native_prefix}.{shard}_input_scale"
                        native_weight = _default_sidecar(num_experts, shard)
                        yield (
                            native_name,
                            self.transform_weight(native_name, native_weight),
                        )
                    continue

            transformed_name = _transform_gemma4_weight_name(name)
            yield from super().map_weights(((transformed_name, weight),))

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if hf_name == "model.vision_tower.patch_embedder.input_proj.weight":
            return _flatten_gemma4_patch_embed_weight(weight)
        return weight
