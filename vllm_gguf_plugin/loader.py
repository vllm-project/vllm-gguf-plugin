# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple, cast

import torch
import torch.nn as nn
from huggingface_hub import ResolvedRevision, hf_hub_download
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from .gguf_files import GGUFModelFiles
from .gguf_utils import (
    detect_gguf_multimodal,
    get_remote_gguf_repo_id,
    resolve_explicit_mm_proj,
)
from .quantization import GGUFConfig, recursive_replace_vocab_modules
from .weight_utils import (
    download_gguf,
    download_mmproj,
    get_gguf_shard_files,
    get_gguf_tensor_names,
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
    resolve_local_gguf,
)
from .weights_adapter import BaseGGUFWeightsAdapter, get_weights_adapter

if TYPE_CHECKING:
    from .quantization.layout import GGUFLinearLayout

logger = init_logger(__name__)


def _load_weights_strict(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load weights and reject parameters silently skipped by model loaders."""
    loaded = model.load_weights(weights)
    if loaded is None:
        raise ValueError(
            f"{model.__class__.__name__}.load_weights() did not report which "
            "parameters were initialized; strict GGUF loading requires a set."
        )

    loaded = set(loaded)
    expected = {name for name, _ in model.named_parameters()}
    missing = expected - loaded
    if missing:
        formatted = "\n  ".join(sorted(missing))
        raise ValueError(
            "Following model parameters were not initialized from the GGUF "
            f"checkpoint ({len(missing)}):\n  {formatted}"
        )

    logger.info(
        "Strict GGUF weight audit passed: %d model parameters initialized.",
        len(expected),
    )
    return loaded


def _normalize_kimi_k3_mla_context_parallelism(model: nn.Module) -> None:
    """Resolve Kimi-K3's disabled-CP sentinel for the FA3 decode kernel."""
    for module in model.modules():
        if (
            module.__class__.__module__ == "vllm.models.kimi_k3.nvidia.mla"
            and module.__class__.__name__ == "MultiHeadLatentAttention"
            and getattr(module.impl, "dcp_world_size", -1) < 1
        ):
            # Kimi-K3's fused MLA wrapper rejects context parallelism but calls
            # the backend directly, bypassing the regular MLAAttention wrapper
            # that resolves this sentinel. FA3 requires the neutral value 1.
            module.impl.dcp_world_size = 1
            module.impl.dcp_rank = 0


class GGUFLoadPlan(NamedTuple):
    files: GGUFModelFiles
    name_map: dict[str, str]
    unquantized_modules: tuple[str, ...]
    linear_layouts: dict[str, GGUFLinearLayout]


def _revision_for_weights_repo(
    model_config: ModelConfig, weights_repo_id: str
) -> str | None:
    """Return a revision that is valid for the weights repository.

    A ``ResolvedRevision`` is bound to the repository for which vLLM loaded
    the model config. If the GGUF weights live in another repository, reuse
    the user's original revision instead of that repository's resolved SHA.
    An explicitly requested SHA remains unchanged, while ``None`` lets the
    weights repository resolve its own default branch.
    """
    revision = model_config.revision
    if isinstance(revision, ResolvedRevision) and weights_repo_id != model_config.model:
        return revision.initial
    return revision


def _get_unquantized_modules(
    files: GGUFModelFiles,
    name_map: dict[str, str],
) -> tuple[str, ...]:
    unquantized_tensors = get_gguf_unquantized_params(list(files.all_files))
    modules = {
        mapped_name.removesuffix(".weight")
        for gguf_name in unquantized_tensors
        if (mapped_name := name_map.get(gguf_name)) is not None
        and mapped_name.endswith(".weight")
    }
    return tuple(sorted(modules))


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config or {}
        unknown_options = set(extra_config) - {"mm_proj"}
        if unknown_options:
            raise ValueError(
                "Unsupported GGUF model loader extra config options: "
                f"{sorted(unknown_options)}"
            )
        self._mm_proj_reference = extra_config.get("mm_proj")

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model_weights or model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # local_dir:quant_type (e.g. /path/to/gguf-dir:Q8_0)
        if ":" in model_name_or_path:
            local_dir, quant_type = model_name_or_path.rsplit(":", 1)
            if os.path.isdir(local_dir):
                return resolve_local_gguf(local_dir, quant_type)
            # remote repo_id:quant_type
            return download_gguf(
                local_dir,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=_revision_for_weights_repo(model_config, local_dir),
                ignore_patterns=self.load_config.ignore_patterns,
            )
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, <local_dir>:<quant_type>, "
            "<repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
        )

    def _prepare_model_files(self, model_config: ModelConfig) -> GGUFModelFiles:
        """Resolve backbone shards and the optional multimodal projector."""
        model_path = self._prepare_weights(model_config)

        mm_proj = None
        if self._mm_proj_reference is not None:
            mm_proj = resolve_explicit_mm_proj(
                self._mm_proj_reference,
                model_path,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
            )
        elif detected := detect_gguf_multimodal(model_path):
            mm_proj = str(detected)
        elif getattr(model_config.hf_config, "vision_config", None) is not None:
            repo_id = get_remote_gguf_repo_id(
                model_config.model_weights or model_config.model
            )
            if repo_id is not None:
                mm_proj = download_mmproj(
                    repo_id,
                    cache_dir=self.load_config.download_dir,
                    revision=model_config.revision,
                )

        return GGUFModelFiles(
            backbone=tuple(get_gguf_shard_files(model_path)),
            mm_proj=mm_proj,
        )

    def _prepare_adapter(
        self, model_config: ModelConfig
    ) -> tuple[BaseGGUFWeightsAdapter, GGUFLoadPlan]:
        files = self._prepare_model_files(model_config)
        adapter = get_weights_adapter(model_config.hf_config)
        model_config.hf_config = adapter.patch_hf_config(
            files,
            model_config.hf_config,
        )

        text_config = model_config.hf_config.get_text_config()
        backbone_names = get_gguf_tensor_names(files.backbone)
        text_config.update(
            {"tie_word_embeddings": "output.weight" not in backbone_names}
        )

        name_map = adapter.build_name_map(files, model_config)
        unquantized_modules = _get_unquantized_modules(files, name_map) + tuple(
            adapter.extra_unquantized_modules
        )
        unquantized_modules += tuple(
            adapter.extend_unquantized_modules(files, name_map, unquantized_modules)
        )
        linear_layouts = adapter.get_linear_layouts(files, model_config, name_map)
        return adapter, GGUFLoadPlan(
            files, name_map, unquantized_modules, linear_layouts
        )

    def _iter_weights(
        self,
        adapter: BaseGGUFWeightsAdapter,
        plan: GGUFLoadPlan,
        model_config: ModelConfig,
    ):
        weights = gguf_quant_weights_iterator_multi(
            list(plan.files.all_files),
            plan.name_map,
        )
        return adapter.transform_weights(weights, model_config)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_model_files(model_config)

    @staticmethod
    def _load_model_weights(
        model: nn.Module,
        adapter: BaseGGUFWeightsAdapter,
        weights,
    ) -> None:
        if adapter.strict_weight_audit:
            _load_weights_strict(model, weights)
        else:
            model.load_weights(weights)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        adapter, plan = self._prepare_adapter(model_config)
        self._load_model_weights(
            model, adapter, self._iter_weights(adapter, plan, model_config)
        )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        adapter, plan = self._prepare_adapter(model_config)
        logger.debug("GGUF unquantized modules: %s", plan.unquantized_modules)
        vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(plan.unquantized_modules)
        vllm_config.quant_config.register_linear_layouts(
            plan.linear_layouts,
            prefix=prefix,
        )

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
                recursive_replace_vocab_modules(
                    model,
                    vllm_config.quant_config,
                    prefix=prefix,
                )
            self._load_model_weights(
                model,
                adapter,
                self._iter_weights(adapter, plan, model_config),
            )
            process_weights_after_loading(model, model_config, target_device)
            _normalize_kimi_k3_mla_context_parallelism(model)
        return model
