# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from .quantization import GGUFConfig
from .weight_utils import download_gguf, resolve_local_gguf
from .weights_adapter import get_weights_adapter

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


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

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
                revision=model_config.revision,
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

    def _prepare_adapter(self, model_config: ModelConfig):
        local_model_path = self._prepare_weights(model_config)
        adapter = get_weights_adapter(model_config.hf_config)
        adapter.prepare_loading(local_model_path, model_config)
        return adapter

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        adapter = self._prepare_adapter(model_config)
        _load_weights_strict(model, adapter.prepare_weights(model_config))

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        adapter = self._prepare_adapter(model_config)
        vllm_config.model_config.hf_config = model_config.hf_config
        logger.debug(
            "GGUF unquantized modules: %s", adapter.load_spec.unquantized_modules
        )
        vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(
            adapter.load_spec.unquantized_modules
        )

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)
            _load_weights_strict(
                model,
                adapter.prepare_weights(model_config),
            )
            process_weights_after_loading(model, model_config, target_device)
            _normalize_kimi_k3_mla_context_parallelism(model)
        return model
