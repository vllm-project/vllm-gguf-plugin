# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
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

from .gguf_utils import find_nextn_block_index
from .quantization import GGUFConfig
from .weight_utils import download_gguf, resolve_local_gguf
from .weights_adapter import GGUFWeightsAdapter, get_weights_adapter

logger = init_logger(__name__)


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
        model.load_weights(adapter.prepare_weights(model_config))

    @staticmethod
    def _prefetch_mtp_weights(model_config: ModelConfig) -> str | None:
        """Fetch only the ``mtp.*`` safetensors shards into the HF cache,
        avoiding the full (tens of GiB) checkpoint. Returns the snapshot dir,
        or None to let the default loader download normally."""
        import json

        repo = str(model_config.model)
        if os.path.isdir(repo):
            return None
        try:
            index_path = hf_hub_download(
                repo, "model.safetensors.index.json", revision=model_config.revision
            )
        except Exception as e:
            # No index (single-file checkpoint), or the fetch failed — the
            # caller then downloads the whole repo, so say why.
            logger.info("No MTP shard index for %s (%s)", repo, e)
            return None
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({f for name, f in weight_map.items() if "mtp" in name.lower()})
        if not shards:
            return None
        logger.info("Prefetching %d MTP shard(s): %s", len(shards), shards)
        for shard in shards:
            hf_hub_download(repo, shard, revision=model_config.revision)
        return os.path.dirname(index_path)

    def _gguf_has_mtp(self, target_mc: ModelConfig) -> bool:
        """Whether the target's GGUF carries the MTP/nextn draft block."""
        files = GGUFWeightsAdapter._get_all_gguf_files(self._prepare_weights(target_mc))
        return find_nextn_block_index(files) is not None

    def _load_hf_draft(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str
    ) -> nn.Module:
        """Load an MTP draft from the HF safetensors repo in native dtype,
        clearing GGUF quantization for this load only."""
        from vllm.model_executor.model_loader import get_model_loader

        logger.info(
            "Loading speculative draft '%s' from HF repo %s (not in GGUF)",
            getattr(model_config.hf_config, "model_type", "?"),
            model_config.model,
        )
        saved_quant_config = vllm_config.quant_config
        saved_quantization = model_config.quantization
        saved_model = model_config.model
        vllm_config.quant_config = None
        model_config.quantization = None
        model_config.model_weights = None  # use model_config.model (HF repo)
        # Point the loader at a snapshot with only the MTP shards when possible.
        mtp_dir = self._prefetch_mtp_weights(model_config)
        if mtp_dir is not None:
            model_config.model = mtp_dir
        try:
            loader = get_model_loader(LoadConfig(load_format="auto"))
            return loader.load_model(
                vllm_config=vllm_config, model_config=model_config, prefix=prefix
            )
        finally:
            vllm_config.quant_config = saved_quant_config
            model_config.quantization = saved_quantization
            model_config.model = saved_model

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        # A draft ModelConfig has no GGUF reference of its own.
        target_mc = vllm_config.model_config
        is_draft = model_config is not target_mc
        if is_draft and not model_config.model_weights:
            model_type = getattr(model_config.hf_config, "model_type", "") or ""
            # Only unsloth's *-MTP-GGUF keeps the nextn block.
            if "mtp" in model_type and not self._gguf_has_mtp(target_mc):
                return self._load_hf_draft(vllm_config, model_config, prefix)
            model_config.model_weights = target_mc.model_weights
        adapter = self._prepare_adapter(model_config)
        if not is_draft:
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
                # vllm_config.model_config stays the target's even for a draft.
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            model.load_weights(
                adapter.prepare_weights(model_config),
            )
            process_weights_after_loading(model, model_config, target_device)
        return model
