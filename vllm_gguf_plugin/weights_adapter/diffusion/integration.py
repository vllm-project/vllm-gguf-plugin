# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patch DiffusersPipelineLoader for OOT GGUF diffusion support.

When both vllm-omni and vllm-gguf-plugin are installed, this module
patches ``DiffusersPipelineLoader.load_weights`` so that GGUF quantized
models are loaded through the plugin's GGUF path.  vllm-omni contains
zero GGUF-specific code — all logic lives here.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from .loader import (
    DiffusionWeightSource,
    get_gguf_model_from_config,
    is_gguf_quant_config,
    load_diffusion_gguf_weights,
    resolve_gguf_model_path,
)

if TYPE_CHECKING:
    from torch import nn

logger = init_logger(__name__)


def _extend_unquantized_modules(
    quant_config: object,
    module_names: tuple[str, ...],
) -> None:
    if not module_names:
        return

    if isinstance(quant_config, dict):
        unquantized_modules = quant_config.setdefault("unquantized_modules", [])
    else:
        unquantized_modules = getattr(quant_config, "unquantized_modules", None)
        if unquantized_modules is None:
            unquantized_modules = []
            quant_config.unquantized_modules = unquantized_modules

    for module_name in module_names:
        if module_name not in unquantized_modules:
            unquantized_modules.append(module_name)


def _patch_diffusers_loader() -> None:
    """Patch ``DiffusersPipelineLoader.load_weights`` with GGUF support.

    Idempotent — safe to call multiple times.
    """
    try:
        from vllm_omni.diffusion.model_loader.diffusers_loader import (
            DiffusersPipelineLoader,
        )
    except ImportError:
        logger.debug("vllm-omni not installed, skipping diffusion GGUF patch")
        return

    if getattr(
        DiffusersPipelineLoader.load_weights, "_gguf_plugin_patched", False
    ) and getattr(DiffusersPipelineLoader.load_model, "_gguf_plugin_patched", False):
        return

    _original_load_weights = DiffusersPipelineLoader.load_weights
    _original_load_model = DiffusersPipelineLoader.load_model

    @wraps(_original_load_weights)
    def _gguf_load_weights(self: object, model: nn.Module) -> None:
        """Load weights using GGUF when the quant config is GGUF."""
        if not is_gguf_quant_config(self.quant_config):
            return _original_load_weights(self, model)

        gguf_model = get_gguf_model_from_config(self.quant_config)
        if not gguf_model:
            raise ValueError("GGUF quantization requires gguf_model")

        sources = self._get_weight_sources(model)
        dw_sources = [
            DiffusionWeightSource(prefix=s.prefix, subfolder=s.subfolder)
            for s in sources
        ]
        model_type = (
            self.od_config.tf_model_config.get("model_type")
            if self.od_config.tf_model_config is not None
            else None
        )

        def _hf_fn(dw_src: DiffusionWeightSource):
            for s in sources:
                if s.prefix == dw_src.prefix and s.subfolder == dw_src.subfolder:
                    return self._get_weights_iterator(s)
            return iter(())

        loaded = load_diffusion_gguf_weights(
            gguf_model=gguf_model,
            model=model,
            model_class_name=self.od_config.model_class_name,
            model_type=model_type,
            sources=dw_sources,
            hf_weights_fn=_hf_fn,
            revision=self.od_config.revision,
            download_dir=self.load_config.download_dir,
            ignore_patterns=self.load_config.ignore_patterns,
        )

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "GGUF weight loading took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )

        weights_to_load = self._get_expected_parameter_names(model)
        weights_not_loaded = weights_to_load - loaded
        if weights_not_loaded:
            raise ValueError(
                f"Following weights were not initialized from "
                f"checkpoint: {weights_not_loaded}"
            )

    @wraps(_original_load_model)
    def _gguf_load_model(self, *args, **kwargs):
        """Patched load_model that uses GGUF weight loading when appropriate.

        For GGUF models, the normal weight-loading path is replaced with
        GGUF loading. For non-GGUF models, the original path is used.
        """
        # Check if GGUF before any init — if not, delegate to original entirely
        if not is_gguf_quant_config(self.quant_config):
            return _original_load_model(self, *args, **kwargs)

        # GGUF path: reuse original for model init, intercept weight loading
        load_device = kwargs.get("load_device", args[0] if args else "auto")
        load_format = kwargs.get("load_format", args[1] if len(args) > 1 else None)
        custom_pipeline_name = kwargs.get(
            "custom_pipeline_name", args[2] if len(args) > 2 else None
        )
        device = kwargs.get("device", args[3] if len(args) > 3 else None)

        if load_format is None:
            load_format = "default"

        import torch

        from . import get_diffusion_gguf_adapter

        target_device = torch.device(load_device)
        gguf_model = get_gguf_model_from_config(self.quant_config)
        if not gguf_model:
            raise ValueError("GGUF quantization requires gguf_model")
        model_type = (
            self.od_config.tf_model_config.get("model_type")
            if self.od_config.tf_model_config is not None
            else None
        )
        gguf_file = resolve_gguf_model_path(
            gguf_model=gguf_model,
            revision=self.od_config.revision,
            download_dir=self.load_config.download_dir,
            ignore_patterns=self.load_config.ignore_patterns,
        )
        adapter = get_diffusion_gguf_adapter(
            gguf_file, self.od_config.model_class_name, model_type
        )
        _extend_unquantized_modules(
            self.quant_config, adapter.unquantized_module_names()
        )

        # Handle CPU offload — same logic as original load_model
        offload_after_quant = False
        if (
            load_device == "cpu"
            and self.quant_config is not None
            and device is not None
        ):
            quant_cfg = self.quant_config
            is_offline = getattr(quant_cfg, "data_type", None) == "mx_fp" or getattr(
                quant_cfg, "is_checkpoint_quantized", False
            )
            if not is_offline:
                offload_after_quant = True
                logger.info(
                    "Online quantization with CPU offload, using %s for "
                    "weight loading (will offload back to CPU)",
                    device.type,
                )
            else:
                logger.info(
                    "Offline-quantized model with CPU offload, "
                    "loading weights directly on CPU"
                )

        with set_default_torch_dtype(self.od_config.dtype):
            if self.parallel_config.use_hsdp:
                model = self._load_model_with_hsdp(
                    target_device=device,
                    load_format=load_format,
                    custom_pipeline_name=custom_pipeline_name,
                )
            else:
                model = self._init_from_load_format(
                    load_format, target_device, custom_pipeline_name, is_hsdp=False
                )
                logger.debug("Loading weights on %s ...", load_device)
                # Use the patched load_weights which handles GGUF
                self.load_weights(model)

            self._process_weights_after_loading(model, target_device)

            if offload_after_quant:
                model.to("cpu")
                logger.info("Quantization complete, offloaded model back to CPU")

        return model.eval()

    _gguf_load_weights._gguf_plugin_patched = True
    _gguf_load_model._gguf_plugin_patched = True
    DiffusersPipelineLoader.load_weights = _gguf_load_weights
    DiffusersPipelineLoader.load_model = _gguf_load_model
    DiffusersPipelineLoader._gguf_load_weights_patched = True
    logger.debug("Patched DiffusersPipelineLoader for GGUF diffusion support")
