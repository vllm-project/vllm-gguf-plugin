# SPDX-License-Identifier: Apache-2.0

import sys
from functools import wraps
from pathlib import Path

import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.quantization as quantization_module
import vllm.transformers_utils.config as config_module
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.transformers_utils.config import get_config_parser, register_config_parser

from .config_parser import GGUFConfigParser
from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    resolve_gguf_config_source,
    split_remote_gguf,
)
from .loader import GGUFModelLoader
from .quantization import GGUFConfig

OOTGGUFConfig = GGUFConfig
OOTGGUFModelLoader = GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    return model.endswith(".gguf") or is_remote_gguf(model) or is_gguf(model)


def _gguf_config_redirects_to_base_model(
    model: str,
    hf_config_path: str | None,
    revision: str | None,
) -> bool:
    if hf_config_path is not None:
        return False
    if check_gguf_file(model):
        return (
            resolve_gguf_config_source(model, revision=revision) != Path(model).parent
        )
    if is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        return resolve_gguf_config_source(model, revision=revision) != repo_id
    return False


def _patch_engine_args() -> None:
    if getattr(EngineArgs, "_gguf_create_model_config_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if _is_gguf_reference(self.model):
            gguf_model = self.model
            if self.quantization is None:
                self.quantization = "gguf"
            if self.load_format == "auto":
                self.load_format = "gguf"
            if self.config_format == "auto":
                self.config_format = "gguf"
            if not self.model_weights:
                self.model_weights = gguf_model
            if self.served_model_name is None:
                self.served_model_name = [gguf_model]
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True


def _patch_speculator_probe() -> None:
    if getattr(arg_utils_module, "_gguf_speculator_probe_patched", False):
        return

    original_maybe_override = arg_utils_module.maybe_override_with_speculators

    @wraps(original_maybe_override)
    def maybe_override_with_speculators(model, tokenizer, *args, **kwargs):
        if _is_gguf_reference(model):
            trust_remote_code = kwargs.get("trust_remote_code", False)
            revision = kwargs.get("revision")
            hf_config_path = kwargs.get("hf_config_path")
            if (
                trust_remote_code
                and isinstance(model, str)
                and _gguf_config_redirects_to_base_model(
                    model,
                    hf_config_path,
                    revision,
                )
            ):
                trust_remote_code = False
            return (
                model,
                tokenizer,
                kwargs.get("vllm_speculative_config"),
                trust_remote_code,
            )
        return original_maybe_override(model, tokenizer, *args, **kwargs)

    arg_utils_module.maybe_override_with_speculators = maybe_override_with_speculators
    config_module.maybe_override_with_speculators = maybe_override_with_speculators
    arg_utils_module._gguf_speculator_probe_patched = True
    config_module._gguf_speculator_probe_patched = True


def _patch_quantization_config_lookup() -> None:
    if getattr(quantization_module, "_gguf_config_lookup_patched", False):
        return

    original_get_quantization_config = quantization_module.get_quantization_config

    @wraps(original_get_quantization_config)
    def get_quantization_config(quantization: str):
        if quantization == "gguf":
            return GGUFConfig
        return original_get_quantization_config(quantization)

    quantization_module.get_quantization_config = get_quantization_config

    for module_name in ("vllm.model_executor.model_loader.weight_utils",):
        module = sys.modules.get(module_name)
        if (
            module is not None
            and getattr(module, "get_quantization_config", None)
            is original_get_quantization_config
        ):
            module.get_quantization_config = get_quantization_config

    quantization_module._gguf_config_lookup_patched = True


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    register_quantization_config("gguf")(GGUFConfig)
    _patch_quantization_config_lookup()

    if "gguf" not in _LOAD_FORMAT_TO_MODEL_LOADER or not isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), GGUFModelLoader
    ):
        register_model_loader("gguf")(GGUFModelLoader)

    try:
        parser = get_config_parser("gguf")
    except ValueError:
        parser = None
    if not isinstance(parser, GGUFConfigParser):
        register_config_parser("gguf")(GGUFConfigParser)
    _patch_engine_args()
    _patch_speculator_probe()
