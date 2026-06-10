# SPDX-License-Identifier: Apache-2.0

import sys
from functools import wraps
from pathlib import Path

import huggingface_hub
import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.quantization as quantization_module
import vllm.transformers_utils.config as config_module
from transformers import PretrainedConfig
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME
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
from vllm.transformers_utils.repo_utils import file_or_path_exists
from vllm.transformers_utils.utils import without_trust_remote_code

from .config_parser import GGUFConfigParser
from .gguf_utils import (
    check_gguf_file,
    get_gguf_file_path_from_hf,
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
    def maybe_override_with_speculators(
        model,
        tokenizer,
        trust_remote_code,
        revision=None,
        hf_config_path=None,
        vllm_speculative_config=None,
        hf_token=None,
        **kwargs,
    ):
        if not _is_gguf_reference(model):
            return original_maybe_override(
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=trust_remote_code,
                revision=revision,
                hf_config_path=hf_config_path,
                vllm_speculative_config=vllm_speculative_config,
                hf_token=hf_token,
                **kwargs,
            )

        disable_verifier_trust_remote_code = False

        def maybe_disable_trust_remote_code() -> bool:
            if not disable_verifier_trust_remote_code or not trust_remote_code:
                return trust_remote_code
            config_module.logger.warning_once(
                "Disabling `trust_remote_code` because model config was "
                "selected from a GGUF-derived config source. Pass an "
                "explicit `--hf-config-path` to opt in for that repository.",
            )
            return False

        if check_gguf_file(model):
            if hf_config_path is None:
                gguf_repo = Path(model).parent
                gguf_model_repo = resolve_gguf_config_source(
                    model,
                    revision=revision,
                )
                if gguf_model_repo != gguf_repo:
                    revision = None
                    disable_verifier_trust_remote_code = True
                elif not file_or_path_exists(
                    gguf_repo,
                    HF_CONFIG_NAME,
                    revision=revision,
                ):
                    kwargs["gguf_file"] = Path(model).name
                    disable_verifier_trust_remote_code = True
            else:
                gguf_model_repo = Path(model).parent
        elif is_remote_gguf(model):
            repo_id, quant_type = split_remote_gguf(model)
            gguf_model_repo = resolve_gguf_config_source(model, revision=revision)
            if gguf_model_repo != repo_id:
                revision = None
                disable_verifier_trust_remote_code = True
            elif not file_or_path_exists(repo_id, HF_CONFIG_NAME, revision=revision):
                kwargs["gguf_file"] = get_gguf_file_path_from_hf(
                    repo_id,
                    quant_type,
                    revision=revision,
                )
                disable_verifier_trust_remote_code = True
        else:
            return (
                model,
                tokenizer,
                vllm_speculative_config,
                maybe_disable_trust_remote_code(),
            )

        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
        config_source = hf_config_path or gguf_model_repo
        config_dict, _ = PretrainedConfig.get_config_dict(
            config_source,
            revision=revision,
            token=hf_token,
            **without_trust_remote_code(kwargs),
        )
        speculators_config = config_dict.get("speculators_config")
        if speculators_config is None:
            return (
                model,
                tokenizer,
                vllm_speculative_config,
                maybe_disable_trust_remote_code(),
            )

        from vllm.transformers_utils.configs.speculators.base import (
            SpeculatorsConfig,
        )

        speculative_config = SpeculatorsConfig.extract_vllm_speculative_config(
            config_dict=config_dict
        )
        speculative_config["model"] = model

        verifier_model = speculators_config["verifier"]["name_or_path"]
        return (
            verifier_model,
            verifier_model,
            speculative_config,
            maybe_disable_trust_remote_code(),
        )

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
