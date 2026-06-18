# SPDX-License-Identifier: Apache-2.0

import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any

import huggingface_hub
import vllm.config.model as model_config_module
import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.quantization as quantization_module
import vllm.transformers_utils.config as config_module
from transformers import PretrainedConfig
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    register_quantization_config,
)
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    register_model_loader,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config_parser, register_config_parser
from vllm.transformers_utils.repo_utils import file_or_path_exists
from vllm.transformers_utils.utils import without_trust_remote_code

from .gguf_tokenizer_builder import build_tokenizer_from_gguf
from .gguf_utils import (
    check_gguf_file,
    get_gguf_file_path_from_hf,
    is_gguf,
    is_local_gguf_sidecar_source,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    resolve_gguf_config_source,
    split_remote_gguf,
)
from .weight_utils import split_remote_gguf_file_ref

logger = init_logger(__name__)

GGUFConfig: Any | None = None
GGUFConfigParser: Any | None = None
GGUFModelLoader: Any | None = None
OOTGGUFConfig: Any | None = None
OOTGGUFModelLoader: Any | None = None


def _load_oot_gguf_classes() -> tuple[type, type, type]:
    """Load plugin classes lazily to avoid custom-op registration on import.

    Importing ``vllm_gguf_plugin.quantization`` registers plugin custom ops.
    On vLLM builds that still include in-tree GGUF, those names already exist
    and importing the plugin package can fail before ``register()`` has a chance
    to decide whether the plugin should be active. Keep these imports behind the
    registration decision.
    """
    global GGUFConfig, GGUFConfigParser, GGUFModelLoader
    global OOTGGUFConfig, OOTGGUFModelLoader

    if GGUFConfig is None:
        from .config_parser import GGUFConfigParser as _GGUFConfigParser
        from .loader import GGUFModelLoader as _GGUFModelLoader
        from .quantization import GGUFConfig as _GGUFConfig

        GGUFConfig = _GGUFConfig
        GGUFConfigParser = _GGUFConfigParser
        GGUFModelLoader = _GGUFModelLoader
        OOTGGUFConfig = _GGUFConfig
        OOTGGUFModelLoader = _GGUFModelLoader

    assert GGUFConfig is not None
    assert GGUFConfigParser is not None
    assert GGUFModelLoader is not None
    return GGUFConfig, GGUFConfigParser, GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    return model.endswith(".gguf") or is_remote_gguf(model) or is_gguf(model)


def _get_explicit_gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str | None:
    if hf_config_path is not None:
        return hf_config_path
    if tokenizer is not None and not _is_gguf_reference(tokenizer):
        return tokenizer
    return None


def _uses_gguf_derived_config_source(
    model: str | None,
    revision: str | None = None,
) -> bool:
    if not _is_gguf_reference(model):
        return False

    if check_gguf_file(model):
        gguf_path = Path(model)
        resolved_source = resolve_gguf_config_source(model, revision=revision)
        if not is_local_gguf_sidecar_source(gguf_path, resolved_source):
            return True
        return not file_or_path_exists(resolved_source, HF_CONFIG_NAME, revision)

    if is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        resolved_source = resolve_gguf_config_source(model, revision=revision)
        if resolved_source != repo_id:
            return True
        return not file_or_path_exists(repo_id, HF_CONFIG_NAME, revision)

    remote_file_ref = split_remote_gguf_file_ref(str(model))
    if remote_file_ref is not None:
        repo_id, _ = remote_file_ref
        resolved_source = resolve_gguf_config_source(model, revision=revision)
        if resolved_source != repo_id:
            return True
        return not file_or_path_exists(repo_id, HF_CONFIG_NAME, revision)

    return False


def _get_implicit_gguf_config_source(
    model: str,
    revision: str | None = None,
) -> str | None:
    if check_gguf_file(model):
        gguf_source: str | Path = Path(model).parent
    elif is_remote_gguf(model):
        gguf_source, _ = split_remote_gguf(model)
    elif (remote_file_ref := split_remote_gguf_file_ref(str(model))) is not None:
        gguf_source, _ = remote_file_ref
    else:
        return None

    resolved_source = resolve_gguf_config_source(model, revision=revision)
    if resolved_source == gguf_source:
        return None
    return str(resolved_source)


def _get_gguf_config_probe_model(engine_args: EngineArgs) -> str | None:
    if _is_gguf_reference(engine_args.model):
        return engine_args.model

    speculative_config = getattr(engine_args, "speculative_config", None)
    if isinstance(speculative_config, dict):
        speculative_model = speculative_config.get("model")
        if isinstance(speculative_model, str) and _is_gguf_reference(speculative_model):
            return speculative_model

    return None


def _maybe_set_blackwell_gguf_dtype(engine_args: EngineArgs) -> None:
    if engine_args.dtype != "auto":
        return
    if not current_platform.has_device_capability(100):
        return

    logger.warning_once(
        "Defaulting GGUF `dtype=auto` to `float16` on Blackwell because "
        "bfloat16 GGUF kernels are disabled for precision issues. Pass an "
        "explicit `--dtype` to override this default.",
    )
    engine_args.dtype = "float16"


def _patch_quantization_config_lookup() -> None:
    if getattr(quantization_module, "_gguf_config_lookup_patched", False):
        return

    assert GGUFConfig is not None
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


def _patch_engine_args() -> None:
    if getattr(EngineArgs, "_gguf_create_model_config_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if (
            self.trust_remote_code
            and self.hf_config_path is None
            and (gguf_config_model := _get_gguf_config_probe_model(self)) is not None
            and _uses_gguf_derived_config_source(
                gguf_config_model,
                revision=self.revision,
            )
        ):
            config_module.logger.warning_once(
                "Disabling `trust_remote_code` because model config was "
                "selected from a GGUF-derived config source. Pass an "
                "explicit `--hf-config-path` to opt in for that repository.",
            )
            self.trust_remote_code = False

        if _is_gguf_reference(self.model):
            gguf_model = self.model
            _maybe_set_blackwell_gguf_dtype(self)
            explicit_tokenizer = (
                self.tokenizer if isinstance(self.tokenizer, str) else None
            )
            if (
                self.tokenizer is None
                and check_gguf_file(gguf_model)
                and (tokenizer_path := build_tokenizer_from_gguf(gguf_model))
            ):
                self.tokenizer = tokenizer_path
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
            config_source = _get_explicit_gguf_config_source(
                gguf_model,
                explicit_tokenizer,
                self.hf_config_path,
            )
            if config_source is None:
                config_source = _get_implicit_gguf_config_source(
                    gguf_model,
                    revision=self.revision,
                )
            if config_source is not None:
                self.model = config_source
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True


def _patch_gguf_config_helpers() -> None:
    if getattr(model_config_module, "_gguf_config_helpers_patched", False):
        return

    # ModelConfig imports this helper by value, so update that binding too.
    model_config_module.maybe_patch_hf_config_from_gguf = (
        maybe_patch_hf_config_from_gguf
    )
    model_config_module._gguf_config_helpers_patched = True


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

        remote_file_ref = split_remote_gguf_file_ref(str(model))
        if check_gguf_file(model):
            if hf_config_path is None:
                gguf_path = Path(model)
                gguf_model_repo = resolve_gguf_config_source(
                    model,
                    revision=revision,
                )
                if not is_local_gguf_sidecar_source(gguf_path, gguf_model_repo):
                    revision = None
                elif not file_or_path_exists(
                    gguf_model_repo,
                    HF_CONFIG_NAME,
                    revision=revision,
                ):
                    kwargs["gguf_file"] = gguf_path.name
            else:
                gguf_model_repo = Path(model).parent
        elif is_remote_gguf(model):
            repo_id, quant_type = split_remote_gguf(model)
            gguf_model_repo = resolve_gguf_config_source(model, revision=revision)
            if gguf_model_repo != repo_id:
                revision = None
            elif not file_or_path_exists(repo_id, HF_CONFIG_NAME, revision=revision):
                kwargs["gguf_file"] = get_gguf_file_path_from_hf(
                    repo_id,
                    quant_type,
                    revision=revision,
                )
        elif remote_file_ref is not None:
            repo_id, filename = remote_file_ref
            gguf_model_repo = resolve_gguf_config_source(model, revision=revision)
            if gguf_model_repo != repo_id:
                revision = None
            elif not file_or_path_exists(repo_id, HF_CONFIG_NAME, revision=revision):
                kwargs["gguf_file"] = filename
        else:
            return model, tokenizer, vllm_speculative_config

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
            return model, tokenizer, vllm_speculative_config

        from vllm.transformers_utils.configs.speculators.base import (
            SpeculatorsConfig,
        )

        speculative_config = SpeculatorsConfig.extract_vllm_speculative_config(
            config_dict=config_dict
        )
        speculative_config["model"] = model

        verifier_model = speculators_config["verifier"]["name_or_path"]
        return verifier_model, verifier_model, speculative_config

    arg_utils_module.maybe_override_with_speculators = maybe_override_with_speculators
    config_module.maybe_override_with_speculators = maybe_override_with_speculators
    arg_utils_module._gguf_speculator_probe_patched = True
    config_module._gguf_speculator_probe_patched = True


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    if (
        "gguf" in QUANTIZATION_METHODS
        and os.environ.get("VLLM_GGUF_PLUGIN_OVERRIDE_IN_TREE") != "1"
    ):
        logger.warning(
            "Skipping vllm-gguf-plugin registration because vLLM already has "
            "a GGUF quantization method. Set VLLM_GGUF_PLUGIN_OVERRIDE_IN_TREE=1 "
            "to force plugin registration."
        )
        return

    GGUFConfig, GGUFConfigParser, GGUFModelLoader = _load_oot_gguf_classes()

    register_quantization_config("gguf")(GGUFConfig)
    _patch_quantization_config_lookup()

    if _LOAD_FORMAT_TO_MODEL_LOADER.get("gguf") is not GGUFModelLoader:
        register_model_loader("gguf")(GGUFModelLoader)

    try:
        parser = get_config_parser("gguf")
    except ValueError:
        parser = None
    if not isinstance(parser, GGUFConfigParser):
        register_config_parser("gguf")(GGUFConfigParser)
    _patch_gguf_config_helpers()
    _patch_engine_args()
    _patch_speculator_probe()
