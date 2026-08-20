# SPDX-License-Identifier: Apache-2.0

import inspect
import os
from functools import wraps
from pathlib import Path

import vllm.engine.arg_utils as arg_utils_module
import vllm.envs as envs_module
import vllm.transformers_utils.config as config_module
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.model_executor.models import ModelRegistry
from vllm.transformers_utils.config import get_config_parser, register_config_parser

try:
    from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
except ImportError:
    # The installed vLLM predates native Kimi-K3 support; the plugin stays
    # usable for other models, only Kimi-K3 GGUF loading is unavailable.
    KimiK3Config = None

from .config_parser import (
    KIMI_K3_GGUF_TEXT_ARCH,
    KIMI_K3_GGUF_TEXT_MARKER,
    GGUFConfigParser,
)
from .gguf_utils import check_gguf_file, is_gguf, is_remote_gguf, split_remote_gguf
from .loader import GGUFModelLoader
from .quantization import DiffusionGGUFConfig, GGUFConfig
from .weights_adapter.diffusion.integration import _patch_diffusers_loader

OOTGGUFConfig = GGUFConfig
OOTGGUFModelLoader = GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    return model.endswith(".gguf") or is_remote_gguf(model) or is_gguf(model)


def _get_gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str:
    if hf_config_path is not None:
        return hf_config_path
    if tokenizer is not None and not _is_gguf_reference(tokenizer):
        return tokenizer
    if is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        return repo_id
    if check_gguf_file(model):
        return str(Path(model).parent)
    return model


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
            self.model = _get_gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
        model_config = original_create_model_config(self, *args, **kwargs)
        if (
            getattr(
                getattr(model_config, "hf_config", None),
                KIMI_K3_GGUF_TEXT_MARKER,
                False,
            )
            and getattr(model_config, "tokenizer_mode", None) == "auto"
        ):
            model_config.tokenizer_mode = "kimi_k3"
        return model_config

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True

    original_create_speculative_config = EngineArgs.create_speculative_config

    @wraps(original_create_speculative_config)
    def create_speculative_config(self, *args, **kwargs):
        configured_model = getattr(self, "spec_model", None)
        if self.speculative_config is not None:
            configured_model = configured_model or self.speculative_config.get("model")

        config = original_create_speculative_config(self, *args, **kwargs)
        gguf_model = self.model_weights
        if (
            config is not None
            and config.method == "mtp"
            and configured_model is None
            and _is_gguf_reference(gguf_model)
        ):
            config.draft_model_config.model_weights = gguf_model
        return config

    EngineArgs.create_speculative_config = create_speculative_config


def _patch_speculator_probe() -> None:
    if getattr(arg_utils_module, "_gguf_speculator_probe_patched", False):
        return

    original_maybe_override = arg_utils_module.maybe_override_with_speculators

    @wraps(original_maybe_override)
    def maybe_override_with_speculators(model, tokenizer, *args, **kwargs):
        if _is_gguf_reference(model):
            return model, tokenizer, kwargs.get("vllm_speculative_config")
        return original_maybe_override(model, tokenizer, *args, **kwargs)

    arg_utils_module.maybe_override_with_speculators = maybe_override_with_speculators
    config_module.maybe_override_with_speculators = maybe_override_with_speculators
    arg_utils_module._gguf_speculator_probe_patched = True
    config_module._gguf_speculator_probe_patched = True


def _patch_kimi_k3_mla_cache_spec() -> None:
    """Bridge the incomplete Kimi-K3 cache-spec merge in pinned vLLM.

    Kimi's regular MLA path passes ``non_causal_multi_token_decode=False`` to
    ``MLAAttentionSpec``, but that dataclass does not yet define the field in
    the pinned vLLM revision. The optional DSpark draft path passes ``True``;
    reject that case rather than silently dropping behavior vLLM cannot
    represent.
    """
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    if (
        "non_causal_multi_token_decode"
        in inspect.signature(MLAAttentionSpec).parameters
    ):
        return
    if getattr(MLAAttentionSpec, "_gguf_kimi_k3_compat_patched", False):
        return

    original_init = MLAAttentionSpec.__init__

    @wraps(original_init)
    def compat_init(
        self,
        *args,
        non_causal_multi_token_decode: bool = False,
        **kwargs,
    ):
        if non_causal_multi_token_decode:
            raise NotImplementedError(
                "The pinned vLLM MLAAttentionSpec cannot represent "
                "non-causal multi-token decode."
            )
        return original_init(self, *args, **kwargs)

    MLAAttentionSpec.__init__ = compat_init
    MLAAttentionSpec._gguf_kimi_k3_compat_patched = True


def _patch_kimi_k3_environment() -> None:
    """Restore the Kimi-K3 stream threshold omitted from pinned vLLM."""
    name = "VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD"
    if name not in envs_module.environment_variables:
        envs_module.environment_variables[name] = lambda: int(os.getenv(name, "256"))


def _register_omni_diffusion_quantization() -> None:
    try:
        from vllm_omni.quantization import register_quantization_override
    except ImportError:
        return

    register_quantization_override("gguf", lambda **kw: DiffusionGGUFConfig(**kw))


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    # vLLM carries the native Kimi-K3 config/model implementation, but the
    # pinned revision does not include kimi_k3 in _CONFIG_REGISTRY. Registering
    # it here makes HFConfigParser ignore the repository auto_map and keeps
    # trust_remote_code disabled.
    if KimiK3Config is not None:
        config_module._CONFIG_REGISTRY["kimi_k3"] = KimiK3Config
    ModelRegistry.register_model(
        KIMI_K3_GGUF_TEXT_ARCH,
        "vllm.models.kimi_k3:KimiLinearForCausalLM",
    )
    register_quantization_config("gguf")(GGUFConfig)
    _register_omni_diffusion_quantization()

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
    _patch_kimi_k3_mla_cache_spec()
    _patch_kimi_k3_environment()
    _patch_diffusers_loader()
