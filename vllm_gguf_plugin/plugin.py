# SPDX-License-Identifier: Apache-2.0

from functools import cached_property, wraps
from pathlib import Path

from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.transformers_utils.config import get_config_parser, register_config_parser

from .config_parser import GGUFConfigParser
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


def _name_the_drafts_config_source(engine_args) -> bool:
    """Fill in ``hf_config_path`` for a separate-file GGUF draft.

    A ``.gguf`` path carries no config of its own, and the directory a draft
    sits in belongs to the model it drafts for.  ``SpeculativeConfig`` takes
    ``hf_config_path`` for exactly this case, so the only thing left to do is
    derive the default when the caller did not name one, matching what
    ``create_model_config`` derives for the target.

    Reports whether the draft was given a config source of its own.
    """
    speculative_config = engine_args.speculative_config
    if not isinstance(speculative_config, dict):
        return False
    draft_model = speculative_config.get("model")
    if not _is_gguf_reference(draft_model):
        return False

    source = _get_gguf_config_source(
        draft_model, None, speculative_config.get("hf_config_path")
    )
    if source == draft_model:
        return False

    local = Path(source)
    if local.is_dir() and not (local / "config.json").exists():
        raise ValueError(
            f"The GGUF speculative draft {draft_model!r} needs a config, and "
            f"{source!r} does not contain config.json. GGUF files carry no "
            "config.json of their own, and the directory a draft sits in "
            "belongs to the model it drafts for. Pass the draft's own config "
            'directory as speculative_config={"hf_config_path": ...}.'
        )

    speculative_config["hf_config_path"] = source
    if speculative_config.get("quantization") is None:
        # Without this the draft builds unquantized layers and is then handed
        # packed bytes.  The target gets the same treatment in
        # ``create_model_config``.
        speculative_config["quantization"] = "gguf"
    return True


def _open_the_drafts_declaration_channel(draft_model_config) -> None:
    """Give the loader a dict to record the draft's declaration in.

    A draft resolves its quantization config separately from the target, by
    rebuilding it from ``hf_config.quantization_config``.  Nothing the loader
    records on the config object the target's layers were built against can
    reach it, so that dict is the only channel for the one thing a draft has to
    be told before its own layers are built: which of its modules the GGUF file
    stores unquantized.  ``_publish_declaration_for_a_draft`` writes the
    declaration there, and can only do so if the dict is already present.

    The contents planted here do not matter -- ``GGUFConfig.from_config`` reads
    only the declaration, which the loader has not made yet.
    """
    hf_config = draft_model_config.hf_config
    if getattr(hf_config, "quantization_config", None) is None:
        hf_config.quantization_config = {"quant_method": "gguf"}


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
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True

    original_create_speculative_config = EngineArgs.create_speculative_config

    @wraps(original_create_speculative_config)
    def create_speculative_config(self, *args, **kwargs):
        configured_model = getattr(self, "spec_model", None)
        if self.speculative_config is not None:
            configured_model = configured_model or self.speculative_config.get("model")

        draft_names_its_own_config = _name_the_drafts_config_source(self)

        config = original_create_speculative_config(self, *args, **kwargs)
        gguf_model = self.model_weights
        if (
            config is not None
            and config.method == "mtp"
            and configured_model is None
            and _is_gguf_reference(gguf_model)
        ):
            config.draft_model_config.model_weights = gguf_model
        if config is not None and draft_names_its_own_config:
            _open_the_drafts_declaration_channel(config.draft_model_config)
        return config

    EngineArgs.create_speculative_config = create_speculative_config


def _gguf_unsupported_modalities(model_config) -> tuple[str, ...]:
    if getattr(model_config, "quantization", None) != "gguf":
        return ()
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return ()

    from .weights_adapter import get_weights_adapter

    return tuple(get_weights_adapter(hf_config).UNSUPPORTED_MODALITIES)


def _patch_mm_limits() -> None:
    """Hide modalities an adapter cannot reconstruct from its GGUF weights.

    Wrapping the base ``supported_mm_limits`` rather than each model's
    ``get_supported_mm_limits`` covers every subclass at once, and every
    consumer -- request validation, the advertised modality list, and profiling
    -- reads that one property. Dropping a modality from it makes vLLM reject
    those requests with its usual validation error.
    """
    from vllm.multimodal.processing.context import BaseProcessingInfo

    if getattr(BaseProcessingInfo, "_gguf_mm_limits_patched", False):
        return

    original = BaseProcessingInfo.supported_mm_limits.func

    @wraps(original)
    def supported_mm_limits(self):
        limits = original(self)
        unsupported = _gguf_unsupported_modalities(self.ctx.model_config)
        if not unsupported:
            return limits
        return {
            modality: limit
            for modality, limit in limits.items()
            if modality not in unsupported
        }

    patched = cached_property(supported_mm_limits)
    BaseProcessingInfo.supported_mm_limits = patched
    # Assigning a cached_property after class creation skips __set_name__, which
    # is what tells it which attribute to cache under.
    patched.__set_name__(BaseProcessingInfo, "supported_mm_limits")
    BaseProcessingInfo._gguf_mm_limits_patched = True


def _register_omni_diffusion_quantization() -> None:
    try:
        from vllm_omni.quantization import register_quantization_override
    except ImportError:
        return

    register_quantization_override("gguf", lambda **kw: DiffusionGGUFConfig(**kw))


def register() -> None:
    """Register the out-of-tree GGUF integration."""
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
    _patch_mm_limits()
    _patch_diffusers_loader()
