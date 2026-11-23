# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    detect_gguf_multimodal,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)
from .weights_adapter import get_adapter_architecture

# Set by plugin._patch_engine_args when model is rewritten from GGUF path to HF source
_last_gguf_path: str | None = None


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        original_model = model
        resolved_model = self._resolve_config_source(model)
        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )

        if config.model_type == "qwen3_moe" and "norm_topk_prob" not in config_dict:
            config_dict["norm_topk_prob"] = True
            config.update({"norm_topk_prob": True})

        architecture = get_adapter_architecture(config)
        if (
            architecture is None
            and config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        ):
            architecture = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        if architecture is None:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        config_dict["architectures"] = [architecture]
        config.update({"architectures": [architecture]})

        # Determine GGUF path for vision stripping: original_model or preserved global
        _gguf_path = None
        if is_gguf(original_model):
            _gguf_path = str(original_model)
            config = maybe_patch_hf_config_from_gguf(_gguf_path, config)
        elif _last_gguf_path is not None and is_gguf(_last_gguf_path):
            _gguf_path = str(_last_gguf_path)

        # v0.34 regression: ModelConfig/MultiModalBudget is created early
        # from HFConfig. If HF repo has vision_config but GGUF has no
        # vision tensors (text-only, e.g. Heretic), strip vision_config
        # here so the engine does not try to load a multimodal processor.
        # This must happen in config_parser (before ModelConfig), not only
        # in adapter.patch_hf_config (too late). The GGUF path is available
        # via _last_gguf_path when EngineArgs rewrote model to HF source.
        if _gguf_path is not None:
            try:
                has_vision = getattr(config, "vision_config", None) is not None
                if has_vision:
                    mm = None
                    try:
                        mm = detect_gguf_multimodal(_gguf_path)
                    except Exception:
                        mm = None
                    if mm is None:
                        for k in ("vision_config", "vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id"):
                            config_dict.pop(k, None)
                            if hasattr(config, k):
                                try:
                                    setattr(config, k, None)
                                except Exception:
                                    pass
                        try:
                            if hasattr(config, "vision_config"):
                                config.vision_config = None
                        except Exception:
                            pass
                        tc = config_dict.get("text_config")
                        if isinstance(tc, dict):
                            for k in ("vision_config",):
                                tc.pop(k, None)
                        # Downgrade qwen3_5 multimodal to text-only so renderer/model
                        # does not try to create a multimodal processor.
                        try:
                            if getattr(config, "model_type", None) == "qwen3_5":
                                from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig as _TC
                                import inspect as _ins
                                _d = config.to_dict()
                                for kk in ("vision_config", "vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id"):
                                    _d.pop(kk, None)
                                _tc = _d.pop("text_config", None)
                                if isinstance(_tc, dict):
                                    for kk, vv in _tc.items():
                                        if kk.startswith("linear_") or kk not in _d or _d.get(kk) is None:
                                            _d[kk] = vv
                                _d["model_type"] = "qwen3_5_text"
                                _sig = _ins.signature(_TC.__init__)
                                _valid = set(_sig.parameters.keys()) - {"self", "args", "kwargs"}
                                _filtered = {k: v for k, v in _d.items() if k in _valid}
                                config = _TC(**_filtered)
                                config_dict = config.to_dict()
                                # recompute architecture as text-only
                                _arch = get_adapter_architecture(config)
                                if _arch:
                                    config_dict["architectures"] = [_arch]
                                    try:
                                        config.update({"architectures": [_arch]})
                                    except Exception:
                                        config.architectures = [_arch]
                        except Exception:
                            pass
            except Exception:
                pass

        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model
