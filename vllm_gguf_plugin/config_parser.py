# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    is_text_only_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)
from .gguf_context import get_explicit_mm_proj, get_gguf_weights
from .weights_adapter import get_adapter_architecture


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

        weights_reference = get_gguf_weights() or original_model
        text_only = is_text_only_gguf(weights_reference, get_explicit_mm_proj())
        architecture = get_adapter_architecture(config, text_only)
        if (
            architecture is None
            and config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        ):
            architecture = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        if architecture is None:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        if text_only:
            text_config = config.get_text_config()
            if text_config is not config:
                # A multimodal HF config for a backbone with no vision tower:
                # keep only the text half so the causal LM is built from it.
                config = text_config
                config_dict = config.to_dict()

        config_dict["architectures"] = [architecture]
        config.update({"architectures": [architecture]})

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model
