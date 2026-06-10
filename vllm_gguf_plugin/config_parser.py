# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase
from vllm.transformers_utils.repo_utils import file_or_path_exists

from .gguf_utils import (
    check_gguf_file,
    get_gguf_file_path_from_hf,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    resolve_gguf_config_source,
    split_remote_gguf,
)


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
        gguf_path = None
        if (gguf_file := kwargs.pop("gguf_file", None)) is not None:
            candidate = Path(model) / gguf_file
            if check_gguf_file(candidate):
                original_model = candidate
                gguf_path = candidate

        resolved_model = self._resolve_config_source(model, revision=revision)

        if gguf_path is not None or check_gguf_file(model):
            gguf_path = gguf_path or Path(model)
            resolved_model = self._resolve_config_source(
                gguf_path,
                revision=revision,
            )
            gguf_repo = gguf_path.parent
            if resolved_model != gguf_repo:
                trust_remote_code = False
                revision = None
            elif not file_or_path_exists(gguf_repo, HF_CONFIG_NAME, revision=revision):
                kwargs["gguf_file"] = gguf_path.name
        elif is_remote_gguf(model):
            repo_id, quant_type = split_remote_gguf(model)
            if resolved_model != repo_id:
                trust_remote_code = False
                revision = None
            elif not file_or_path_exists(repo_id, HF_CONFIG_NAME, revision=revision):
                kwargs["gguf_file"] = get_gguf_file_path_from_hf(
                    repo_id,
                    quant_type,
                    revision=revision,
                )

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

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        if config.model_type in ("gemma4_assistant", "gemma4_mtp"):
            config_dict["architectures"] = ["Gemma4MTPModel"]
            config.update({"architectures": ["Gemma4MTPModel"]})
            return config_dict, config

        if config.architectures:
            config_dict["architectures"] = list(config.architectures)
            return config_dict, config

        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config_dict["architectures"] = [model_type]
        config.update({"architectures": [model_type]})

        return config_dict, config

    @staticmethod
    def _resolve_config_source(
        model: str | Path,
        revision: str | None = None,
    ) -> str | Path:
        return resolve_gguf_config_source(model, revision=revision)
