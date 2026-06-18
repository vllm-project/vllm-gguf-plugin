# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import gguf
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase
from vllm.transformers_utils.repo_utils import file_or_path_exists

from .gguf_utils import (
    _gguf_reader_value,
    _gguf_scalar_value,
    check_gguf_file,
    get_gguf_file_path_from_hf,
    is_gguf,
    is_local_gguf_sidecar_source,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    resolve_gguf_config_source,
    split_remote_gguf,
)
from .weight_utils import first_split_gguf_filename, split_remote_gguf_file_ref


def _gguf_int(reader: gguf.GGUFReader, key: str) -> int:
    value = _gguf_scalar_value(_gguf_reader_value(reader, key))
    if value is None:
        raise ValueError(f"Missing required GGUF metadata key: {key}")
    return int(value)


def _gguf_float(reader: gguf.GGUFReader, key: str) -> float:
    value = _gguf_scalar_value(_gguf_reader_value(reader, key))
    if value is None:
        raise ValueError(f"Missing required GGUF metadata key: {key}")
    return float(value)


def _gguf_str(reader: gguf.GGUFReader, key: str) -> str | None:
    value = _gguf_scalar_value(_gguf_reader_value(reader, key))
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _gguf_list(reader: gguf.GGUFReader, key: str) -> list[Any]:
    value = _gguf_reader_value(reader, key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def _is_gpt_oss_gguf(reader: gguf.GGUFReader) -> bool:
    return _gguf_str(reader, "general.architecture") == "gpt-oss"


def _parse_gpt_oss_gguf_config(
    gguf_path: str | Path,
) -> tuple[dict[str, Any], PretrainedConfig] | None:
    try:
        reader = gguf.GGUFReader(str(gguf_path))
    except Exception:
        return None
    if not _is_gpt_oss_gguf(reader):
        return None

    num_hidden_layers = _gguf_int(reader, "gpt-oss.block_count")
    original_context_length = _gguf_int(
        reader, "gpt-oss.rope.scaling.original_context_length"
    )
    rope_scaling = {
        "rope_type": _gguf_str(reader, "gpt-oss.rope.scaling.type"),
        "factor": _gguf_float(reader, "gpt-oss.rope.scaling.factor"),
        "original_max_position_embeddings": original_context_length,
        "truncate": False,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }

    tokens = _gguf_list(reader, "tokenizer.ggml.tokens")
    eos_token_id = _gguf_scalar_value(
        _gguf_reader_value(reader, "tokenizer.ggml.eos_token_id")
    )
    pad_token_id = _gguf_scalar_value(
        _gguf_reader_value(reader, "tokenizer.ggml.padding_token_id")
    )

    config_dict: dict[str, Any] = {
        "architectures": ["GptOssForCausalLM"],
        "attention_bias": True,
        "attention_dropout": 0.0,
        "eos_token_id": int(eos_token_id) if eos_token_id is not None else None,
        "experts_per_token": _gguf_int(reader, "gpt-oss.expert_used_count"),
        "head_dim": _gguf_int(reader, "gpt-oss.attention.key_length"),
        "hidden_act": "silu",
        "hidden_size": _gguf_int(reader, "gpt-oss.embedding_length"),
        "initial_context_length": original_context_length,
        "initializer_range": 0.02,
        "intermediate_size": _gguf_int(reader, "gpt-oss.expert_feed_forward_length"),
        "layer_types": [
            "sliding_attention" if idx % 2 == 0 else "full_attention"
            for idx in range(num_hidden_layers)
        ],
        "max_position_embeddings": _gguf_int(reader, "gpt-oss.context_length"),
        "model_type": "gpt_oss",
        "num_attention_heads": _gguf_int(reader, "gpt-oss.attention.head_count"),
        "num_experts_per_tok": _gguf_int(reader, "gpt-oss.expert_used_count"),
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": _gguf_int(reader, "gpt-oss.attention.head_count_kv"),
        "num_local_experts": _gguf_int(reader, "gpt-oss.expert_count"),
        "output_router_logits": False,
        "pad_token_id": int(pad_token_id) if pad_token_id is not None else None,
        "rms_norm_eps": _gguf_float(reader, "gpt-oss.attention.layer_norm_rms_epsilon"),
        "rope_scaling": rope_scaling,
        "rope_theta": _gguf_float(reader, "gpt-oss.rope.freq_base"),
        "router_aux_loss_coef": 0.9,
        "sliding_window": _gguf_int(reader, "gpt-oss.attention.sliding_window"),
        "swiglu_limit": 7.0,
        "tie_word_embeddings": all(
            tensor.name != "output.weight" for tensor in reader.tensors
        ),
        "use_cache": True,
        "vocab_size": len(tokens),
    }
    config = AutoConfig.for_model(
        "gpt_oss",
        **{key: value for key, value in config_dict.items() if key != "model_type"},
    )
    return config_dict, config


def _local_gguf_fallback_path(
    gguf_path: Path | None,
    resolved_model: str | Path,
    gguf_file: str | None,
) -> Path | None:
    if gguf_path is not None:
        return gguf_path
    if gguf_file is None:
        return None
    if not isinstance(resolved_model, Path):
        return None
    candidate = resolved_model / gguf_file
    return candidate if check_gguf_file(candidate) else None


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
            if not is_local_gguf_sidecar_source(gguf_path, resolved_model):
                trust_remote_code = False
                revision = None
            elif not file_or_path_exists(resolved_model, HF_CONFIG_NAME, revision):
                kwargs["gguf_file"] = Path(first_split_gguf_filename(gguf_path)).name
        elif is_remote_gguf(model):
            repo_id, quant_type = split_remote_gguf(model)
            if resolved_model != repo_id:
                trust_remote_code = False
                revision = None
            elif not file_or_path_exists(repo_id, HF_CONFIG_NAME, revision):
                kwargs["gguf_file"] = get_gguf_file_path_from_hf(
                    repo_id,
                    quant_type,
                    revision=revision,
                )
        elif (remote_file_ref := split_remote_gguf_file_ref(str(model))) is not None:
            repo_id, filename = remote_file_ref
            if resolved_model != repo_id:
                trust_remote_code = False
                revision = None
            elif not file_or_path_exists(repo_id, HF_CONFIG_NAME, revision):
                kwargs["gguf_file"] = first_split_gguf_filename(filename)

        fallback_path = _local_gguf_fallback_path(
            gguf_path, resolved_model, kwargs.get("gguf_file")
        )
        fallback = (
            _parse_gpt_oss_gguf_config(fallback_path)
            if fallback_path is not None
            and not file_or_path_exists(resolved_model, HF_CONFIG_NAME, revision)
            else None
        )

        if fallback is not None:
            config_dict, config = fallback
        else:
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

        if getattr(config, "architectures", None):
            config_dict["architectures"] = config.architectures
        elif config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
            config_dict["architectures"] = [model_type]
            config.update({"architectures": [model_type]})
        else:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config

    @staticmethod
    def _resolve_config_source(
        model: str | Path,
        revision: str | None = None,
    ) -> str | Path:
        return resolve_gguf_config_source(model, revision=revision)
