# SPDX-License-Identifier: Apache-2.0

"""Build Hugging Face configs directly from GGUF metadata.

This is a best-effort path for GGUF files that do not ship a sidecar
``config.json`` and whose model card has no usable base model fallback.
Unsupported or incomplete metadata returns ``None`` so the existing HF/GGUF
fallback chain remains unchanged.
"""

from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any

import gguf
from transformers import AutoConfig, PretrainedConfig
from vllm.logger import init_logger

from .gguf_utils import (
    _gguf_reader_value,
    _gguf_scalar_value,
    _gguf_sequence_edge,
    check_gguf_file,
    detect_gguf_multimodal,
)

logger = init_logger(__name__)


def _decode_string(value: Any) -> str | None:
    value = _gguf_scalar_value(value)
    if isinstance(value, bytes):
        with suppress(UnicodeDecodeError):
            return value.decode("utf-8")
        return None
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value)


def _read(reader: gguf.GGUFReader, key: str) -> Any:
    return _gguf_scalar_value(_gguf_reader_value(reader, key))


def _read_int(reader: gguf.GGUFReader, key: str) -> int | None:
    value = _read(reader, key)
    if value is None:
        return None
    with suppress(TypeError, ValueError):
        return int(value)
    return None


def _read_float(reader: gguf.GGUFReader, key: str) -> float | None:
    value = _read(reader, key)
    if value is None:
        return None
    with suppress(TypeError, ValueError):
        return float(value)
    return None


def _read_bool(reader: gguf.GGUFReader, key: str) -> bool | None:
    value = _read(reader, key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    with suppress(TypeError, ValueError):
        return bool(value)
    return None


def _read_list(reader: gguf.GGUFReader, key: str) -> list[Any] | None:
    value = _gguf_reader_value(reader, key)
    if value is None or isinstance(value, (str, bytes)):
        return None
    with suppress(AttributeError):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    return [_gguf_scalar_value(item) for item in value]


def _token_count(reader: gguf.GGUFReader) -> int | None:
    tokens = _gguf_reader_value(reader, "tokenizer.ggml.tokens")
    if tokens is None:
        return None
    with suppress(TypeError):
        return len(tokens)
    return None


def _has_lm_head(reader: gguf.GGUFReader) -> bool:
    return any(getattr(tensor, "name", None) == "output.weight" for tensor in reader.tensors)


def _non_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _gguf_architecture(reader: gguf.GGUFReader) -> str | None:
    return _decode_string(_gguf_reader_value(reader, "general.architecture"))


def _token_id_fields(reader: gguf.GGUFReader) -> dict[str, Any]:
    return _non_none(
        {
            "bos_token_id": _read_int(reader, "tokenizer.ggml.bos_token_id"),
            "eos_token_id": _read_int(reader, "tokenizer.ggml.eos_token_id"),
            "pad_token_id": _read_int(reader, "tokenizer.ggml.padding_token_id"),
        }
    )


def _sliding_layer_types(
    pattern: list[Any] | None,
    *,
    num_layers: int | None,
) -> list[str] | None:
    if pattern:
        return [
            "sliding_attention" if bool(is_sliding) else "full_attention"
            for is_sliding in pattern
        ]
    if num_layers is None:
        return None
    return ["sliding_attention"] * num_layers


def _qwen35_layer_types(
    interval: int | None,
    *,
    num_layers: int | None,
) -> list[str] | None:
    if interval is None or interval <= 0 or num_layers is None:
        return None
    return [
        "full_attention" if (layer + 1) % interval == 0 else "linear_attention"
        for layer in range(num_layers)
    ]


def _build_qwen35moe_config(
    reader: gguf.GGUFReader,
    model_path: Path,
) -> PretrainedConfig:
    prefix = "qwen35moe"
    block_count = _read_int(reader, f"{prefix}.block_count")
    nextn_layers = _read_int(reader, f"{prefix}.nextn_predict_layers") or 0
    num_layers = None
    if block_count is not None:
        num_layers = max(block_count - nextn_layers, 0)

    head_dim = _read_int(reader, f"{prefix}.attention.key_length")
    rope_dim = _read_int(reader, f"{prefix}.rope.dimension_count")
    partial_rotary_factor = None
    if head_dim and rope_dim:
        partial_rotary_factor = rope_dim / head_dim

    full_attention_interval = _read_int(reader, f"{prefix}.full_attention_interval")
    tie_word_embeddings = not _has_lm_head(reader)
    text_config = _non_none(
        {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": _token_count(reader),
            "hidden_size": _read_int(reader, f"{prefix}.embedding_length"),
            "intermediate_size": _read_int(reader, f"{prefix}.feed_forward_length"),
            "num_hidden_layers": num_layers,
            "mtp_num_hidden_layers": nextn_layers or None,
            "num_nextn_predict_layers": nextn_layers or None,
            "num_attention_heads": _read_int(
                reader, f"{prefix}.attention.head_count"
            ),
            "num_key_value_heads": _read_int(
                reader, f"{prefix}.attention.head_count_kv"
            ),
            "head_dim": head_dim,
            "max_position_embeddings": _read_int(reader, f"{prefix}.context_length"),
            "rms_norm_eps": _read_float(
                reader, f"{prefix}.attention.layer_norm_rms_epsilon"
            ),
            "num_experts": _read_int(reader, f"{prefix}.expert_count"),
            "num_experts_per_tok": _read_int(
                reader, f"{prefix}.expert_used_count"
            ),
            "moe_intermediate_size": _read_int(
                reader, f"{prefix}.expert_feed_forward_length"
            ),
            "shared_expert_intermediate_size": _read_int(
                reader, f"{prefix}.expert_shared_feed_forward_length"
            ),
            "partial_rotary_factor": partial_rotary_factor,
            "rope_theta": _read_float(reader, f"{prefix}.rope.freq_base"),
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": _read_float(reader, f"{prefix}.rope.freq_base"),
                "partial_rotary_factor": partial_rotary_factor,
            },
            "layer_types": _qwen35_layer_types(
                full_attention_interval,
                num_layers=num_layers,
            ),
            "tie_word_embeddings": tie_word_embeddings,
            **_token_id_fields(reader),
        }
    )

    is_multimodal = detect_gguf_multimodal(str(model_path)) is not None
    architecture = (
        "Qwen3_5MoeForConditionalGeneration"
        if is_multimodal
        else "Qwen3_5MoeForCausalLM"
    )
    return AutoConfig.for_model(
        "qwen3_5_moe",
        text_config=text_config,
        architectures=[architecture],
        tie_word_embeddings=tie_word_embeddings,
        vocab_size=text_config.get("vocab_size"),
    )


def _build_gemma4_text_config(
    reader: gguf.GGUFReader,
    prefix: str,
) -> dict[str, Any]:
    block_count = _read_int(reader, f"{prefix}.block_count")
    sliding_pattern = _read_list(reader, f"{prefix}.attention.sliding_window_pattern")
    head_count_kv = _gguf_reader_value(reader, f"{prefix}.attention.head_count_kv")
    rope_theta = _read_float(reader, f"{prefix}.rope.freq_base")
    rope_theta_swa = _read_float(reader, f"{prefix}.rope.freq_base_swa")

    return _non_none(
        {
            "model_type": "gemma4_text",
            "vocab_size": _token_count(reader),
            "hidden_size": _read_int(reader, f"{prefix}.embedding_length"),
            "intermediate_size": _read_int(reader, f"{prefix}.feed_forward_length"),
            "num_hidden_layers": block_count,
            "num_attention_heads": _read_int(
                reader, f"{prefix}.attention.head_count"
            ),
            "num_key_value_heads": _gguf_sequence_edge(head_count_kv, first=True),
            "num_global_key_value_heads": _gguf_sequence_edge(
                head_count_kv,
                first=False,
            ),
            "head_dim": _read_int(reader, f"{prefix}.attention.key_length_swa"),
            "global_head_dim": _read_int(reader, f"{prefix}.attention.key_length"),
            "max_position_embeddings": _read_int(reader, f"{prefix}.context_length"),
            "rms_norm_eps": _read_float(
                reader, f"{prefix}.attention.layer_norm_rms_epsilon"
            ),
            "sliding_window": _read_int(
                reader, f"{prefix}.attention.sliding_window"
            ),
            "layer_types": _sliding_layer_types(
                sliding_pattern,
                num_layers=block_count,
            ),
            "attention_k_eq_v": _read_bool(
                reader, f"{prefix}.attention.shared_kv_layers"
            ),
            "attention_bias": False,
            "num_kv_shared_layers": _read_int(
                reader, f"{prefix}.attention.shared_kv_layers"
            ),
            "enable_moe_block": _read_int(reader, f"{prefix}.expert_count")
            is not None,
            "num_experts": _read_int(reader, f"{prefix}.expert_count"),
            "top_k_experts": _read_int(reader, f"{prefix}.expert_used_count"),
            "moe_intermediate_size": _read_int(
                reader, f"{prefix}.expert_feed_forward_length"
            ),
            "rope_local_base_freq": rope_theta_swa,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": rope_theta_swa,
                },
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": rope_theta,
                },
            },
            "tie_word_embeddings": not _has_lm_head(reader),
            **_token_id_fields(reader),
        }
    )


def _build_gemma4_config(
    reader: gguf.GGUFReader,
    model_path: Path,
) -> PretrainedConfig:
    text_config = _build_gemma4_text_config(reader, "gemma4")
    is_multimodal = detect_gguf_multimodal(str(model_path)) is not None
    architecture = (
        "Gemma4ForConditionalGeneration" if is_multimodal else "Gemma4ForCausalLM"
    )
    return AutoConfig.for_model(
        "gemma4",
        text_config=text_config,
        architectures=[architecture],
        tie_word_embeddings=text_config.get("tie_word_embeddings"),
        vocab_size=text_config.get("vocab_size"),
    )


def _build_gemma4_assistant_config(reader: gguf.GGUFReader) -> PretrainedConfig:
    prefix = "gemma4-assistant"
    text_config = _build_gemma4_text_config(reader, prefix)
    text_config.pop("model_type", None)
    nextn_layers = _read_int(reader, f"{prefix}.nextn_predict_layers")
    text_config.update(
        _non_none(
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": _read_int(
                    reader, f"{prefix}.embedding_length_out"
                ),
                "num_nextn_predict_layers": nextn_layers,
                "mtp_num_hidden_layers": nextn_layers,
                "n_predict": 1,
            }
        )
    )
    text_config.pop("model_type", None)
    return AutoConfig.for_model(
        "gemma4_assistant",
        **text_config,
        architectures=["Gemma4MTPModel"],
    )


_SIMPLE_ARCH_CONFIGS: dict[str, tuple[str, str]] = {
    "llama": ("llama", "LlamaForCausalLM"),
    "mistral": ("mistral", "MistralForCausalLM"),
    "qwen2": ("qwen2", "Qwen2ForCausalLM"),
    "qwen2moe": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "qwen3_moe": ("qwen3_moe", "Qwen3MoeForCausalLM"),
    "gemma2": ("gemma2", "Gemma2ForCausalLM"),
    "gemma3_text": ("gemma3_text", "Gemma3ForCausalLM"),
    "gemma3": ("gemma3_text", "Gemma3ForCausalLM"),
}


def _build_simple_causal_config(
    reader: gguf.GGUFReader,
    architecture: str,
) -> PretrainedConfig | None:
    mapped = _SIMPLE_ARCH_CONFIGS.get(architecture)
    if mapped is None:
        return None
    model_type, model_class = mapped
    prefix = architecture
    fields = _non_none(
        {
            "architectures": [model_class],
            "vocab_size": _token_count(reader),
            "hidden_size": _read_int(reader, f"{prefix}.embedding_length"),
            "intermediate_size": _read_int(reader, f"{prefix}.feed_forward_length"),
            "num_hidden_layers": _read_int(reader, f"{prefix}.block_count"),
            "num_attention_heads": _read_int(
                reader, f"{prefix}.attention.head_count"
            ),
            "num_key_value_heads": _read_int(
                reader, f"{prefix}.attention.head_count_kv"
            ),
            "head_dim": _read_int(reader, f"{prefix}.attention.key_length"),
            "max_position_embeddings": _read_int(reader, f"{prefix}.context_length"),
            "rms_norm_eps": _read_float(
                reader, f"{prefix}.attention.layer_norm_rms_epsilon"
            ),
            "rope_theta": _read_float(reader, f"{prefix}.rope.freq_base"),
            "tie_word_embeddings": not _has_lm_head(reader),
            **_token_id_fields(reader),
        }
    )
    return AutoConfig.for_model(model_type, **fields)


def build_config_from_gguf(model: str | PathLike) -> PretrainedConfig | None:
    """Build a HF config from local GGUF metadata if this plugin supports it."""
    model_path = Path(model)
    if not check_gguf_file(model_path):
        return None

    try:
        reader = gguf.GGUFReader(str(model_path))
    except Exception as e:
        logger.debug("Failed to read GGUF metadata from %s: %s", model_path, e)
        return None

    architecture = _gguf_architecture(reader)
    try:
        if architecture == "qwen35moe":
            return _build_qwen35moe_config(reader, model_path)
        if architecture == "gemma4":
            return _build_gemma4_config(reader, model_path)
        if architecture == "gemma4-assistant":
            return _build_gemma4_assistant_config(reader)
        if architecture is not None:
            return _build_simple_causal_config(reader, architecture)
    except Exception as e:
        logger.debug(
            "Failed to build native GGUF config for %s (%s): %s",
            model_path,
            architecture,
            e,
        )
        return None

    logger.debug("Unsupported GGUF architecture for native config: %s", architecture)
    return None
