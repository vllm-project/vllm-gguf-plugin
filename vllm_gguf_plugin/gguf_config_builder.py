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
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig

from .gguf_utils import (
    _gguf_reader_value,
    _gguf_scalar_value,
    _gguf_sequence_edge,
    _qwen35_text_config_updates_from_gguf,
    check_gguf_file,
    detect_gguf_multimodal,
)

logger = init_logger(__name__)

_GEMMA4_HF_DEFAULT_SOURCE = (
    "Gemma4 HF config defaults observed in google/gemma-3n-E4B-it "
    "and Unsloth Gemma4 GGUF sidecars"
)
_GEMMA4_DEFAULT_OUTPUT_LENGTH = 280
_GEMMA4_POOLING_KERNEL_SIZE = 3
_GEMMA4_POSITION_EMBEDDING_SIZE = 10240
_GEMMA4_ASSISTANT_N_PREDICT = 1


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
    return any(
        getattr(tensor, "name", None) == "output.weight" for tensor in reader.tensors
    )


def _read_int_or_warn_default(
    reader: gguf.GGUFReader,
    key: str,
    default: int,
    hf_field: str,
) -> int:
    value = _read_int(reader, key)
    if value is not None:
        return value
    logger.warning_once(
        "GGUF metadata key %s is missing; using %s=%s from %s.",
        key,
        hf_field,
        default,
        _GEMMA4_HF_DEFAULT_SOURCE,
    )
    return default


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


def _read_mmproj_reader(model_path: Path) -> gguf.GGUFReader | None:
    mmproj_path = detect_gguf_multimodal(str(model_path))
    if mmproj_path is None:
        return None
    try:
        return gguf.GGUFReader(str(mmproj_path))
    except Exception as e:
        logger.debug("Failed to read GGUF mmproj metadata from %s: %s", mmproj_path, e)
        return None


def _build_qwen35_vision_config(
    reader: gguf.GGUFReader | None,
) -> dict[str, Any] | None:
    if reader is None:
        return None

    image_size = _read_int(reader, "clip.vision.image_size")
    patch_size = _read_int(reader, "clip.vision.patch_size")
    num_position_embeddings = None
    if image_size and patch_size:
        num_position_embeddings = (image_size // patch_size) ** 2

    vision_config = _non_none(
        {
            "depth": _read_int(reader, "clip.vision.block_count"),
            "hidden_size": _read_int(reader, "clip.vision.embedding_length"),
            "intermediate_size": _read_int(
                reader,
                "clip.vision.feed_forward_length",
            ),
            "num_heads": _read_int(reader, "clip.vision.attention.head_count"),
            "patch_size": patch_size,
            "spatial_merge_size": _read_int(
                reader,
                "clip.vision.spatial_merge_size",
            ),
            "temporal_patch_size": _read_int(
                reader,
                "clip.vision.temporal_patch_size",
            ),
            "out_hidden_size": _read_int(reader, "clip.vision.projection_dim"),
            "num_position_embeddings": num_position_embeddings,
        }
    )
    return vision_config or None


def _build_gemma4_vision_config(
    reader: gguf.GGUFReader | None,
) -> dict[str, Any] | None:
    if reader is None:
        return None

    hidden_size = _read_int(reader, "clip.vision.embedding_length")
    num_heads = _read_int(reader, "clip.vision.attention.head_count")
    head_dim = None
    if hidden_size and num_heads:
        head_dim = hidden_size // num_heads

    # GGUF mmproj metadata does not currently carry this Gemma4 HF field, but
    # the Transformers config class preserves it and vLLM requires it.
    default_output_length = _read_int_or_warn_default(
        reader,
        "clip.vision.default_output_length",
        _GEMMA4_DEFAULT_OUTPUT_LENGTH,
        "vision_config.default_output_length",
    )
    pooling_kernel_size = _read_int_or_warn_default(
        reader,
        "clip.vision.pooling_kernel_size",
        _GEMMA4_POOLING_KERNEL_SIZE,
        "vision_config.pooling_kernel_size",
    )
    position_embedding_size = _read_int_or_warn_default(
        reader,
        "clip.vision.position_embedding_size",
        _GEMMA4_POSITION_EMBEDDING_SIZE,
        "vision_config.position_embedding_size",
    )

    vision_config = _non_none(
        {
            "hidden_size": hidden_size,
            "intermediate_size": _read_int(
                reader,
                "clip.vision.feed_forward_length",
            ),
            "num_hidden_layers": _read_int(reader, "clip.vision.block_count"),
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_heads,
            "head_dim": head_dim,
            "patch_size": _read_int(reader, "clip.vision.patch_size"),
            "rms_norm_eps": _read_float(
                reader,
                "clip.vision.attention.layer_norm_epsilon",
            ),
            "pooling_kernel_size": pooling_kernel_size,
            "position_embedding_size": position_embedding_size,
            "standardize": True,
            "use_clipped_linears": False,
            "default_output_length": default_output_length,
        }
    )
    return vision_config or None


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


def _qwen35_ssm_value_heads(
    reader: gguf.GGUFReader,
    prefix: str,
) -> int | None:
    inner_size = _read_int(reader, f"{prefix}.ssm.inner_size")
    state_size = _read_int(reader, f"{prefix}.ssm.state_size")
    if inner_size is None or state_size in (None, 0):
        return None
    if inner_size % state_size != 0:
        logger.warning_once(
            "Ignoring %s.ssm.inner_size=%s because it is not divisible by "
            "%s.ssm.state_size=%s",
            prefix,
            inner_size,
            prefix,
            state_size,
        )
        return None
    return inner_size // state_size


def _build_qwen35moe_config(
    reader: gguf.GGUFReader,
    model_path: Path,
    has_lm_head: bool,
) -> PretrainedConfig:
    prefix = "qwen35moe"
    block_count = _read_int(reader, f"{prefix}.block_count")
    nextn_layers = _read_int(reader, f"{prefix}.nextn_predict_layers") or 0
    num_layers = None
    if block_count is not None:
        num_layers = max(block_count - nextn_layers, 0)

    head_dim = _read_int(reader, f"{prefix}.attention.key_length")
    full_attention_interval = _read_int(reader, f"{prefix}.full_attention_interval")
    tie_word_embeddings = not has_lm_head
    text_config = _non_none(
        {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": _token_count(reader),
            "hidden_size": _read_int(reader, f"{prefix}.embedding_length"),
            "intermediate_size": _read_int(reader, f"{prefix}.feed_forward_length"),
            "num_hidden_layers": num_layers,
            "mtp_num_hidden_layers": nextn_layers or None,
            "num_nextn_predict_layers": nextn_layers or None,
            "num_attention_heads": _read_int(reader, f"{prefix}.attention.head_count"),
            "num_key_value_heads": _read_int(
                reader, f"{prefix}.attention.head_count_kv"
            ),
            "head_dim": head_dim,
            "max_position_embeddings": _read_int(reader, f"{prefix}.context_length"),
            "rms_norm_eps": _read_float(
                reader, f"{prefix}.attention.layer_norm_rms_epsilon"
            ),
            "num_experts": _read_int(reader, f"{prefix}.expert_count"),
            "num_experts_per_tok": _read_int(reader, f"{prefix}.expert_used_count"),
            "moe_intermediate_size": _read_int(
                reader, f"{prefix}.expert_feed_forward_length"
            ),
            "shared_expert_intermediate_size": _read_int(
                reader, f"{prefix}.expert_shared_feed_forward_length"
            ),
            "layer_types": _qwen35_layer_types(
                full_attention_interval,
                num_layers=num_layers,
            ),
            "tie_word_embeddings": tie_word_embeddings,
            **_qwen35_text_config_updates_from_gguf(reader, prefix),
            **_token_id_fields(reader),
        }
    )

    mmproj_reader = _read_mmproj_reader(model_path)
    is_multimodal = mmproj_reader is not None
    architecture = (
        "Qwen3_5MoeForConditionalGeneration"
        if is_multimodal
        else "Qwen3_5MoeForCausalLM"
    )
    return Qwen3_5MoeConfig(
        text_config=text_config,
        vision_config=_build_qwen35_vision_config(mmproj_reader),
        architectures=[architecture],
        tie_word_embeddings=tie_word_embeddings,
        vocab_size=text_config.get("vocab_size"),
    )


def _build_qwen35_config(
    reader: gguf.GGUFReader,
    model_path: Path,
    has_lm_head: bool,
) -> PretrainedConfig:
    prefix = "qwen35"
    block_count = _read_int(reader, f"{prefix}.block_count")
    nextn_layers = _read_int(reader, f"{prefix}.nextn_predict_layers") or 0
    num_layers = None
    if block_count is not None:
        num_layers = max(block_count - nextn_layers, 0)
    head_dim = _read_int(reader, f"{prefix}.attention.key_length")
    full_attention_interval = _read_int(reader, f"{prefix}.full_attention_interval")
    tie_word_embeddings = not has_lm_head
    text_config = _non_none(
        {
            "model_type": "qwen3_5_text",
            "vocab_size": _token_count(reader),
            "hidden_size": _read_int(reader, f"{prefix}.embedding_length"),
            "intermediate_size": _read_int(reader, f"{prefix}.feed_forward_length"),
            "num_hidden_layers": num_layers,
            "mtp_num_hidden_layers": nextn_layers or None,
            "num_nextn_predict_layers": nextn_layers or None,
            "num_attention_heads": _read_int(reader, f"{prefix}.attention.head_count"),
            "num_key_value_heads": _read_int(
                reader, f"{prefix}.attention.head_count_kv"
            ),
            "head_dim": head_dim,
            "max_position_embeddings": _read_int(reader, f"{prefix}.context_length"),
            "rms_norm_eps": _read_float(
                reader, f"{prefix}.attention.layer_norm_rms_epsilon"
            ),
            "linear_conv_kernel_dim": _read_int(reader, f"{prefix}.ssm.conv_kernel"),
            "linear_key_head_dim": _read_int(reader, f"{prefix}.ssm.state_size"),
            "linear_value_head_dim": _read_int(reader, f"{prefix}.ssm.state_size"),
            "linear_num_key_heads": _read_int(reader, f"{prefix}.ssm.group_count"),
            "linear_num_value_heads": _qwen35_ssm_value_heads(reader, prefix),
            "layer_types": _qwen35_layer_types(
                full_attention_interval,
                num_layers=num_layers,
            ),
            "tie_word_embeddings": tie_word_embeddings,
            **_qwen35_text_config_updates_from_gguf(reader, prefix),
            **_token_id_fields(reader),
        }
    )

    mmproj_reader = _read_mmproj_reader(model_path)
    is_multimodal = mmproj_reader is not None
    architecture = (
        "Qwen3_5ForConditionalGeneration"
        if is_multimodal
        else "Qwen3_5ForCausalLM"
    )
    return Qwen3_5Config(
        text_config=text_config,
        vision_config=_build_qwen35_vision_config(mmproj_reader),
        architectures=[architecture],
        tie_word_embeddings=tie_word_embeddings,
        vocab_size=text_config.get("vocab_size"),
    )


def _build_gemma4_text_config(
    reader: gguf.GGUFReader,
    prefix: str,
    has_lm_head: bool,
) -> dict[str, Any]:
    block_count = _read_int(reader, f"{prefix}.block_count")
    sliding_pattern = _read_list(reader, f"{prefix}.attention.sliding_window_pattern")
    head_count_kv = _gguf_reader_value(reader, f"{prefix}.attention.head_count_kv")
    rope_theta = _read_float(reader, f"{prefix}.rope.freq_base")
    rope_theta_swa = _read_float(reader, f"{prefix}.rope.freq_base_swa")
    layer_types = _sliding_layer_types(
        sliding_pattern,
        num_layers=block_count,
    )

    return _non_none(
        {
            "model_type": "gemma4_text",
            "vocab_size": _token_count(reader),
            "hidden_size": _read_int(reader, f"{prefix}.embedding_length"),
            "hidden_size_per_layer_input": _read_int(
                reader,
                f"{prefix}.embedding_length_per_layer_input",
            ),
            "intermediate_size": _read_int(reader, f"{prefix}.feed_forward_length"),
            "num_hidden_layers": block_count,
            "num_attention_heads": _read_int(reader, f"{prefix}.attention.head_count"),
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
            "sliding_window": _read_int(reader, f"{prefix}.attention.sliding_window"),
            "layer_types": layer_types,
            "attention_k_eq_v": True,
            "attention_bias": False,
            "num_kv_shared_layers": _read_int(
                reader, f"{prefix}.attention.shared_kv_layers"
            ),
            "enable_moe_block": _read_int(reader, f"{prefix}.expert_count") is not None,
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
            "tie_word_embeddings": not has_lm_head,
            **_token_id_fields(reader),
        }
    )


def _build_gemma4_config(
    reader: gguf.GGUFReader,
    model_path: Path,
    has_lm_head: bool,
) -> PretrainedConfig:
    text_config = _build_gemma4_text_config(reader, "gemma4", has_lm_head)
    mmproj_reader = _read_mmproj_reader(model_path)
    is_multimodal = mmproj_reader is not None
    architecture = (
        "Gemma4ForConditionalGeneration" if is_multimodal else "Gemma4ForCausalLM"
    )
    return AutoConfig.for_model(
        "gemma4",
        text_config=text_config,
        vision_config=_build_gemma4_vision_config(mmproj_reader),
        architectures=[architecture],
        tie_word_embeddings=text_config.get("tie_word_embeddings"),
        vocab_size=text_config.get("vocab_size"),
    )


def _build_gemma4_assistant_config(
    reader: gguf.GGUFReader,
    has_lm_head: bool,
) -> PretrainedConfig:
    prefix = "gemma4-assistant"
    text_config = _build_gemma4_text_config(reader, prefix, has_lm_head)
    text_config.pop("model_type", None)
    nextn_layers = _read_int(reader, f"{prefix}.nextn_predict_layers")
    n_predict = _read_int_or_warn_default(
        reader,
        f"{prefix}.nextn_predict_count",
        _GEMMA4_ASSISTANT_N_PREDICT,
        "n_predict",
    )
    text_config.update(
        _non_none(
            {
                "backbone_hidden_size": _read_int(
                    reader, f"{prefix}.embedding_length_out"
                ),
                "num_nextn_predict_layers": nextn_layers,
                "mtp_num_hidden_layers": nextn_layers,
                "n_predict": n_predict,
            }
        )
    )
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
    has_lm_head: bool,
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
            "num_attention_heads": _read_int(reader, f"{prefix}.attention.head_count"),
            "num_key_value_heads": _read_int(
                reader, f"{prefix}.attention.head_count_kv"
            ),
            "head_dim": _read_int(reader, f"{prefix}.attention.key_length"),
            "max_position_embeddings": _read_int(reader, f"{prefix}.context_length"),
            "rms_norm_eps": _read_float(
                reader, f"{prefix}.attention.layer_norm_rms_epsilon"
            ),
            "rope_theta": _read_float(reader, f"{prefix}.rope.freq_base"),
            "tie_word_embeddings": not has_lm_head,
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
    has_lm_head = _has_lm_head(reader)
    try:
        if architecture == "qwen35":
            return _build_qwen35_config(reader, model_path, has_lm_head)
        if architecture == "qwen35moe":
            return _build_qwen35moe_config(reader, model_path, has_lm_head)
        if architecture == "gemma4":
            return _build_gemma4_config(reader, model_path, has_lm_head)
        if architecture == "gemma4-assistant":
            return _build_gemma4_assistant_config(reader, has_lm_head)
        if architecture is not None:
            return _build_simple_causal_config(reader, architecture, has_lm_head)
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
