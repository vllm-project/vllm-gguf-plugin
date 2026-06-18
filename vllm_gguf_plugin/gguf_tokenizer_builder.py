# SPDX-License-Identifier: Apache-2.0

"""Build a HF-compatible tokenizer directory from GGUF embedded metadata."""

import hashlib
import json
import os
from contextlib import suppress
from os import PathLike
from pathlib import Path
from shutil import copyfile
from typing import Any

import gguf
from transformers import PreTrainedTokenizerFast
from transformers.integrations.ggml import (
    GGUF_TOKENIZER_MAPPING,
    convert_gguf_tokenizer,
)
from vllm.logger import init_logger

from .gguf_utils import (
    _gguf_reader_value,
    _gguf_scalar_value,
    check_gguf_file,
    detect_gguf_multimodal,
    gguf_sidecar_search_dirs,
)

logger = init_logger(__name__)

_TOKENIZER_CACHE_ENV = "VLLM_GGUF_TOKENIZER_CACHE"
_DEFAULT_TOKENIZER_CACHE = "~/.cache/vllm-gguf-plugin/tokenizers"
_PROCESSOR_SIDECAR_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "image_processor_config.json",
)
_PROCESSOR_DEFAULT_SOURCE = (
    "non-GGUF processor defaults mirrored from Unsloth Qwen3.6/Gemma4 "
    "GGUF sidecars observed on 2026-06-11; GGUF metadata values take "
    "precedence when present"
)

_TOKENIZER_ARCH_ALIASES = {
    "qwen35": "qwen3",
    "qwen3_5": "qwen3",
    "qwen35moe": "qwen3_moe",
    "qwen3_5_moe": "qwen3_moe",
    "gemma4": "gemma3_text",
    "gemma4-assistant": "gemma3_text",
    "gemma4_assistant": "gemma3_text",
}


def _decode_value(value: Any) -> Any:
    value = _gguf_scalar_value(value)
    if isinstance(value, bytes):
        with suppress(UnicodeDecodeError):
            return value.decode("utf-8")
    return value


def _decode_sequence(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes)):
        return _decode_value(value)
    with suppress(AttributeError):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return _decode_value(value)


def _gguf_architecture(reader: gguf.GGUFReader) -> str | None:
    value = _decode_value(_gguf_reader_value(reader, "general.architecture"))
    return value if isinstance(value, str) else None


def _tokenizer_cache_root() -> Path:
    return Path(
        os.environ.get(_TOKENIZER_CACHE_ENV, _DEFAULT_TOKENIZER_CACHE)
    ).expanduser()


def _cache_key(model_path: Path) -> str | None:
    try:
        stat = model_path.stat()
    except OSError as e:
        logger.debug("Failed to stat GGUF tokenizer source %s: %s", model_path, e)
        return None
    raw_key = f"{model_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]


def _extract_tokenizer_dict(reader: gguf.GGUFReader) -> dict[str, Any]:
    tokenizer_dict: dict[str, Any] = {}
    field_mapping = GGUF_TOKENIZER_MAPPING["tokenizer"]
    for gguf_suffix, hf_name in field_mapping.items():
        value = _gguf_reader_value(reader, f"tokenizer.{gguf_suffix}")
        if value is not None:
            tokenizer_dict[hf_name] = _decode_sequence(value)
    token_type_value = _gguf_reader_value(reader, "tokenizer.ggml.token_type")
    if token_type_value is not None:
        tokenizer_dict["token_type"] = _decode_sequence(token_type_value)
    return tokenizer_dict


def _extract_tokenizer_config(reader: gguf.GGUFReader) -> dict[str, Any]:
    tokenizer_config: dict[str, Any] = {}
    field_mapping = GGUF_TOKENIZER_MAPPING["tokenizer_config"]
    for gguf_suffix, hf_name in field_mapping.items():
        value = _gguf_reader_value(reader, f"tokenizer.{gguf_suffix}")
        if value is not None:
            tokenizer_config[hf_name] = _decode_sequence(value)
    return tokenizer_config


def _token_by_id(tokens: list[Any], token_id: Any) -> str | None:
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        if not token_id:
            return None
        token_id = token_id[0]
    with suppress(TypeError, ValueError, IndexError):
        token = tokens[int(token_id)]
        if isinstance(token, str):
            return token
    return None


def _special_token_kwargs(tokenizer_dict: dict[str, Any]) -> dict[str, str]:
    tokens = tokenizer_dict.get("tokens")
    if not isinstance(tokens, list):
        return {}
    special_ids = {
        "bos_token": tokenizer_dict.get("bos_token_id"),
        "eos_token": tokenizer_dict.get("eos_token_id"),
        "pad_token": tokenizer_dict.get("pad_token_id"),
        "unk_token": tokenizer_dict.get("unk_token_id"),
    }
    return {
        key: token
        for key, token_id in special_ids.items()
        if (token := _token_by_id(tokens, token_id)) is not None
    }


def _local_config_path(model_path: Path) -> Path | None:
    """Return the nearest local HF config next to a GGUF file, if present."""
    for candidate in (model_path.parent, model_path.parent.parent):
        config_path = candidate / "config.json"
        if config_path.is_file():
            return config_path
    return None


def _read_special_token_ids_from_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}

    special_ids: dict[str, Any] = {}
    for token_name, config_key in {
        "bos_token": "bos_token_id",
        "eos_token": "eos_token_id",
        "pad_token": "pad_token_id",
        "unk_token": "unk_token_id",
    }.items():
        token_id = config.get(config_key)
        if token_id is None:
            token_id = text_config.get(config_key)
        if token_id is not None:
            special_ids[token_name] = token_id
    return special_ids


def _local_config_special_token_kwargs(
    model_path: Path,
    tokenizer_dict: dict[str, Any],
) -> dict[str, str]:
    tokens = tokenizer_dict.get("tokens")
    if not isinstance(tokens, list):
        return {}

    config_path = _local_config_path(model_path)
    if config_path is None:
        return {}

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Failed to read local GGUF config %s: %s", config_path, e)
        return {}

    return {
        token_name: token
        for token_name, token_id in _read_special_token_ids_from_config(
            config,
        ).items()
        if (token := _token_by_id(tokens, token_id)) is not None
    }


_GEMMA4_MODEL_SPECIFIC_TOKENS = {
    "audio_token": "<|audio|>",
    "boa_token": "<|audio>",
    "boi_token": "<|image>",
    "eoa_token": "<audio|>",
    "eoc_token": "<channel|>",
    "eoi_token": "<image|>",
    "eot_token": "<turn|>",
    "escape_token": '<|"|>',
    "etc_token": "<tool_call|>",
    "etd_token": "<tool|>",
    "etr_token": "<tool_response|>",
    "image_token": "<|image|>",
    "soc_token": "<|channel>",
    "sot_token": "<|turn>",
    "stc_token": "<|tool_call>",
    "std_token": "<|tool>",
    "str_token": "<|tool_response>",
    "think_token": "<|think|>",
    "video_token": "<|video|>",
}

_QWEN_MM_SPECIAL_TOKENS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def _tokens_present(
    tokenizer_dict: dict[str, Any],
    candidates: tuple[str, ...],
) -> list[str]:
    tokens = tokenizer_dict.get("tokens")
    if not isinstance(tokens, list):
        return []
    token_set = {token for token in tokens if isinstance(token, str)}
    return [token for token in candidates if token in token_set]


def _gemma4_model_specific_special_tokens(
    tokenizer_dict: dict[str, Any],
) -> dict[str, str]:
    return {
        name: token
        for name, token in _GEMMA4_MODEL_SPECIFIC_TOKENS.items()
        if token in _tokens_present(tokenizer_dict, (token,))
    }


def _append_additional_special_tokens(
    tokenizer_config: dict[str, Any],
    tokens: list[str],
) -> bool:
    if not tokens:
        return False

    existing = tokenizer_config.get("additional_special_tokens")
    if existing is None:
        additional_tokens: list[Any] = []
    elif isinstance(existing, list):
        additional_tokens = list(existing)
    else:
        additional_tokens = [existing]

    existing_tokens: set[str] = set()
    for item in additional_tokens:
        if isinstance(item, str):
            existing_tokens.add(item)
        elif isinstance(item, dict) and isinstance(item.get("content"), str):
            existing_tokens.add(item["content"])

    changed = False
    for token in tokens:
        if token in existing_tokens:
            continue
        additional_tokens.append(token)
        existing_tokens.add(token)
        changed = True

    if changed:
        tokenizer_config["additional_special_tokens"] = additional_tokens
    return changed


def _remove_additional_special_tokens(
    tokenizer_config: dict[str, Any],
    tokens: set[str],
) -> bool:
    if not tokens:
        return False

    existing = tokenizer_config.get("additional_special_tokens")
    if existing is None:
        return False
    additional_tokens = list(existing) if isinstance(existing, list) else [existing]

    filtered_tokens: list[Any] = []
    changed = False
    for item in additional_tokens:
        token = None
        if isinstance(item, str):
            token = item
        elif isinstance(item, dict) and isinstance(item.get("content"), str):
            token = item["content"]
        if token in tokens:
            changed = True
            continue
        filtered_tokens.append(item)

    if changed:
        tokenizer_config["additional_special_tokens"] = filtered_tokens
    return changed


_GGUF_SPECIAL_TOKEN_TYPES = {
    int(gguf.TokenType.CONTROL),
    int(gguf.TokenType.USER_DEFINED),
}


def _gguf_special_control_tokens(
    tokenizer_dict: dict[str, Any],
    special_token_kwargs: dict[str, str] | None = None,
) -> list[str]:
    """Restore GGUF control/user-defined tokens as HF special tokens."""
    tokens = tokenizer_dict.get("tokens")
    token_types = tokenizer_dict.get("token_type")
    if not isinstance(tokens, list) or not isinstance(token_types, list):
        return []

    named_special_tokens = set(_special_token_kwargs(tokenizer_dict).values())
    if special_token_kwargs:
        named_special_tokens.update(special_token_kwargs.values())
    special_tokens: list[str] = []
    for token, token_type in zip(tokens, token_types, strict=False):
        if not isinstance(token, str) or token in named_special_tokens:
            continue
        with suppress(TypeError, ValueError):
            if int(token_type) in _GGUF_SPECIAL_TOKEN_TYPES:
                special_tokens.append(token)
    return special_tokens


def _patch_tokenizer_config_from_gguf(
    cache_dir: Path,
    architecture: str,
    tokenizer_dict: dict[str, Any],
    model_path: Path,
) -> None:
    tokenizer_config_path = cache_dir / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        return

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Failed to read tokenizer config %s: %s", tokenizer_config_path, e)
        return

    changed = False
    special_token_kwargs = _local_config_special_token_kwargs(
        model_path,
        tokenizer_dict,
    )
    for token_name, token in special_token_kwargs.items():
        if tokenizer_config.get(token_name) != token:
            tokenizer_config[token_name] = token
            changed = True
    changed |= _remove_additional_special_tokens(
        tokenizer_config,
        set(special_token_kwargs.values()),
    )

    changed |= _append_additional_special_tokens(
        tokenizer_config,
        _gguf_special_control_tokens(tokenizer_dict, special_token_kwargs),
    )

    if architecture in {"gemma4", "gemma4-assistant", "gemma4_assistant"}:
        model_specific_tokens = _gemma4_model_specific_special_tokens(tokenizer_dict)
        if model_specific_tokens:
            tokenizer_config["processor_class"] = "Gemma4Processor"
            tokenizer_config["model_specific_special_tokens"] = model_specific_tokens
            for name, token in model_specific_tokens.items():
                tokenizer_config.setdefault(name, token)
            tokenizer_config["extra_special_tokens"] = {
                **(
                    tokenizer_config["extra_special_tokens"]
                    if isinstance(tokenizer_config.get("extra_special_tokens"), dict)
                    else {}
                ),
                **model_specific_tokens,
            }
            changed = True

    if architecture in {"qwen35", "qwen3_5", "qwen35moe", "qwen3_5_moe"}:
        mm_tokens = _tokens_present(tokenizer_dict, _QWEN_MM_SPECIAL_TOKENS)
        changed |= _append_additional_special_tokens(tokenizer_config, mm_tokens)
        if "<|image_pad|>" in mm_tokens:
            tokenizer_config.setdefault("image_token", "<|image_pad|>")
            changed = True
        if "<|video_pad|>" in mm_tokens:
            tokenizer_config.setdefault("video_token", "<|video_pad|>")
            changed = True

    if changed:
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, indent=2) + "\n",
            encoding="utf-8",
        )


def _read_int(reader: gguf.GGUFReader, key: str) -> int | None:
    value = _gguf_scalar_value(_gguf_reader_value(reader, key))
    if value is None:
        return None
    with suppress(TypeError, ValueError):
        return int(value)
    return None


def _copy_local_processor_sidecars(model_path: Path, cache_dir: Path) -> None:
    """Copy local sidecar files for the GGUF, without network fallback."""
    for filename in _PROCESSOR_SIDECAR_FILES:
        target = cache_dir / filename
        if target.is_file():
            continue
        for directory in gguf_sidecar_search_dirs(model_path):
            source = directory / filename
            if not source.is_file():
                continue
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                copyfile(source, target)
            except Exception as e:
                logger.debug("Failed to copy local GGUF sidecar %s: %s", source, e)
            break


def _write_json_if_missing(
    cache_dir: Path,
    filename: str,
    data: dict[str, Any],
) -> None:
    target = cache_dir / filename
    if target.is_file():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _read_mmproj_reader(model_path: Path) -> gguf.GGUFReader | None:
    mmproj_path = detect_gguf_multimodal(str(model_path))
    if mmproj_path is None:
        return None
    try:
        return gguf.GGUFReader(str(mmproj_path))
    except Exception as e:
        logger.debug("Failed to read GGUF mmproj sidecar %s: %s", mmproj_path, e)
        return None


def _qwen_image_processor_config(reader: gguf.GGUFReader) -> dict[str, Any]:
    # See _PROCESSOR_DEFAULT_SOURCE for the provenance of non-GGUF defaults.
    return {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_processor_type": "Qwen2VLImageProcessor",
        "image_std": [0.5, 0.5, 0.5],
        "merge_size": _read_int(reader, "clip.vision.spatial_merge_size") or 2,
        "patch_size": _read_int(reader, "clip.vision.patch_size") or 16,
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
        "size": {
            "longest_edge": 16777216,
            "shortest_edge": 65536,
        },
        "temporal_patch_size": (
            _read_int(reader, "clip.vision.temporal_patch_size") or 2
        ),
    }


def _qwen_video_processor_config(reader: gguf.GGUFReader) -> dict[str, Any]:
    # See _PROCESSOR_DEFAULT_SOURCE for the provenance of non-GGUF defaults.
    config = _qwen_image_processor_config(reader)
    config.update(
        {
            "do_sample_frames": True,
            "fps": 2,
            "max_frames": 768,
            "min_frames": 4,
            "return_metadata": False,
            "size": {
                "longest_edge": 25165824,
                "shortest_edge": 4096,
            },
            "video_processor_type": "Qwen3VLVideoProcessor",
        }
    )
    config.pop("image_processor_type", None)
    return config


def _gemma4_image_processor_config(reader: gguf.GGUFReader) -> dict[str, Any]:
    # See _PROCESSOR_DEFAULT_SOURCE for the provenance of non-GGUF defaults.
    patch_size = _read_int(reader, "clip.vision.patch_size") or 16
    image_seq_length = _read_int(reader, "clip.vision.default_output_length") or 280
    return {
        "do_convert_rgb": True,
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.0, 0.0, 0.0],
        "image_processor_type": "Gemma4ImageProcessor",
        "image_seq_length": image_seq_length,
        "image_std": [1.0, 1.0, 1.0],
        "max_soft_tokens": image_seq_length,
        "patch_size": patch_size,
        "pooling_kernel_size": 3,
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
    }


def _gemma4_video_processor_config(reader: gguf.GGUFReader) -> dict[str, Any]:
    # See _PROCESSOR_DEFAULT_SOURCE for the provenance of non-GGUF defaults.
    patch_size = _read_int(reader, "clip.vision.patch_size") or 16
    return {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "do_sample_frames": True,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "max_soft_tokens": 70,
        "num_frames": 32,
        "patch_size": patch_size,
        "pooling_kernel_size": 3,
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
        "return_metadata": False,
        "video_processor_type": "Gemma4VideoProcessor",
    }


def _gemma4_feature_extractor_config() -> dict[str, Any]:
    # See _PROCESSOR_DEFAULT_SOURCE for the provenance of non-GGUF defaults.
    return {
        "dither": 0.0,
        "feature_extractor_type": "Gemma4AudioFeatureExtractor",
        "feature_size": 128,
        "fft_length": 512,
        "fft_overdrive": False,
        "frame_length": 320,
        "hop_length": 160,
        "input_scale_factor": 1.0,
        "max_frequency": 8000.0,
        "mel_floor": 0.001,
        "min_frequency": 0.0,
        "padding_side": "right",
        "padding_value": 0.0,
        "per_bin_mean": None,
        "per_bin_stddev": None,
        "preemphasis": 0.0,
        "preemphasis_htk_flavor": True,
        "return_attention_mask": True,
        "sampling_rate": 16000,
    }


def _processor_sidecars_from_metadata(
    architecture: str,
    mmproj_reader: gguf.GGUFReader,
) -> dict[str, dict[str, Any]]:
    if architecture in {"qwen35", "qwen3_5", "qwen35moe", "qwen3_5_moe"}:
        image_config = _qwen_image_processor_config(mmproj_reader)
        video_config = _qwen_video_processor_config(mmproj_reader)
        return {
            "processor_config.json": {
                "image_processor": image_config,
                "processor_class": "Qwen3VLProcessor",
                "video_processor": video_config,
            },
            "preprocessor_config.json": image_config,
            "video_preprocessor_config.json": video_config,
        }

    if architecture == "gemma4":
        image_config = _gemma4_image_processor_config(mmproj_reader)
        video_config = _gemma4_video_processor_config(mmproj_reader)
        return {
            "processor_config.json": {
                "audio_ms_per_token": 40,
                "audio_seq_length": 750,
                "feature_extractor": _gemma4_feature_extractor_config(),
                "image_processor": image_config,
                "image_seq_length": image_config["image_seq_length"],
                "processor_class": "Gemma4Processor",
                "video_processor": video_config,
            },
            "preprocessor_config.json": image_config,
            "video_preprocessor_config.json": video_config,
        }

    return {}


def _materialize_processor_sidecars(
    model_path: Path,
    cache_dir: Path,
    architecture: str | None = None,
) -> None:
    """Materialize processor sidecars without implicit HF Hub access."""
    _copy_local_processor_sidecars(model_path, cache_dir)
    mmproj_reader = _read_mmproj_reader(model_path)
    if mmproj_reader is None:
        return

    if architecture is None:
        try:
            architecture = _gguf_architecture(gguf.GGUFReader(str(model_path)))
        except Exception as e:
            logger.debug("Failed to read GGUF architecture from %s: %s", model_path, e)
            return
    if architecture is None:
        return

    for filename, data in _processor_sidecars_from_metadata(
        architecture,
        mmproj_reader,
    ).items():
        _write_json_if_missing(cache_dir, filename, data)


def _patch_cached_tokenizer_from_gguf(
    model_path: Path,
    cache_dir: Path,
) -> str | None:
    try:
        reader = gguf.GGUFReader(str(model_path))
        architecture = _gguf_architecture(reader)
        if architecture is None:
            return None
        tokenizer_dict = _extract_tokenizer_dict(reader)
    except Exception as e:
        logger.debug(
            "Failed to read cached GGUF tokenizer metadata %s: %s", model_path, e
        )
        return None

    _patch_tokenizer_config_from_gguf(
        cache_dir,
        architecture,
        tokenizer_dict,
        model_path,
    )
    _materialize_processor_sidecars(model_path, cache_dir, architecture)
    return architecture


def build_tokenizer_from_gguf(model: str | PathLike) -> str | None:
    """Materialize a tokenizer directory from GGUF embedded metadata.

    Returns the cache directory path on success. Returns ``None`` if the GGUF
    tokenizer is unsupported or incomplete, so callers can use existing
    tokenizer fallback behavior.
    """
    model_path = Path(model)
    if not check_gguf_file(model_path):
        return None

    cache_key = _cache_key(model_path)
    if cache_key is None:
        return None
    cache_dir = _tokenizer_cache_root() / cache_key
    if (cache_dir / "tokenizer.json").is_file():
        _patch_cached_tokenizer_from_gguf(model_path, cache_dir)
        return str(cache_dir)

    try:
        reader = gguf.GGUFReader(str(model_path))
        architecture = _gguf_architecture(reader)
        if architecture is None:
            return None
        tokenizer_architecture = _TOKENIZER_ARCH_ALIASES.get(
            architecture,
            architecture,
        )
        tokenizer_dict = _extract_tokenizer_dict(reader)
        tokenizer_config = _extract_tokenizer_config(reader)
        backend_tokenizer, additional_kwargs = convert_gguf_tokenizer(
            tokenizer_architecture,
            tokenizer_dict,
        )
        special_token_kwargs = _special_token_kwargs(tokenizer_dict)
        special_token_kwargs.update(
            _local_config_special_token_kwargs(model_path, tokenizer_dict)
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            **additional_kwargs,
            **special_token_kwargs,
        )
        if chat_template := tokenizer_config.get("chat_template"):
            tokenizer.chat_template = chat_template
    except Exception as e:
        logger.debug("Failed to build tokenizer from GGUF %s: %s", model_path, e)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(cache_dir)
    _patch_tokenizer_config_from_gguf(
        cache_dir,
        architecture,
        tokenizer_dict,
        model_path,
    )
    _materialize_processor_sidecars(model_path, cache_dir, architecture)
    return str(cache_dir)
