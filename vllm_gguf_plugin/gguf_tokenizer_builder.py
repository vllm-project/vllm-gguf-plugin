# SPDX-License-Identifier: Apache-2.0

"""Build a HF-compatible tokenizer directory from GGUF embedded metadata."""

import hashlib
import os
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any

import gguf
from transformers import PreTrainedTokenizerFast
from transformers.integrations.ggml import (
    GGUF_TOKENIZER_MAPPING,
    convert_gguf_tokenizer,
)
from vllm.logger import init_logger

from .gguf_utils import _gguf_reader_value, _gguf_scalar_value, check_gguf_file

logger = init_logger(__name__)

_TOKENIZER_CACHE_ENV = "VLLM_GGUF_TOKENIZER_CACHE"
_DEFAULT_TOKENIZER_CACHE = "~/.cache/vllm-gguf-plugin/tokenizers"

_TOKENIZER_ARCH_ALIASES = {
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


def _cache_key(model_path: Path) -> str:
    stat = model_path.stat()
    raw_key = f"{model_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]


def _extract_tokenizer_dict(reader: gguf.GGUFReader) -> dict[str, Any]:
    tokenizer_dict: dict[str, Any] = {}
    field_mapping = GGUF_TOKENIZER_MAPPING["tokenizer"]
    for gguf_suffix, hf_name in field_mapping.items():
        value = _gguf_reader_value(reader, f"tokenizer.{gguf_suffix}")
        if value is not None:
            tokenizer_dict[hf_name] = _decode_sequence(value)
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


def build_tokenizer_from_gguf(model: str | PathLike) -> str | None:
    """Materialize a tokenizer directory from GGUF embedded metadata.

    Returns the cache directory path on success. Returns ``None`` if the GGUF
    tokenizer is unsupported or incomplete, so callers can use existing
    tokenizer fallback behavior.
    """
    model_path = Path(model)
    if not check_gguf_file(model_path):
        return None

    cache_dir = _tokenizer_cache_root() / _cache_key(model_path)
    if (cache_dir / "tokenizer.json").is_file():
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
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            **additional_kwargs,
            **_special_token_kwargs(tokenizer_dict),
        )
        if chat_template := tokenizer_config.get("chat_template"):
            tokenizer.chat_template = chat_template
    except Exception as e:
        logger.debug("Failed to build tokenizer from GGUF %s: %s", model_path, e)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(cache_dir)
    return str(cache_dir)
