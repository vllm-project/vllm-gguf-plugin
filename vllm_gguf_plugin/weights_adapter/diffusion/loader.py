# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF loader for diffusion models.

Provides the complete GGUF weight-loading path (resolve, iterate, load, HF
fallback) so that the calling framework (e.g. vllm-omni) only needs thin glue.
Zero dependency on vllm-omni.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from ...weight_utils import download_gguf
from . import get_diffusion_gguf_adapter


@dataclass
class DiffusionWeightSource:
    """Minimal description of a diffusion-model weight source."""

    prefix: str
    subfolder: str | None = None


def is_gguf_quant_config(quant_config: object) -> bool:
    """Return True if *quant_config* describes GGUF quantization.

    Uses duck-typing: works with both ``DiffusionGGUFConfig`` objects and
    plain dicts (``{"method": "gguf", ...}``).
    """
    if hasattr(quant_config, "get_name") and quant_config.get_name() == "gguf":
        return True
    return isinstance(quant_config, dict) and quant_config.get("method") == "gguf"


def get_gguf_model_from_config(quant_config: object) -> str | None:
    """Extract the ``gguf_model`` path from *quant_config*."""
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        return quant_config.get("gguf_model")
    return getattr(quant_config, "gguf_model", None)


def resolve_gguf_model_path(
    gguf_model: str,
    revision: str | None = None,
    download_dir: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Resolve a GGUF model reference to a local file path.

    Accepts three formats:
      1. Local file path (``/path/to/model.gguf``)
      2. HuggingFace file (``repo_id/filename.gguf``)
      3. HuggingFace quant-type selector (``repo_id:Q4_K_M``)
    """
    if os.path.isfile(gguf_model):
        return gguf_model
    # repo_id/filename.gguf
    if "/" in gguf_model and gguf_model.endswith(".gguf"):
        repo_id, filename = gguf_model.rsplit("/", 1)
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=download_dir,
        )
    # repo_id:quant_type
    if "/" in gguf_model and ":" in gguf_model:
        repo_id, quant_type = gguf_model.rsplit(":", 1)
        return download_gguf(
            repo_id,
            quant_type,
            cache_dir=download_dir,
            revision=revision,
            ignore_patterns=ignore_patterns,
        )
    raise ValueError(
        f"Unrecognized GGUF reference: {gguf_model!r} (expected local file, "
        "<repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
    )


def _is_transformer_source(source: DiffusionWeightSource) -> bool:
    if source.subfolder == "transformer":
        return True
    return source.prefix.startswith("transformer.")


def _get_loadable_names(model: nn.Module) -> set[str]:
    """Collect loadable names without using ``state_dict()``.

    ``UninitializedParameter`` (used by GGUF) raises during ``detach()``,
    so we collect names directly from ``named_parameters`` / ``named_buffers``.
    """
    return {name for name, _ in model.named_parameters()} | {
        name for name, _ in model.named_buffers()
    }


def load_diffusion_gguf_weights(
    gguf_model: str,
    model: nn.Module,
    model_class_name: str | None,
    model_type: str | None,
    sources: list[DiffusionWeightSource],
    hf_weights_fn: Callable[
        [DiffusionWeightSource], Iterable[tuple[str, torch.Tensor]]
    ],
    revision: str | None = None,
    download_dir: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> set[str]:
    """Load diffusion-model weights from a GGUF file with HF fallback.

    For each source:
      - **Transformer sources**: load from GGUF first. If some weights are
        still missing (partial quantization), fall back to HF safetensors
        for only those missing weights.
      - **Non-transformer sources** (text encoder, VAE, etc.): load
        entirely from HF.

    Args:
        gguf_model: GGUF model reference (local path, ``repo/file.gguf``, or
            ``repo:quant_type``).
        model: The ``nn.Module`` to load weights into.
        model_class_name: Model class name (e.g. ``"QwenImagePipeline"``).
        model_type: Model type string from config (e.g. ``"qwen_image"``).
        sources: Weight sources to load.
        hf_weights_fn: Callback that returns an HF weight iterator for a
            given source (used for non-transformer components and GGUF fallback).
        revision: Optional HuggingFace revision.
        download_dir: Optional download cache directory.
        ignore_patterns: Optional patterns to ignore during download.

    Returns:
        Set of loaded weight names.
    """
    gguf_file = resolve_gguf_model_path(
        gguf_model, revision, download_dir, ignore_patterns
    )
    adapter = get_diffusion_gguf_adapter(gguf_file, model_class_name, model_type)
    loaded: set[str] = set()
    loadable_names: set[str] | None = None

    for source in sources:
        if _is_transformer_source(source):
            # Load transformer from GGUF first
            gguf_iter = (
                (source.prefix + name, tensor)
                for name, tensor in adapter.weights_iterator()
            )
            loaded |= model.load_weights(gguf_iter)

            # GGUF checkpoints can be transformer-only or partially quantized.
            # Only fall back to HF if this source still has missing loadable weights.
            loadable_names = loadable_names or _get_loadable_names(model)
            has_missing = any(
                name.startswith(source.prefix) and name not in loaded
                for name in loadable_names
            )
            if not has_missing:
                continue

            hf_iter = (
                (name, tensor)
                for name, tensor in hf_weights_fn(source)
                if name in loadable_names and name not in loaded
            )
            loaded |= model.load_weights(hf_iter)
        else:
            # Non-transformer components always load from HF
            loaded |= model.load_weights(hf_weights_fn(source))

    return loaded
