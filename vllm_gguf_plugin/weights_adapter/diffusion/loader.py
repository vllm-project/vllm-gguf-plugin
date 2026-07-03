# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF loader for diffusion models.

Provides the complete GGUF weight-loading path (resolve, iterate, load) so
that the calling framework (e.g. vllm-omni) only needs thin glue. Zero
dependency on vllm-omni.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from gguf import dequantize
from huggingface_hub import hf_hub_download
from torch import nn

from ... import ops
from ...quantization.utils import UNQUANTIZED_TYPES
from ...weight_utils import download_gguf, resolve_local_gguf


def get_diffusion_gguf_adapter(*args, **kwargs):
    from . import get_diffusion_gguf_adapter as _get_adapter

    return _get_adapter(*args, **kwargs)


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
      3. Quant-type selector (``repo_id:Q4_K_M`` or ``/local/dir:Q4_K_M``)
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
    # repo_id:quant_type or local_dir:quant_type
    if "/" in gguf_model and ":" in gguf_model:
        repo_id, quant_type = gguf_model.rsplit(":", 1)
        if os.path.isdir(repo_id):
            return resolve_local_gguf(repo_id, quant_type)
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


def _is_gguf_source(source: DiffusionWeightSource, adapter: object) -> bool:
    source_prefix = getattr(adapter, "source_prefix", "transformer.")
    source_subfolder = getattr(adapter, "source_subfolder", "transformer")
    return source.prefix == source_prefix or source.subfolder == source_subfolder


def _get_loadable_names(model: nn.Module) -> set[str]:
    """Collect loadable names without using ``state_dict()``.

    ``UninitializedParameter`` (used by GGUF) raises during ``detach()``,
    so we collect names directly from ``named_parameters`` / ``named_buffers``.
    """
    return {name for name, _ in model.named_parameters()} | {
        name for name, _ in model.named_buffers()
    }


def _dense_weight_from_gguf_qweight(
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    qtype = WeightType(qweight_type)
    if qtype in UNQUANTIZED_TYPES:
        return qweight

    if not qweight.is_cuda:
        weight = dequantize(qweight.detach().cpu().numpy(), qtype)
        return torch.from_numpy(weight).to(dtype=torch.float32)

    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    return ops.ggml_dequantize(qweight, int(qtype), *shape, torch.float32)


def _gguf_weights_for_loadable_names(
    weights: Iterable[tuple[str, torch.Tensor]],
    loadable_names: set[str],
) -> Iterable[tuple[str, torch.Tensor]]:
    qweight_types: dict[str, int] = {}

    for name, tensor in weights:
        if name.endswith(".qweight_type"):
            base_name = name[: -len(".qweight_type")]
            qweight_types[base_name] = int(tensor.item())
            if f"{base_name}.weight" not in loadable_names:
                yield name, tensor
            continue

        if not name.endswith(".qweight"):
            yield name, tensor
            continue

        base_name = name[: -len(".qweight")]
        weight_name = f"{base_name}.weight"
        if weight_name not in loadable_names:
            yield name, tensor
            continue

        if base_name not in qweight_types:
            raise ValueError(f"Missing GGUF qweight_type for {name}")

        yield (
            weight_name,
            _dense_weight_from_gguf_qweight(tensor, qweight_types[base_name]),
        )


def _hf_weights_for_loadable_names(
    weights: Iterable[tuple[str, torch.Tensor]],
    loadable_names: set[str],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if name in loadable_names:
            yield name, tensor


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
    """Load diffusion-model weights from a GGUF file.

    For each source:
      - The adapter-selected GGUF source loads exclusively from GGUF.
        GGUF qweights may be restored to dense tensors when the host module
        intentionally keeps that parameter unquantized.
      - All other sources (text encoder, VAE, etc.) load exclusively from HF.

    Args:
        gguf_model: GGUF model reference (local path, ``repo/file.gguf``, or
            ``repo:quant_type``).
        model: The ``nn.Module`` to load weights into.
        model_class_name: Model class name (e.g. ``"QwenImagePipeline"``).
        model_type: Model type string from config (e.g. ``"qwen_image"``).
        sources: Weight sources to load.
        hf_weights_fn: Callback that returns an HF weight iterator for a
            given non-transformer source.
        revision: Optional HuggingFace revision.
        download_dir: Optional download cache directory.
        ignore_patterns: Optional patterns to ignore during download.

    Returns:
        Set of loaded weight names.
    """
    gguf_file = resolve_gguf_model_path(
        gguf_model=gguf_model,
        revision=revision,
        download_dir=download_dir,
        ignore_patterns=ignore_patterns,
    )
    adapter = get_diffusion_gguf_adapter(gguf_file, model_class_name, model_type)
    loaded: set[str] = set()
    loadable_names: set[str] | None = None

    for source in sources:
        if _is_gguf_source(source, adapter):
            loadable_names = loadable_names or _get_loadable_names(model)
            gguf_iter = (
                (source.prefix + name, tensor)
                for name, tensor in adapter.weights_iterator()
            )
            loaded |= model.load_weights(
                _gguf_weights_for_loadable_names(gguf_iter, loadable_names)
            )
        else:
            # Non-transformer components always load from HF.
            loadable_names = loadable_names or _get_loadable_names(model)
            loaded |= model.load_weights(
                _hf_weights_for_loadable_names(
                    (
                        (name, tensor)
                        for name, tensor in hf_weights_fn(source)
                        if not source.prefix or name.startswith(source.prefix)
                    ),
                    loadable_names,
                )
            )

    return loaded
