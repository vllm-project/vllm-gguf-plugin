# SPDX-License-Identifier: Apache-2.0

import glob
import itertools
import os
import re
from collections.abc import Generator, Iterable
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from vllm.logger import init_logger
from vllm.transformers_utils.repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    prefix_list = ["*.", "*-"]
    suffix_list = ["-*", ""]
    allow_patterns = [
        f"{prefix}{qt}{suffix}.gguf"
        for qt in (quant_type.upper(), quant_type.lower())
        for prefix, suffix in itertools.product(prefix_list, suffix_list)
    ]

    folder = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )

    local_files: list[str] = []
    for pattern in allow_patterns:
        # backbone shards may live in per-quant subdirectories
        # (e.g. unsloth/Kimi-K3-GGUF "UD-IQ1_M/*.gguf")
        local_files.extend(
            glob.glob(os.path.join(folder, "**", pattern), recursive=True)
        )

    if not local_files:
        raise ValueError(
            f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}"
        )

    local_files.sort(key=lambda x: (x.count("-"), x))
    return local_files[0]


def download_mmproj(
    repo_id: str,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> str | None:
    """Download one multimodal projector from a GGUF repository."""
    candidates = list_filtered_repo_files(
        repo_id,
        allow_patterns=["*mmproj*.gguf"],
        revision=revision,
    )
    if not candidates:
        return None

    precision_order = ("BF16", "F16", "F32")

    def candidate_key(filename: str) -> tuple[int, str]:
        upper_name = filename.upper()
        precision = next(
            (
                index
                for index, value in enumerate(precision_order)
                if value in upper_name
            ),
            len(precision_order),
        )
        return precision, filename

    filename = min(candidates, key=candidate_key)
    logger.info("Downloading multimodal projector %s from %s", filename, repo_id)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        revision=revision,
    )


def resolve_local_gguf(local_dir: str, quant_type: str) -> str:
    """Find a GGUF file matching *quant_type* in a local directory."""
    import glob as glob_mod

    patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
    ]
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob_mod.glob(os.path.join(local_dir, pat)))
    # A sibling mmproj projector shares the F16/F32-style suffix but is not a
    # model backbone.
    matches = [m for m in matches if "mmproj" not in os.path.basename(m).lower()]
    if not matches:
        raise ValueError(
            f"No GGUF file matching quant_type '{quant_type}' found in {local_dir}"
        )
    matches.sort(key=lambda x: (x.count("-"), x))
    return matches[0]


def get_gguf_shard_files(model_path: str) -> list[str]:
    """Return every shard belonging to *model_path*, or the path itself."""
    match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
    if not match:
        return [model_path]

    total = int(match.group(2))
    num_digits = len(match.group(1))
    prefix = model_path[: match.start(1)]
    suffix = model_path[match.end(2) :]
    files = [
        f"{prefix}{index:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
        for index in range(1, total + 1)
    ]
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} of {total} GGUF shard files: {missing}"
        )
    logger.info("Discovered %d GGUF shard files", len(files))
    return files


def get_gguf_tensor_names(gguf_files: Iterable[str]) -> set[str]:
    """Return raw tensor names across all supplied GGUF files."""
    return {
        tensor.name
        for gguf_file in gguf_files
        for tensor in gguf.GGUFReader(gguf_file).tensors
    }


def gguf_quant_weights_iterator(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str] | None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    yield from gguf_quant_weights_iterator_multi([gguf_file], gguf_to_hf_name_map)


def gguf_quant_weights_iterator_multi(
    gguf_files: list[str],
    gguf_to_hf_name_map: dict[str, str] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield ``(name, tensor)`` for all tensors in *gguf_files*.

    When *gguf_to_hf_name_map* is ``None``, raw GGUF tensor names are used
    directly (useful when a caller will apply a :class:`WeightsMapper`
    afterwards).  When a mapping is provided, tensors not present in the map
    are skipped and names are translated accordingly.
    """
    _QUANT_TYPES = ("F32", "BF16", "F16")

    for gguf_file in gguf_files:
        reader = gguf.GGUFReader(gguf_file)
        for tensor in reader.tensors:
            if gguf_to_hf_name_map is not None:
                if tensor.name not in gguf_to_hf_name_map:
                    continue
                name = gguf_to_hf_name_map[tensor.name]
            else:
                name = tensor.name

            weight_type = tensor.tensor_type
            if weight_type.name not in _QUANT_TYPES:
                yield name.replace("weight", "qweight_type"), torch.tensor(weight_type)
                name = name.replace("weight", "qweight")

            weight = tensor.data
            if weight_type.name == "BF16" and weight.dtype == np.uint8:
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    weight = weight.byteswap()
                param = torch.tensor(weight).view(torch.bfloat16)
            else:
                param = torch.tensor(weight)
            yield name, param


def split_stacked_experts(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Split stacked GGUF expert tensors into vLLM per-expert weights."""
    for name, weight in weights:
        if weight.ndim == 3 and ".experts.0." in name:
            for expert_id, expert_weight in enumerate(weight.unbind()):
                expert_name = name.replace(".experts.0.", f".experts.{expert_id}.")
                yield expert_name, expert_weight
        else:
            yield name, weight


def dequantize_gguf_tensor(
    weight: torch.Tensor, weight_type: int | None
) -> torch.Tensor:
    """Return a float tensor for a GGUF payload of any storage type.

    Quantized payloads go through the plugin's dequant kernel (CUDA, with a
    Triton fallback); F32/F16/BF16 payloads are just cast. Runs on GPU — GGUF
    inference requires CUDA anyway.
    """
    if weight_type is None or weight_type in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    ):
        return weight.float()
    from . import ops

    weight_2d = weight.reshape(-1, weight.shape[-1]).contiguous().cuda()
    block_size, type_size = gguf.GGML_QUANT_SIZES[
        gguf.GGMLQuantizationType(weight_type)
    ]
    n = weight_2d.shape[1] // type_size * block_size
    dequantized = ops.ggml_dequantize(
        weight_2d, weight_type, weight_2d.shape[0], n, torch.float32
    )
    return dequantized.reshape(*weight.shape[:-1], n)


def get_gguf_unquantized_params(gguf_files: list[str]) -> list[str]:
    _QUANT_TYPES = ("F32", "BF16", "F16")
    return list(
        {
            tensor.name
            for gguf_file in gguf_files
            for tensor in gguf.GGUFReader(gguf_file).tensors
            if tensor.tensor_type.name in _QUANT_TYPES
        }
    )
