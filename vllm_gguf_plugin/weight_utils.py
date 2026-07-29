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
        local_files.extend(glob.glob(os.path.join(folder, pattern)))

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
) -> None:
    """Fetch the multimodal projector into the backbone's snapshot dir.

    ``download_gguf`` only matches the requested quant type, so the mmproj
    file of a multimodal repo is never pulled by it. Repos ship the projector
    in several precisions; take one, both to save the download and to keep
    ``detect_gguf_multimodal`` deterministic. Downloading through the hub
    cache keeps concurrent TP workers serialized on the cache lock.
    """
    try:
        mmproj_files = list_filtered_repo_files(
            repo_id, allow_patterns=["*mmproj*.gguf"], revision=revision
        )
        if not mmproj_files:
            return
        filename = sorted(mmproj_files)[0]
        logger.info("Downloading mmproj file %s from %s", filename, repo_id)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
        )
    except Exception as e:
        logger.warning("Failed to download mmproj from %s: %s", repo_id, e)


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
    if not matches:
        raise ValueError(
            f"No GGUF file matching quant_type '{quant_type}' found in {local_dir}"
        )
    matches.sort(key=lambda x: (x.count("-"), x))
    return matches[0]


def get_gguf_shard_files(model_path: str) -> list[str]:
    """All shards of a split GGUF, or just *model_path* when it is whole."""
    match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
    if not match:
        return [model_path]
    total = int(match.group(2))
    num_digits = len(match.group(1))
    prefix = model_path[: match.start(1)]
    suffix = model_path[match.end(2) :]
    files = []
    for i in range(1, total + 1):
        shard_path = f"{prefix}{i:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
        if os.path.isfile(shard_path):
            files.append(shard_path)
    if files:
        logger.info("Discovered %d GGUF shard files", len(files))
    return files if files else [model_path]


def get_gguf_extra_tensor_names(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> list[str]:
    reader = gguf.GGUFReader(gguf_file)
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = {tensor.name for tensor in reader.tensors}
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


def get_gguf_tensor_names(gguf_files: list[str]) -> set[str]:
    """Names of every tensor in *gguf_files*, regardless of any name map."""
    return {
        tensor.name
        for gguf_file in gguf_files
        for tensor in gguf.GGUFReader(gguf_file).tensors
    }


def get_gguf_weight_type_map(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> dict[str, str]:
    reader = gguf.GGUFReader(gguf_file)
    return {
        gguf_to_hf_name_map[tensor.name]: tensor.tensor_type.name
        for tensor in reader.tensors
        if tensor.name in gguf_to_hf_name_map
    }


def gguf_quant_weights_iterator(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str] | None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    yield from gguf_quant_weights_iterator_multi([gguf_file], gguf_to_hf_name_map)


def gguf_quant_weights_iterator_multi(
    gguf_files: list[str],
    gguf_to_hf_name_map: dict[str, str] | None = None,
    dequant_suffixes: tuple[str, ...] = ("embed_tokens.weight", "lm_head.weight"),
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
            # These load as plain weights; emitting GGUF quant params here
            # would leave them zeroed.
            if weight_type.name not in _QUANT_TYPES and name.endswith(dequant_suffixes):
                deq = gguf.quants.dequantize(tensor.data, weight_type)
                yield name, torch.tensor(deq)
                continue
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
    """Split a stacked GGUF expert tensor into the per-expert weights vLLM
    loads, leaving every other weight untouched."""
    for name, weight in weights:
        if weight.ndim == 3 and ".experts.0." in name:
            for expert_id, expert_weight in enumerate(weight.unbind()):
                expert_name = name.replace(".experts.0.", f".experts.{expert_id}.")
                yield expert_name, expert_weight
        else:
            yield name, weight


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
    # for gguf_file in gguf_files:
    #     reader = gguf.GGUFReader(gguf_file)
    #     for tensor in reader.tensors:
    #         if tensor.tensor_type.name in unquant_types:
    #             yield tensor.name.rsplit(".", 1)[0]
