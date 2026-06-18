# SPDX-License-Identifier: Apache-2.0

import glob
import itertools
import os
import re
from collections.abc import Generator
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from vllm.logger import init_logger

logger = init_logger(__name__)


_SPLIT_GGUF_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")


def _split_gguf_match(path: str | Path) -> re.Match[str] | None:
    return _SPLIT_GGUF_RE.search(str(path))


def resolve_gguf_file_set(model_path: str | Path) -> list[str]:
    """Return the ordered GGUF files needed to load ``model_path``.

    Split GGUF files use names such as ``model-00001-of-00003.gguf``.  vLLM's
    loader should never silently continue with a partial split set, because
    that produces missing weights later in model load with much worse errors.
    """
    model_path = str(model_path)
    match = _split_gguf_match(model_path)
    if match is None:
        return [model_path]

    total = int(match.group(2))
    shard_digits = len(match.group(1))
    total_digits = len(match.group(2))
    prefix = model_path[: match.start(1)]
    suffix = model_path[match.end(2) :]
    files = [
        f"{prefix}{idx:0{shard_digits}d}-of-{total:0{total_digits}d}{suffix}"
        for idx in range(1, total + 1)
    ]
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise ValueError(
            "Incomplete split GGUF model: expected "
            f"{total} shards for {model_path}, missing {len(missing)}: "
            + ", ".join(missing)
        )
    logger.info("Resolved split GGUF model to %d shard files", len(files))
    return files


def _download_candidate_sort_key(path: str) -> tuple[bool, int, str, int, str]:
    match = _split_gguf_match(path)
    if match is None:
        return (False, path.count("-"), path, 0, path)
    normalized = _SPLIT_GGUF_RE.sub(".gguf", path)
    return (True, normalized.count("-"), normalized, int(match.group(1)), path)


def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    prefix_list = ["*.", "*-"]
    suffix_list = ["-*", ""]
    base_patterns = [
        f"{prefix}{qt}{suffix}.gguf"
        for qt in (quant_type.upper(), quant_type.lower())
        for prefix, suffix in itertools.product(prefix_list, suffix_list)
    ]
    allow_patterns = list(
        itertools.chain.from_iterable(
            (pattern, f"*/{pattern}") for pattern in base_patterns
        )
    )

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

    local_files = sorted(set(local_files), key=_download_candidate_sort_key)
    return resolve_gguf_file_set(local_files[0])[0]


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


def get_gguf_extra_tensor_names(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> list[str]:
    reader = gguf.GGUFReader(gguf_file)
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = {tensor.name for tensor in reader.tensors}
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


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
    gguf_files: list[str], gguf_to_hf_name_map: dict[str, str] | None = None
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
