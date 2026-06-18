# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import re
from collections.abc import Generator
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

import gguf
import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from vllm.logger import init_logger

logger = init_logger(__name__)


_SPLIT_GGUF_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")
_HF_REPO_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PROCESSOR_SIDECAR_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "image_processor_config.json",
)


def _split_gguf_match(path: str | Path) -> re.Match[str] | None:
    return _SPLIT_GGUF_RE.search(str(path))


def expand_split_gguf_filenames(filename: str) -> list[str]:
    match = _split_gguf_match(filename)
    if match is None:
        return [filename]

    total = int(match.group(2))
    shard_digits = len(match.group(1))
    total_digits = len(match.group(2))
    prefix = filename[: match.start(1)]
    suffix = filename[match.end(2) :]
    return [
        f"{prefix}{idx:0{shard_digits}d}-of-{total:0{total_digits}d}{suffix}"
        for idx in range(1, total + 1)
    ]


def first_split_gguf_filename(filename: str | Path) -> str:
    """Return the first filename in a split GGUF set, or filename itself."""
    return expand_split_gguf_filenames(str(filename))[0]


def resolve_gguf_file_set(model_path: str | Path) -> list[str]:
    """Return the ordered GGUF files needed to load ``model_path``.

    Split GGUF files use names such as ``model-00001-of-00003.gguf``.  vLLM's
    loader should never silently continue with a partial split set, because
    that produces missing weights later in model load with much worse errors.
    """
    model_path = str(model_path)
    files = expand_split_gguf_filenames(model_path)
    if len(files) == 1:
        return files

    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise ValueError(
            "Incomplete split GGUF model: expected "
            f"{len(files)} shards for {model_path}, missing {len(missing)}: "
            + ", ".join(missing)
        )
    logger.info("Resolved split GGUF model to %d shard files", len(files))
    return files


def split_remote_gguf_file_ref(model_ref: str) -> tuple[str, str] | None:
    """Split ``namespace/repo/path.gguf`` into HF repo ID and filename."""
    parts = model_ref.split("/", 2)
    if len(parts) != 3 or not parts[2].endswith(".gguf"):
        return None
    if not _HF_REPO_PART_RE.fullmatch(parts[0]):
        return None
    if not _HF_REPO_PART_RE.fullmatch(parts[1]):
        return None
    if any(segment in ("", ".", "..") for segment in parts[2].split("/")):
        return None
    return f"{parts[0]}/{parts[1]}", parts[2]


def _download_candidate_sort_key(path: str) -> tuple[bool, int, str, int, str]:
    match = _split_gguf_match(path)
    if match is None:
        return (False, path.count("-"), path, 0, path)
    normalized = _SPLIT_GGUF_RE.sub(".gguf", path)
    return (True, normalized.count("-"), normalized, int(match.group(1)), path)


def _mmproj_quant_tokens(quant_type: str) -> tuple[str, ...]:
    quant_type = quant_type.upper()
    tokens = [quant_type]
    for separator in ("-", "."):
        if separator in quant_type:
            tokens.append(quant_type.rsplit(separator, 1)[1])
    return tuple(dict.fromkeys(tokens))


def _mmproj_candidate_sort_key(filename: str, quant_type: str) -> tuple[int, str]:
    basename = Path(filename).name
    name = basename.upper()
    if basename.lower() == "mmproj.gguf":
        priority = 0
    elif any(token in name for token in _mmproj_quant_tokens(quant_type)):
        priority = 1
    elif "BF16" not in name and "F16" in name:
        priority = 2
    elif "BF16" in name:
        priority = 3
    elif "F32" in name:
        priority = 4
    else:
        priority = 5
    return priority, filename


def _list_remote_sidecar_files(repo_id: str, revision: str | None) -> list[str]:
    try:
        return list_repo_files(repo_id, revision=revision)
    except Exception as e:
        logger.debug("Failed to inspect GGUF repo sidecars for %s: %s", repo_id, e)
        return []


def _select_mmproj_filename_from_files(
    files: list[str],
    quant_type: str,
    search_dirs: list[str] | None = None,
) -> str | None:
    search_dir_distances = (
        {directory: idx for idx, directory in enumerate(search_dirs)}
        if search_dirs is not None
        else None
    )
    mmproj_files: list[tuple[str, int]] = []
    for filename in files:
        if not (
            filename.lower().endswith(".gguf")
            and "mmproj" in Path(filename).name.lower()
        ):
            continue
        distance = 0
        if search_dir_distances is not None:
            parent = PurePosixPath(filename).parent
            directory = "" if parent == PurePosixPath(".") else parent.as_posix()
            if directory not in search_dir_distances:
                continue
            distance = search_dir_distances[directory]
        mmproj_files.append((filename, distance))

    if not mmproj_files:
        return None
    filename, _distance = sorted(
        mmproj_files,
        key=lambda item: (
            _mmproj_candidate_sort_key(item[0], quant_type)[0],
            item[1],
            Path(item[0]).name,
            item[0],
        ),
    )[0]
    return filename


def _select_remote_mmproj_for_gguf_file(
    repo_id: str,
    filename: str,
    revision: str | None,
    files: list[str] | None = None,
) -> str | None:
    if files is None:
        files = _list_remote_sidecar_files(repo_id, revision)
    quant_hint = Path(_SPLIT_GGUF_RE.sub(".gguf", filename)).stem
    return _select_mmproj_filename_from_files(
        files,
        quant_hint,
        search_dirs=_remote_sidecar_search_dirs(filename),
    )


def _remote_processor_sidecar_patterns() -> list[str]:
    return [
        pattern
        for filename in _PROCESSOR_SIDECAR_FILES
        for pattern in (filename, f"*/{filename}")
    ]


def _remote_sidecar_search_dirs(filename: str) -> list[str]:
    path = PurePosixPath(filename)
    dirs = []
    directory = path.parent
    while True:
        dirs.append(directory)
        if directory == PurePosixPath("."):
            break
        parent = directory.parent
        if parent == directory:
            break
        directory = parent

    search_dirs: list[str] = []
    seen = set()
    for directory in dirs:
        directory_str = "" if directory == PurePosixPath(".") else directory.as_posix()
        if directory_str in seen:
            continue
        seen.add(directory_str)
        search_dirs.append(directory_str)
    return search_dirs


def _select_remote_processor_sidecars(
    files: list[str],
    filename: str,
) -> list[str]:
    available = set(files)
    sidecars: list[str] = []
    for directory in _remote_sidecar_search_dirs(filename):
        for sidecar_filename in _PROCESSOR_SIDECAR_FILES:
            sidecar = (
                f"{directory}/{sidecar_filename}" if directory else sidecar_filename
            )
            if sidecar in available:
                sidecars.append(sidecar)
    return sidecars


def remote_gguf_quant_allow_patterns(quant_type: str) -> list[str]:
    prefix_list = ["*.", "*-"]
    suffix_list = ["-*", ""]
    base_patterns = [
        f"{prefix}{qt}{suffix}.gguf"
        for qt in (quant_type.upper(), quant_type.lower())
        for prefix, suffix in itertools.product(prefix_list, suffix_list)
    ]
    return list(
        itertools.chain.from_iterable(
            (pattern, f"*/{pattern}") for pattern in base_patterns
        )
    )


def _matches_gguf_quant_filename(filename: str, quant_type: str) -> bool:
    return any(
        fnmatch(filename, pattern)
        for pattern in remote_gguf_quant_allow_patterns(quant_type)
    )


def _select_remote_gguf_filename(
    files: list[str],
    quant_type: str,
) -> str | None:
    gguf_files = [
        filename
        for filename in files
        if filename.lower().endswith(".gguf")
        and "mmproj" not in Path(filename).name.lower()
        and _matches_gguf_quant_filename(filename, quant_type)
    ]
    if not gguf_files:
        return None
    return sorted(set(gguf_files), key=_download_candidate_sort_key)[0]


def select_remote_gguf_filename(
    files: list[str],
    quant_type: str,
) -> str | None:
    """Select the main GGUF model file for ``repo_id:quant`` candidates."""
    return _select_remote_gguf_filename(files, quant_type)


def _resolve_downloaded_gguf_from_patterns(
    folder: str,
    allow_patterns: list[str],
    quant_type: str,
) -> str:
    del allow_patterns
    folder_path = Path(folder)
    local_files = [
        path.relative_to(folder_path).as_posix()
        for path in folder_path.rglob("*.gguf")
        if path.is_file()
    ]
    selected_filename = _select_remote_gguf_filename(local_files, quant_type)

    if selected_filename is None:
        raise ValueError(
            f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}"
        )

    return resolve_gguf_file_set(folder_path / selected_filename)[0]


def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    sidecar_files = _list_remote_sidecar_files(repo_id, revision)
    selected_filename = _select_remote_gguf_filename(sidecar_files, quant_type)
    if selected_filename is None:
        allow_patterns = remote_gguf_quant_allow_patterns(quant_type)
        if mmproj_filename := _select_mmproj_filename_from_files(
            sidecar_files,
            quant_type,
        ):
            allow_patterns.append(mmproj_filename)
            allow_patterns.extend(_remote_processor_sidecar_patterns())

        folder = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            revision=revision,
            ignore_patterns=ignore_patterns,
        )
        return _resolve_downloaded_gguf_from_patterns(
            folder,
            allow_patterns,
            quant_type,
        )

    allow_patterns = expand_split_gguf_filenames(selected_filename)
    mmproj_filename = _select_remote_mmproj_for_gguf_file(
        repo_id,
        selected_filename,
        revision,
        files=sidecar_files,
    )
    if mmproj_filename is not None:
        allow_patterns.append(mmproj_filename)
        allow_patterns.extend(
            _select_remote_processor_sidecars(sidecar_files, selected_filename)
        )

    folder = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )
    return resolve_gguf_file_set(os.path.join(folder, selected_filename))[0]


def download_gguf_file(
    repo_id: str,
    filename: str,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> str:
    """Download an exact GGUF file reference, including split shard sets."""
    filenames = expand_split_gguf_filenames(filename)
    sidecar_files = _list_remote_sidecar_files(repo_id, revision)
    mmproj_filename = _select_remote_mmproj_for_gguf_file(
        repo_id,
        filename,
        revision,
        files=sidecar_files,
    )
    processor_sidecars = _select_remote_processor_sidecars(
        sidecar_files,
        filename,
    )
    if mmproj_filename is None:
        processor_sidecars = []

    sidecar_filenames = []
    if mmproj_filename is not None:
        sidecar_filenames.append(mmproj_filename)
    sidecar_filenames.extend(processor_sidecars)
    if len(filenames) == 1:
        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
        )
        for sidecar_filename in sidecar_filenames:
            hf_hub_download(
                repo_id=repo_id,
                filename=sidecar_filename,
                cache_dir=cache_dir,
                revision=revision,
            )
        return local_file

    allow_patterns = list(filenames)
    allow_patterns.extend(sidecar_filenames)
    folder = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
    )
    return resolve_gguf_file_set(os.path.join(folder, filename))[0]


def resolve_local_gguf(local_dir: str, quant_type: str) -> str:
    """Find a GGUF file matching *quant_type* in a local directory."""
    local_path = Path(local_dir)
    matches = [
        path
        for path in local_path.rglob("*.gguf")
        if path.is_file()
        and "mmproj" not in path.name.lower()
        and _matches_gguf_quant_filename(
            path.relative_to(local_path).as_posix(),
            quant_type,
        )
    ]
    if not matches:
        raise ValueError(
            f"No GGUF file matching quant_type '{quant_type}' found in {local_dir}"
        )
    matches.sort(
        key=lambda path: _download_candidate_sort_key(
            path.relative_to(local_path).as_posix()
        )
    )
    return resolve_gguf_file_set(matches[0])[0]


def get_gguf_extra_tensor_names_multi(
    gguf_files: list[str | Path],
    gguf_to_hf_name_map: dict[str, str],
) -> list[str]:
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = {
        tensor.name
        for gguf_file in gguf_files
        for tensor in gguf.GGUFReader(gguf_file).tensors
    }
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [
        hf_name
        for gguf_key, hf_name in gguf_to_hf_name_map.items()
        if gguf_key in extra_keys
    ]


def get_gguf_extra_tensor_names(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> list[str]:
    return get_gguf_extra_tensor_names_multi([gguf_file], gguf_to_hf_name_map)


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
