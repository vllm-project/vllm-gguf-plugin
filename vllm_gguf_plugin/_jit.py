# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import platform
import re
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from threading import Lock

import torch
from torch.utils import cpp_extension

_GGUF_LIBRARY_NAMESPACE = "_C_gguf"
_JIT_EXTENSION_NAME = "_C_gguf"
_PRECOMPILED_ARTIFACTS_PACKAGE = "vllm_gguf_plugin_precompiled"
_BUILD_LOCK = Lock()


def _gguf_ops_available() -> bool:
    return hasattr(torch.ops, _GGUF_LIBRARY_NAMESPACE) and hasattr(
        torch.ops._C_gguf, "ggml_dequantize"
    )


def _csrc_root() -> Path:
    return Path(__file__).resolve().parent / "csrc"


def _extension_sources() -> list[str]:
    root = _csrc_root()
    return [
        str(root / "torch_bindings.cpp"),
        str(root / "gguf" / "gguf_kernel.cu"),
    ]


def _include_paths() -> list[str]:
    root = _csrc_root()
    return [str(root), str(root / "gguf")]


def _normalize_tag_component(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "-", value).strip("-")


def _runtime_platform_tag() -> str:
    return "-".join(
        [
            _normalize_tag_component(platform.system().lower()),
            _normalize_tag_component(platform.machine().lower()),
        ]
    )


def get_gguf_precompiled_artifact_tag() -> str:
    torch_version = _normalize_tag_component(torch.__version__.split("+", 1)[0])
    cuda_version = _normalize_tag_component(torch.version.cuda or "cpu")
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return (
        f"{_runtime_platform_tag()}/torch-{torch_version}/"
        f"cuda-{cuda_version}/python-{python_tag}"
    )


def _package_root(package_name: str) -> Path | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _precompiled_search_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("VLLM_GGUF_PLUGIN_PRECOMPILED_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path(__file__).resolve().parent / "precompiled")
    package_root = _package_root(_PRECOMPILED_ARTIFACTS_PACKAGE)
    if package_root is not None:
        roots.append(package_root / "artifacts")
    return roots


def _precompiled_glob_patterns() -> set[str]:
    return {f"{_JIT_EXTENSION_NAME}*{suffix}" for suffix in EXTENSION_SUFFIXES}


def _precompiled_gguf_library_paths() -> list[Path]:
    explicit_library = os.environ.get("VLLM_GGUF_PLUGIN_PRECOMPILED_LIB")
    if explicit_library:
        return [Path(explicit_library)]

    matches: list[Path] = []
    tag = get_gguf_precompiled_artifact_tag()
    for root in _precompiled_search_roots():
        candidate_dir = root / tag
        if not candidate_dir.is_dir():
            continue
        for pattern in sorted(_precompiled_glob_patterns()):
            matches.extend(sorted(candidate_dir.glob(pattern)))
    return matches


def _load_precompiled_gguf_library() -> bool:
    for library_path in _precompiled_gguf_library_paths():
        torch.ops.load_library(str(library_path))
        if _gguf_ops_available():
            return True
    return False


def ensure_gguf_cuda_ops_loaded() -> None:
    if _gguf_ops_available():
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "vllm-gguf-plugin CUDA kernels require an available CUDA device."
        )
    if torch.version.cuda is None:
        raise RuntimeError(
            "vllm-gguf-plugin CUDA kernels require a CUDA-enabled PyTorch build."
        )

    with _BUILD_LOCK:
        if _gguf_ops_available():
            return

        if _load_precompiled_gguf_library():
            return

        if cpp_extension.CUDA_HOME is None:
            raise RuntimeError(
                "vllm-gguf-plugin could not find the CUDA toolkit. Set CUDA_HOME "
                "before using GGUF CUDA ops or install a matching precompiled "
                "artifact."
            )

        build_directory = os.environ.get("VLLM_GGUF_PLUGIN_JIT_BUILD_DIR")
        if build_directory:
            Path(build_directory).mkdir(parents=True, exist_ok=True)

        cpp_extension.load(
            name=_JIT_EXTENSION_NAME,
            sources=_extension_sources(),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-DUSE_CUDA"],
            extra_include_paths=_include_paths(),
            build_directory=build_directory,
            verbose=os.environ.get("VLLM_GGUF_PLUGIN_JIT_VERBOSE") == "1",
            with_cuda=True,
        )
