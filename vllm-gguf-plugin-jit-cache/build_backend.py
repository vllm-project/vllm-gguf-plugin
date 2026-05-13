# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
import shutil
from pathlib import Path

from setuptools import build_meta as _orig
from wheel.bdist_wheel import bdist_wheel


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SUBPROJECT_ROOT = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SUBPROJECT_ROOT / "vllm_gguf_plugin_precompiled"
_BUILD_META_PATH = _PACKAGE_ROOT / "_build_meta.py"


def _load_jit_module():
    spec = importlib.util.spec_from_file_location(
        "vllm_gguf_plugin_jit",
        _REPO_ROOT / "vllm_gguf_plugin" / "_jit.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load JIT helper module for cache wheel build.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_main_package_version() -> str:
    pyproject_text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject_text, re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not determine main package version from pyproject.toml.")
    return match.group(1)


def _write_build_meta() -> str:
    version = _read_main_package_version()
    _BUILD_META_PATH.write_text(
        '"""Build metadata for vllm-gguf-plugin-jit-cache."""\n'
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    return version


def _artifact_output_dir() -> Path:
    jit_module = _load_jit_module()
    return _PACKAGE_ROOT / "artifacts" / jit_module.get_gguf_precompiled_artifact_tag()


def _build_precompiled_artifact() -> None:
    jit_module = _load_jit_module()
    if jit_module.torch.version.cuda is None:
        raise RuntimeError("A CUDA-enabled PyTorch build is required to build the JIT cache wheel.")
    if jit_module.cpp_extension.CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit not found. Set CUDA_HOME before building the JIT cache wheel.")

    artifacts_root = _PACKAGE_ROOT / "artifacts"
    if artifacts_root.exists():
        shutil.rmtree(artifacts_root)

    artifact_output_dir = _artifact_output_dir()
    artifact_output_dir.mkdir(parents=True, exist_ok=True)

    build_directory = _REPO_ROOT / "build" / "jit-cache-wheel"
    build_directory.mkdir(parents=True, exist_ok=True)

    module = jit_module.cpp_extension.load(
        name=jit_module._JIT_EXTENSION_NAME,
        sources=jit_module._extension_sources(),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-DUSE_CUDA"],
        extra_include_paths=jit_module._include_paths(),
        build_directory=str(build_directory),
        verbose=False,
        with_cuda=True,
    )

    library_path = Path(module.__file__)
    if not library_path.is_file():
        raise RuntimeError(f"Expected compiled extension at {library_path}.")

    shutil.copy2(library_path, artifact_output_dir / library_path.name)


def _prepare_build() -> None:
    _write_build_meta()
    _build_precompiled_artifact()


class PlatformSpecificBdistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


class _MonkeyPatchBdistWheel:
    def __enter__(self):
        from setuptools.command import bdist_wheel as setuptools_bdist_wheel

        self.original_bdist_wheel = setuptools_bdist_wheel.bdist_wheel
        setuptools_bdist_wheel.bdist_wheel = PlatformSpecificBdistWheel
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from setuptools.command import bdist_wheel as setuptools_bdist_wheel

        setuptools_bdist_wheel.bdist_wheel = self.original_bdist_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_build()
    with _MonkeyPatchBdistWheel():
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_build()
    orig_build_editable = getattr(_orig, "build_editable", None)
    if orig_build_editable is None:
        raise RuntimeError("build_editable not supported by setuptools backend")
    return orig_build_editable(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _write_build_meta()
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _write_build_meta()
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )


get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = getattr(
    _orig, "get_requires_for_build_editable", None
)
