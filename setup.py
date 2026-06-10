# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

import tomllib
from setuptools import setup


def _package_version() -> str:
    project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    version = project["tool"]["vllm_gguf_plugin"]["base_version"]
    suffix = os.environ.get("VLLM_GGUF_PLUGIN_LOCAL_VERSION_SUFFIX")
    if not suffix:
        return version
    normalized_suffix = suffix if suffix.startswith("+") else f"+{suffix}"
    return f"{version}{normalized_suffix}"


def _should_build_extension() -> bool:
    packaging_commands = {"sdist", "egg_info", "dist_info"}
    return not any(command in packaging_commands for command in sys.argv[1:])


setup_kwargs: dict = {"version": _package_version()}

if _should_build_extension():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    setup_kwargs.update(
        ext_modules=[
            CUDAExtension(
                name="vllm_gguf_plugin._C_gguf",
                sources=[
                    "vllm_gguf_plugin/csrc/torch_bindings.cpp",
                    "vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu",
                ],
                include_dirs=[
                    "vllm_gguf_plugin/csrc",
                    "vllm_gguf_plugin/csrc/gguf",
                ],
                py_limited_api=True,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "--use_fast_math",
                        # Exposes aoti_torch_get_current_cuda_stream in the AOTI shim.
                        "-DUSE_CUDA",
                    ],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp310"}},
    )

setup(**setup_kwargs)
