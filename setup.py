# SPDX-License-Identifier: Apache-2.0

import sys

from setuptools import setup


def _should_build_extension() -> bool:
    packaging_commands = {"sdist", "egg_info", "dist_info"}
    return not any(command in packaging_commands for command in sys.argv[1:])


setup_kwargs = {}

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
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )

setup(**setup_kwargs)
