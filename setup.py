# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
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
