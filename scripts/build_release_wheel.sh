#!/usr/bin/env bash
# Build a release wheel with CUDA arch coverage that mirrors
# vllm-project/vllm CMakeLists.txt CUDA_SUPPORTED_ARCHS for the local nvcc.
# Forwards extra args to `uv build` (e.g. -o /tmp/dist).
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
    echo "error: nvcc not found on PATH" >&2
    exit 1
fi

cuda_release=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
cuda_major=${cuda_release%.*}
cuda_minor=${cuda_release#*.}

# 10.1 is dropped from the 12.8 list: PyTorch's TORCH_CUDA_ARCH_LIST validator
# (torch.utils.cpp_extension._get_cuda_arch_flags) does not list it.
if [ "$cuda_major" -ge 13 ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0"
elif [ "$cuda_major" -ge 12 ] && [ "$cuda_minor" -ge 8 ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;12.0;12.1"
else
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.7;8.9;9.0"
fi

echo "CUDA $cuda_release; TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

exec uv build --wheel --no-build-isolation "$@"
