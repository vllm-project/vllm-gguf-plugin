#!/usr/bin/env bash
# Build a release wheel with CUDA arch coverage for the local nvcc,
# including llama.cpp's architecture-specific Blackwell FP4 targets.
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
# llama.cpp's Blackwell NVFP4 kernels use architecture-specific block-scale MMA;
# plain 12.0 targets sm_120, while 12.0a/12.1a target the required SMs.
# 12.0a and 12.1a binaries are not interchangeable, so include each target.
if [ "$cuda_major" -ge 13 ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0a"
elif [ "$cuda_major" -ge 12 ] && [ "$cuda_minor" -ge 8 ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;10.0"
else
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.7;8.9;9.0"
fi

echo "CUDA $cuda_release; TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

exec uv build --wheel --no-build-isolation "$@"
