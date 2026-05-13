#!/usr/bin/env bash
# Build the optional jit-cache wheel with CUDA arch coverage that mirrors
# scripts/build_release_wheel.sh for the current nvcc.
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
    echo "error: nvcc not found on PATH" >&2
    exit 1
fi

cuda_release=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
cuda_major=${cuda_release%.*}
cuda_minor=${cuda_release#*.}

if [ "$cuda_major" -ge 13 ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0"
elif [ "$cuda_major" -ge 12 ] && [ "$cuda_minor" -ge 8 ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;12.0;12.1"
else
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.7;8.9;9.0"
fi

echo "CUDA $cuda_release; TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

exec uv build ./vllm-gguf-plugin-jit-cache --wheel --no-build-isolation "$@"
