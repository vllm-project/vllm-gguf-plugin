# vLLM GGUF Quantization Plugin

This plugin provides out-of-tree GGUF quantization support for vLLM after
in-tree support deprecation
([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend [uv](https://docs.astral.sh/uv/) for package management. If you
don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-gguf-plugin
   cd vllm-gguf-plugin
   ```

2. Install the plugin in development mode:

   ```bash
   uv pip install -e . --torch-backend=auto
   ```

Or install directly:

```bash
uv pip install . --torch-backend=auto
```

## Development

```bash
uv pip install -e .[dev] --torch-backend=auto
pre-commit install
pre-commit run --all-files
```

The same hooks also run in GitHub Actions on every push and pull request.

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```

## Smoke Tests

Check registration behavior without loading a model:

```bash
python scripts/plugin_gguf_smoke.py --json
```

Force the plugin to take ownership from an in-tree GGUF vLLM build:

```bash
python scripts/plugin_gguf_smoke.py --override-in-tree --expect active --json
```

Exercise the real vLLM GGUF load path with a local model:

```bash
python scripts/plugin_gguf_smoke.py \
  --override-in-tree \
  --expect active \
  --generate \
  --model /path/to/model.gguf \
  --tokenizer /path/to/tokenizer \
  --hf-config-path /path/to/hf-config \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.25 \
  --enforce-eager \
  --json
```

For a large GPT-OSS MXFP4 GGUF smoke on a single GPU, keep the request shape
small and select the native vLLM MoE backend explicitly:

```bash
python scripts/plugin_gguf_smoke.py \
  --override-in-tree \
  --expect active \
  --generate \
  --model /path/to/gpt-oss-20b-mxfp4.gguf \
  --tokenizer openai/gpt-oss-20b \
  --dtype bfloat16 \
  --max-model-len 64 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.20 \
  --moe-backend marlin \
  --max-tokens 4 \
  --disable-v1-multiprocessing \
  --enforce-eager \
  --json
```
