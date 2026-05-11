# vLLM GGUF Quantization Plugin

This plugin provides out-of-tree GGUF quantization supports for vLLM after in-tree support deprecation ([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend to use [uv](https://docs.astral.sh/uv/) for package management. If you don't have it installed:

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

## Usage
```
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```
