# vllm-gguf-plugin

Out-of-tree GGUF quantization plugin for vLLM.

This package hooks into vLLM's `vllm.general_plugins` entry point group and
re-registers the `gguf` quantization method, model loader, and model-format
handler with a fully out-of-tree GGUF implementation.

## Install

```bash
uv pip install -e . --torch-backend=auto
```

## Development

```bash
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

## How it works

- vLLM loads `vllm.general_plugins` during engine setup.
- This package registers `gguf` again, but points it at the plugin-local
  GGUF quantization config, model loader, and model-format handler.
- GGUF loading, kernels, and quantized layer behavior come from this plugin
  instead of the main vLLM repository.
