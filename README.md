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

2. If vLLM is not already installed, install it first:

   ```bash
   uv pip install vllm --torch-backend=auto
   ```

3. Build and install the plugin against the PyTorch installation used by
   vLLM:

   ```bash
   uv pip install -e . --no-build-isolation
   ```

   Disabling build isolation ensures that the CUDA extension is compiled
   against the same PyTorch installation used by vLLM at runtime.

## Development

After completing the editable source installation above, install and run the
development tooling:

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

Qwen 3.5 MTP speculative decoding loads the `nextn` block embedded in the same
GGUF; it does not download separate Hugging Face MTP weights:

```bash
vllm serve unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M \
  --tokenizer Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For a GGUF without a `nextn` block, omit `--speculative-config`; the backbone
loads normally without MTP.

## Adding an architecture

`gguf_map` diffs a GGUF's tensor names against the parameter names the target
model expects, and reports what is left over on each side and where shapes
disagree. It reads headers over an HTTP range request and builds the target on
meta tensors, so it needs no GPU and downloads no weights.

```bash
python -m vllm_gguf_plugin.tools.gguf_map \
  --gguf allenai/OLMoE-1B-7B-0924-Instruct-GGUF:olmoe-1b-7b-0924-instruct-q4_k_m.gguf \
  --target-model allenai/OLMoE-1B-7B-0924-Instruct \
  --target vllm --emit-adapter
```

`--target` picks what to reconcile against: `vllm` (the names `load_weights`
accepts), `hf` (a `transformers` state dict), or `safetensors` (the
checkpoint's own headers). Only the last works before either library supports
the architecture, which is the usual case when GGUF support lands first.

Pass `--gguf` more than once for a vision-language model, whose projector ships
as a separate mmproj file:

```bash
python -m vllm_gguf_plugin.tools.gguf_map \
  --gguf meta-models/Muse-Glimmer-30B-GGUF:Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf \
         meta-models/Muse-Glimmer-30B-GGUF:mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
  --target-model meta-models/Muse-Glimmer-30B --target safetensors
```

`--emit-adapter` prints a `WeightsMapper` skeleton to start from. It is a
starting point, not a result: value-level transforms produce neither a name nor
a shape mismatch, so confirm numerics separately.

## Tested model coverage

The plugin uses vLLM's model implementations and a generic GGUF weight
adapter, so model compatibility is broader than a fixed allowlist. The models
below are covered by the repository's generation tests and are the best-known
starting points:

| Modality | Model family | Tested GGUF quantization |
| --- | --- | --- |
| Text | Qwen 2.5 | Q6_K |
| Text | Qwen 3 | Q8_0 |
| Text | Phi 3.5 | IQ4_XS |
| Text | GPT-2 | Q4_K_M |
| Text | StableLM | Q4_K_M |
| Text | Gemma 3 | Q4_0 |
| Text | OLMoE | Q4_0 |
| Vision-language | Gemma 3 | Q4_0 backbone with F16 projector |
| Vision-language | Qwen 3.5 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.6 | UD-IQ2_XXS backbone with BF16 projector |
| Image generation | Z-Image-Turbo | Q4_0 |
| Image generation | FLUX.2-klein | Q8_0 |

Other vLLM-supported architectures may work when their GGUF tensor names map
to the corresponding Hugging Face model. A model appearing in vLLM's general
supported-model list does not by itself guarantee GGUF compatibility. When
reporting an unsupported model, include the model repository, quantization,
plugin and vLLM versions, and the complete weight-mapping error.
