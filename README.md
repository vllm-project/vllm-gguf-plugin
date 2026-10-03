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

When the `nextn` block is in a separate GGUF, pass that file as the speculative
model. The draft uses the target model's Hugging Face config and loads only the
MTP block from the separate file:

```bash
vllm serve /path/to/Qwen3.8-27B-Q4_0.gguf \
  --tokenizer /path/to/Qwen3.8-27B \
  --speculative-config '{"method":"mtp","model":"/path/to/mtp-Qwen3.8-27B-Q4_0.gguf","num_speculative_tokens":1}'
```

## Kernel backend selection

On CUDA builds the plugin ships three kernel implementations for GGUF
quantized operations — upstream (llama.cpp kernels, enabled by default),
legacy (the plugin's original CUDA kernels), and Triton. The implementation
is chosen per operation through environment variables:

| Variable | Scope | Default |
| --- | --- | --- |
| `VLLM_GGUF_CUDA_KERNEL` | Global default for all operations | `auto` |
| `VLLM_GGUF_CUDA_DENSE_KERNEL` | Dense projections | unset (falls back to the global variable) |
| `VLLM_GGUF_CUDA_MOE_KERNEL` | Routed-expert MoE | unset (falls back to the global variable) |
| `VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL` | Dequantization | unset (falls back to the global variable) |

Each variable accepts `auto`, `upstream`, `legacy`, or `triton`. A
specialized variable overrides the global one; if neither is set, the
operation uses `auto`.

`auto` and `upstream` select methods only within the upstream backend.
Unavailable upstream operations or unsupported inputs raise an error; they
never switch to legacy CUDA or Triton. Select `legacy` or `triton` explicitly
to use those backends. ROCm and legacy-only builds require an explicit
`legacy` selection.

Upstream decision interfaces are `ops.ggml_dense(W, X, type, row)` and
`ops.ggml_moe(X, W, topk_ids, type, row, top_k, tokens)`. Method selection
lives in the C++ Dense/MoE selectors. Fixed methods use the same names with
`_mmvq`, `_mmq`, `_mmvf`, `_mmf`, `_blas`, or `_dequantize_blas` suffixes;
MoE also has `_grouped_dense` and `_mmq_aligned`. Fixed methods check execution
constraints and run even when the performance policy would prefer another
method. `dense_supported_methods` / `moe_supported_methods` return supported
method bits; `dense_select_method` / `moe_select_method` return one selected
`KernelMethod` value. Family-level `supports()` does not check input shapes
or devices.

`blas` accepts floating weights; packed weights use `dequantize_blas`.
Raw MoE MMQ accepts rank-2 expert IDs. Aligned MoE MMQ accepts rank-1 sorted
route IDs plus required `expert_ids` and `padded_count`; these layouts have
separate interfaces. MoE BLAS and grouped Dense currently require host
routing and cannot execute during CUDA graph capture. The decision interface
uses another supported upstream method during capture or raises an error.

Example — pin dense kernels to legacy while keeping the automatic selection
elsewhere:

```bash
VLLM_GGUF_CUDA_DENSE_KERNEL=legacy vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 \
  --tokenizer Qwen/Qwen3-0.6B
```

See `doc/upstream.md` for details on the upstream integration, weight
storage padding, and the supported quantization types per backend.

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
| Vision-language | Gemma 4 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.5 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.6 | UD-IQ2_XXS backbone with BF16 projector |
| Image generation | Z-Image-Turbo | Q4_0 |
| Image generation | FLUX.2-klein | Q8_0 |

Other vLLM-supported architectures may work when their GGUF tensor names map
to the corresponding Hugging Face model. A model appearing in vLLM's general
supported-model list does not by itself guarantee GGUF compatibility. When
reporting an unsupported model, include the model repository, quantization,
plugin and vLLM versions, and the complete weight-mapping error.

## License

This project is licensed under Apache-2.0. It includes and derives portions
from llama.cpp and GGML under the MIT License. See `THIRD_PARTY_NOTICES.md` and
`LICENSES/llama.cpp-MIT.txt` for attribution and the complete license text.
