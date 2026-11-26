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

### Windows builds

CI builds native Windows x64 and ARM64 wheels on pushes and pull requests.
The release workflow reuses these builds and publishes the following wheels:

| Platform | CUDA | Publication |
| --- | --- | --- |
| Windows x64 | 13.0 | GitHub Releases, PyPI and TestPyPI |
| Windows ARM64 | 13.4 | GitHub Releases, PyPI and TestPyPI |

Stable releases go to PyPI; prereleases go to TestPyPI. Manual release runs
only upload workflow artifacts, as for Linux.

Windows builds require native Python, MSVC C++ build tools and the matching
CUDA toolkit. Like Linux, Windows CI requests `torch>=2.9` rather than
pinning a PyTorch release. On x64, `uv` selects the wheel for the requested
CUDA backend. For Windows ARM64 with CUDA 13.4, use the newest `+cu134`
wheel for your Python version from [NVIDIA's PyTorch index](https://pypi.nvidia.com/nvtorch_oot/torch/).

Run source builds from a Visual Studio developer shell targeting the same
architecture as Python. The release helper `scripts/build_release_wheel.sh`
runs under Git Bash on Windows and accepts an explicit
`TORCH_CUDA_ARCH_LIST`; ARM64 releases target RTX Spark (`12.1+PTX`).

CI checks that the installed extension loads and registers its operators
without a GPU. It does not run inference: using the plugin still requires a
Windows-compatible vLLM installation, a supported NVIDIA GPU and driver,
and PyTorch built for the matching CUDA version.

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
