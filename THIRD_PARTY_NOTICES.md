# Third-Party Notices

This distribution contains code from
[llama.cpp](https://github.com/ggml-org/llama.cpp) and its GGML library.

- License: MIT
- Upstream revision used by the native CUDA integration:
  `002a12ad25503a93501b2e188c360029830a241a`
- Complete license text: `LICENSES/llama.cpp-MIT.txt`

The selected, unmodified upstream CUDA sources and headers are recorded in
`vllm_gguf_plugin/llama_cpp_upstream.toml` and compiled from
`third_party/llama.cpp`.

The legacy CUDA implementation also contains code copied or adapted from
llama.cpp build `b2899` (`0350f5815218c483fb3026a86adc44a115481625`),
including files under `vllm_gguf_plugin/csrc/gguf` that identify their source
in file comments.

The vllm-gguf-plugin project and its original integration code remain licensed
under Apache-2.0. The llama.cpp and GGML portions remain subject to the MIT
License provided in `LICENSES/llama.cpp-MIT.txt`.
