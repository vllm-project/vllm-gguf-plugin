#include <optional>

#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

Tensor ggml_dequantize(Tensor W, int64_t type, int64_t m, int64_t n,
                       std::optional<ScalarType> dtype);
Tensor ggml_mul_mat_vec_a8(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_mul_mat_a8(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_moe_a8(Tensor X, Tensor W, Tensor sorted_token_ids,
                   Tensor expert_ids, Tensor num_tokens_post_padded,
                   int64_t type, int64_t row, int64_t top_k, int64_t tokens);
Tensor ggml_moe_a8_vec(Tensor X, Tensor W, Tensor topk_ids, int64_t top_k,
                       int64_t type, int64_t row, int64_t tokens);
int64_t ggml_moe_get_block_size(int64_t type);

STABLE_TORCH_LIBRARY(_C_gguf, ops) {
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");
  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");
  ops.def("ggml_moe_get_block_size(int type) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CUDA, ops) {
  ops.impl("ggml_dequantize", TORCH_BOX(&ggml_dequantize));
  ops.impl("ggml_mul_mat_vec_a8", TORCH_BOX(&ggml_mul_mat_vec_a8));
  ops.impl("ggml_mul_mat_a8", TORCH_BOX(&ggml_mul_mat_a8));
  ops.impl("ggml_moe_a8", TORCH_BOX(&ggml_moe_a8));
  ops.impl("ggml_moe_a8_vec", TORCH_BOX(&ggml_moe_a8_vec));
}

STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CompositeExplicitAutograd, ops) {
  ops.impl("ggml_moe_get_block_size", TORCH_BOX(&ggml_moe_get_block_size));
}

static struct PyModuleDef _module_def = {
    PyModuleDef_HEAD_INIT, "_C_gguf", nullptr, -1, nullptr,
};

extern "C" PyObject* PyInit__C_gguf(void) {
  return PyModule_Create(&_module_def);
}
