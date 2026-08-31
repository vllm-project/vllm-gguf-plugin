# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""STQ1_0 support: gguf-py type registration and a torch reference codec.

STQ1_0 (GGML type id 43, from llama.cpp PR #22836) is a ternary format used by
e.g. AngelSlim/Hy4-preview-GGUF. Each 256-weight block packs into 42 bytes:

- ``qs[32]``: two 4-bit slot codes per byte (low nibble first), one code per
  4-lane group;
- ``sign[8]``: one table-select bit per group (bit k of byte b -> group 8b+k);
- ``d``: fp16 block scale.

A group decodes via ``qpack = CODEBOOK[(sign << 4) | slot]``; lane p is
``(qpack >> 2p) & 3`` mapped to {-1, 0, +1} as {0, 1, 2} - 1, times ``d``.

Beware the layout: group g (chunk = g//16, gloc = g%16) holds the four weights
``chunk*64 + gloc + p*16`` — stride-16, NOT four contiguous lanes. Every codec
in the patched llama.cpp (quantize_row_stq1_0_*, dequantize_row_stq1_0,
ggml_vec_dot_stq1_0_q8_K) agrees on this.

Stock gguf-py cannot parse type 43, so it is registered into
``gguf.GGMLQuantizationType``/``GGML_QUANT_SIZES`` at import time. The actual
matmul runs on native CUDA kernels (see csrc/gguf); ``dequantize_stq1_0`` here
is the reference implementation for tests.
"""

from __future__ import annotations

import gguf
import torch

STQ1_0_TYPE_ID = 43
STQ1_0_BLOCK_SIZE = 256
STQ1_0_TYPE_SIZE = 42

# Index = (sign << 4) | slot; lane p occupies bits [2p, 2p+2) with
# 0b00 -> -1, 0b01 -> 0, 0b10 -> +1. See ggml-common.h stq1_0_codebook.
STQ1_0_CODEBOOK = (
    # sign = 0 (first non-zero lane is +1)
    0xA9,
    0x89,
    0x29,
    0x09,
    0xA6,
    0x86,
    0x26,
    0x06,
    0x9A,
    0x92,
    0x1A,
    0x12,
    0x6A,
    0x62,
    0x4A,
    0x42,
    # sign = 1 (every non-zero lane negated)
    0x01,
    0x21,
    0x81,
    0xA1,
    0x04,
    0x24,
    0x84,
    0xA4,
    0x10,
    0x18,
    0x90,
    0x98,
    0x40,
    0x48,
    0x60,
    0x68,
)


def register_stq1_0_gguf_type() -> None:
    """Register STQ1_0 with gguf-py so GGUFReader can parse such files."""
    if STQ1_0_TYPE_ID in gguf.GGMLQuantizationType._value2member_map_:
        return
    qtype = gguf.GGMLQuantizationType
    member = int.__new__(qtype, STQ1_0_TYPE_ID)
    member._name_ = "STQ1_0"
    member._value_ = STQ1_0_TYPE_ID
    # setattr must come first: once the name is in _member_map_, EnumType
    # rejects the attribute assignment as a member reassignment.
    qtype.STQ1_0 = member
    qtype._value2member_map_[STQ1_0_TYPE_ID] = member
    qtype._member_map_["STQ1_0"] = member
    gguf.GGML_QUANT_SIZES[member] = (STQ1_0_BLOCK_SIZE, STQ1_0_TYPE_SIZE)


def get_stq1_0_type() -> gguf.GGMLQuantizationType:
    """Return the (registered) STQ1_0 quantization type."""
    return gguf.GGMLQuantizationType(STQ1_0_TYPE_ID)


register_stq1_0_gguf_type()


def dequantize_stq1_0(raw: torch.Tensor) -> torch.Tensor:
    """Dequantize raw STQ1_0 blocks to float32 (reference implementation).

    Args:
        raw: uint8 tensor of shape [..., n * 42] (n blocks per row).

    Returns:
        Float32 tensor of shape [..., n * 256].
    """
    *lead, last = raw.shape
    if last % STQ1_0_TYPE_SIZE != 0:
        raise ValueError(f"STQ1_0 blocks are 42 bytes, got shape {raw.shape}")
    n_blocks = last // STQ1_0_TYPE_SIZE
    raw = raw.reshape(-1, STQ1_0_TYPE_SIZE)
    device = raw.device
    codebook = torch.tensor(STQ1_0_CODEBOOK, dtype=torch.int32, device=device)

    qs = raw[..., :32].to(torch.int32)
    sign = raw[..., 32:40].to(torch.int32)
    d = raw[..., 40:42].contiguous().view(torch.float16).to(torch.float32)

    # Low nibble first: byte j holds the slot codes of groups 2j and 2j+1.
    slots = torch.stack((qs & 0xF, qs >> 4), dim=-1).flatten(-2)
    bits = (sign[..., None] >> torch.arange(8, device=device)) & 1
    idx = slots | (bits.flatten(-2) << 4)

    qpack = codebook[idx]
    lanes = (qpack[..., None] >> (2 * torch.arange(4, device=device))) & 3
    vals = (lanes - 1).to(torch.float32)  # [B, group(64), p(4)]
    # group g = chunk*16 + gloc covers weights chunk*64 + gloc + p*16:
    # [B, chunk, gloc, p] -> [B, chunk, p, gloc] -> [B, 256]
    vals = vals.unflatten(-2, (4, 16)).permute(0, 1, 3, 2).reshape(-1, 256)
    return (vals * d).reshape(*lead, n_blocks * STQ1_0_BLOCK_SIZE)
