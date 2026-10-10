# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards for the grid tables vendored in ``csrc/gguf/ggml-common.h``.

The IQ3_S kernels index ``iq3xs_grid`` directly and fold the block scale into
the table, so every entry must be four times one of the odd magnitudes
(1, 3, ..., 15) that llama.cpp stores in ``iq3s_grid``. An entry that is not of
that form dequantizes to a value that is off by a constant factor with no
runtime error, which is how a single wrong byte stayed unnoticed.
"""

import re
from pathlib import Path

HEADER = (
    Path(__file__).resolve().parents[1]
    / "vllm_gguf_plugin"
    / "csrc"
    / "gguf"
    / "ggml-common.h"
)

IQ3_S_GRID_MAGNITUDES = {1, 3, 5, 7, 9, 11, 13, 15}


def _read_uint32_table(name: str, size: int) -> list[int]:
    text = HEADER.read_text()
    marker = f"static const __device__ uint32_t {name}[{size}] = {{"
    start = text.index(marker) + len(marker)
    end = text.index("};", start)
    words = [int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", text[start:end])]
    assert len(words) == size
    return words


def _table_bytes(words: list[int]) -> list[int]:
    return [(word >> (8 * byte)) & 0xFF for word in words for byte in range(4)]


def test_iq3xs_grid_entries_are_scaled_odd_magnitudes():
    entries = _table_bytes(_read_uint32_table("iq3xs_grid", 512))

    assert all(entry % 4 == 0 for entry in entries)
    assert {entry // 4 for entry in entries} == IQ3_S_GRID_MAGNITUDES
