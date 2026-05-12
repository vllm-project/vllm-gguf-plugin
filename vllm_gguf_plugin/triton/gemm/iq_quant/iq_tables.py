from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from gguf.quants import IQ3_S


_COMMON_H = Path(__file__).resolve().parents[3] / "csrc" / "gguf" / "ggml-common.h"


@lru_cache(maxsize=1)
def _common_h_text() -> str:
    return _COMMON_H.read_text()


def _parse_array(name: str, dtype: np.dtype) -> np.ndarray:
    pattern = rf"static const __device__ [^ ]+ {name}\[[^\]]+\] = \{{(.*?)\}};"
    match = re.search(pattern, _common_h_text(), re.S)
    if match is None:
        raise ValueError(f"Could not find {name} in {_COMMON_H}")

    body = match.group(1)
    values = [int(token.strip(), 0) for token in body.replace("\n", " ").split(",") if token.strip()]
    return np.array(values, dtype=dtype)


@lru_cache(maxsize=1)
def _cpu_iq_tables() -> dict[str, np.ndarray]:
    IQ3_S.init_grid()
    iq2xs_grid = _parse_array("iq2xs_grid", np.uint64).view(np.uint8).reshape(-1, 8).copy()
    iq2s_grid = _parse_array("iq2s_grid", np.uint64).view(np.uint8).reshape(-1, 8).copy()
    iq3xxs_grid = _parse_array("iq3xxs_grid", np.uint32).view(np.uint8).reshape(-1, 4).copy()
    iq3xs_grid = _parse_array("iq3xs_grid", np.uint32).view(np.uint8).reshape(-1, 4).copy()
    iq3s_grid = IQ3_S.grid[0, 0].astype(np.uint8).copy()
    iq1s_grid_raw = _parse_array("iq1s_grid_gpu", np.uint64).astype(np.uint32)
    iq1s_bytes = iq1s_grid_raw.view(np.uint8).reshape(-1, 4)
    iq1s_grid = np.concatenate([iq1s_bytes & 0x0F, (iq1s_bytes >> 4) & 0x0F], axis=1).astype(np.int8)

    return {
        "iq2xs_grid": iq2xs_grid,
        "iq2s_grid": iq2s_grid,
        "iq3xxs_grid": iq3xxs_grid,
        "iq3xs_grid": iq3xs_grid,
        "iq3s_grid": iq3s_grid,
        "iq1s_grid": iq1s_grid,
        "ksigns_iq2xs": _parse_array("ksigns_iq2xs", np.uint8).copy(),
        "kvalues_iq4nl": _parse_array("kvalues_iq4nl", np.int8).copy(),
    }


_DEVICE_TABLES: dict[tuple[str, int | None], dict[str, torch.Tensor]] = {}


def get_iq_table_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    key = (device.type, device.index)
    if key in _DEVICE_TABLES:
        return _DEVICE_TABLES[key]

    cpu_tables = _cpu_iq_tables()
    device_tables = {
        name: torch.tensor(values, device=device)
        for name, values in cpu_tables.items()
    }
    _DEVICE_TABLES[key] = device_tables
    return device_tables
