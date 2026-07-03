# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import gguf
import numpy as np
import torch


@dataclass
class MappedTensor:
    name: str
    tensor: Any
    tensor_type: Any
    row_slice: slice | None = None
    swap_scale_shift: bool = False


class DiffusionGGUFAdapter(ABC):
    """Base class for diffusion-model-specific GGUF adapters.

    Decoupled from vllm-omni: uses primitive ``model_class_name`` /
    ``model_type`` strings for compatibility checking instead of
    ``OmniDiffusionConfig``.
    """

    source_prefix = "transformer."
    source_subfolder = "transformer"

    def __init__(self, gguf_file: str) -> None:
        self.gguf_file = gguf_file

    @staticmethod
    def is_compatible(
        model_class_name: str | None,
        model_type: str | None,
    ) -> bool:
        return False

    @abstractmethod
    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError


def gguf_quant_weights_iterator(
    gguf_file: str,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Two-pass iterator over GGUF quantized weights.

    Pass 1 yields all ``qweight_type`` tensors first — weight types MUST
    come before weight data for packed layers with mixed quant types.
    Pass 2 yields all weight data (``qweight`` for quantized, original
    name for unquantized).
    """

    reader = gguf.GGUFReader(gguf_file)

    # Pass 1: weight types first
    for tensor in reader.tensors:
        weight_type = tensor.tensor_type
        name = tensor.name

        if weight_type.name not in ("F32", "F16"):
            weight_type_name = name.replace("weight", "qweight_type")
            weight_type = torch.tensor(weight_type)
            yield weight_type_name, weight_type

    # Pass 2: weight data
    for tensor in reader.tensors:
        weight = tensor.data
        weight_type = tensor.tensor_type
        name = tensor.name
        if weight_type.name not in ("F32", "F16"):
            name = name.replace("weight", "qweight")
        if weight_type.name == "BF16" and tensor.data.dtype == np.uint8:
            # BF16 is currently the only "quantization" type that isn't
            # actually quantized but is read as a raw byte tensor.
            # Reinterpret as `torch.bfloat16` tensor.
            weight = weight.view(np.uint16)
            if reader.byte_order == "S":
                # GGUF endianness != system endianness
                weight = weight.byteswap()
            param = torch.tensor(weight).view(torch.bfloat16)
        else:
            param = torch.tensor(weight)
        yield name, param
