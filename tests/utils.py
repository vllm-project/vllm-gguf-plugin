import random
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor
from huggingface_hub import snapshot_download


def seed_everything(seed: int) -> None:
    """
    Set the seed of each random module.
    `torch.manual_seed` will set seed on all devices.

    Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")


def get_gguf_sample_tensors(
    hidden_size: int, quant_type: GGMLQuantizationType
) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors
