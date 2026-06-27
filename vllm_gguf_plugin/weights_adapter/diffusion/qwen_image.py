# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

from .base import DiffusionGGUFAdapter, gguf_quant_weights_iterator


class QwenImageDiffusionGGUFAdapter(DiffusionGGUFAdapter):
    """GGUF adapter for the Qwen-Image transformer family."""

    @staticmethod
    def is_compatible(
        model_class_name: str | None,
        model_type: str | None,
    ) -> bool:
        if model_class_name and model_class_name.startswith("QwenImage"):
            return True
        return bool(model_type and model_type.lower().startswith("qwen_image"))

    def weights_iterator(self) -> Iterable[tuple[str, object]]:
        yield from gguf_quant_weights_iterator(self.gguf_file)
