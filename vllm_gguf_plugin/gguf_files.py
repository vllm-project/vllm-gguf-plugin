# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GGUFModelFiles:
    """Resolved GGUF files grouped by their role in one model."""

    backbone: tuple[str, ...]
    mm_proj: str | None = None

    def __post_init__(self) -> None:
        if not self.backbone:
            raise ValueError("GGUFModelFiles requires at least one backbone file")

    @property
    def primary_backbone(self) -> str:
        return self.backbone[0]

    @property
    def all_files(self) -> tuple[str, ...]:
        if self.mm_proj is None:
            return self.backbone
        return (*self.backbone, self.mm_proj)
