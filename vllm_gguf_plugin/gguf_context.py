# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Share GGUF request details with the config parser.

``EngineArgs.create_model_config`` rewrites ``model`` to the HF config source
and moves the GGUF reference to ``model_weights``, so by the time the config
parser runs it no longer sees the weights it is parsing a config for. The
parser also decides the model architecture before any model loader exists, so
it cannot read ``model_loader_extra_config`` either. Both are recorded here
while the engine args are resolved.
"""

_explicit_mm_proj: object | None = None
_gguf_weights: str | None = None


def set_gguf_request(weights: str | None, mm_proj: object | None) -> None:
    global _explicit_mm_proj, _gguf_weights
    _gguf_weights = weights
    _explicit_mm_proj = mm_proj


def get_explicit_mm_proj() -> object | None:
    return _explicit_mm_proj


def get_gguf_weights() -> str | None:
    return _gguf_weights
