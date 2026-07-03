# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the diffusion GGUF loader."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from gguf import GGMLQuantizationType as WeightType

from vllm_gguf_plugin.weights_adapter.diffusion import (
    DiffusionWeightSource,
    get_gguf_model_from_config,
    is_gguf_quant_config,
    load_diffusion_gguf_weights,
    resolve_gguf_model_path,
)

pytestmark = [pytest.mark.cpu]


# ---- is_gguf_quant_config ----


def test_is_gguf_true_for_object_with_get_name():
    config = SimpleNamespace(get_name=lambda: "gguf")
    assert is_gguf_quant_config(config) is True


def test_is_gguf_true_for_dict_with_method_gguf():
    assert is_gguf_quant_config({"method": "gguf", "gguf_model": "x.gguf"}) is True


def test_is_gguf_false_for_fp8_dict():
    assert is_gguf_quant_config({"method": "fp8"}) is False


def test_is_gguf_false_for_non_gguf_object():
    config = SimpleNamespace(get_name=lambda: "fp8")
    assert is_gguf_quant_config(config) is False


def test_is_gguf_false_for_none():
    assert is_gguf_quant_config(None) is False


# ---- get_gguf_model_from_config ----


def test_get_gguf_model_from_dict():
    assert get_gguf_model_from_config({"gguf_model": "path.gguf"}) == "path.gguf"


def test_get_gguf_model_from_object():
    assert get_gguf_model_from_config(SimpleNamespace(gguf_model="x.gguf")) == "x.gguf"


def test_get_gguf_model_from_none():
    assert get_gguf_model_from_config(None) is None


def test_get_gguf_model_missing_key():
    assert get_gguf_model_from_config({"method": "fp8"}) is None


# ---- resolve_gguf_model_path ----


def test_resolve_returns_local_file(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"gguf")
    assert resolve_gguf_model_path(str(gguf_file)) == str(gguf_file)


def test_resolve_returns_local_quant_selector(tmp_path):
    gguf_file = tmp_path / "model-Q4_0.gguf"
    gguf_file.write_bytes(b"gguf")
    assert resolve_gguf_model_path(f"{tmp_path}:Q4_0") == str(gguf_file)


def test_resolve_raises_on_unrecognized_format():
    with pytest.raises(ValueError, match="Unrecognized GGUF reference"):
        resolve_gguf_model_path("not-a-valid-ref")


# ---- load_diffusion_gguf_weights ----


class _FakeModel(nn.Module):
    """Minimal model with ComponentSource-like weights_sources."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=True)
        self.vae = nn.Linear(2, 2, bias=False)
        self.register_buffer("transformer_buffer", torch.ones(1))

    def load_weights(self, weights) -> set[str]:
        loadable = {name: param for name, param in self.named_parameters()}
        loadable.update(dict(self.named_buffers()))
        loaded: set[str] = set()
        for name, tensor in weights:
            if name in loadable:
                loadable[name].data.copy_(tensor.to(dtype=loadable[name].dtype))
                loaded.add(name)
        return loaded


def _make_sources():
    return [
        DiffusionWeightSource(prefix="transformer.", subfolder="transformer"),
        DiffusionWeightSource(prefix="vae.", subfolder="vae"),
    ]


def test_load_gguf_transformer_and_hf_non_transformer(monkeypatch: pytest.MonkeyPatch):
    """Transformer loads from GGUF, VAE loads from HF callback."""
    model = _FakeModel()

    class _Adapter:
        def weights_iterator(self):
            yield "weight", torch.ones((2, 2))
            yield "bias", torch.zeros(2)

    import vllm_gguf_plugin.weights_adapter.diffusion.loader as _loader_mod

    monkeypatch.setattr(
        _loader_mod, "resolve_gguf_model_path", lambda **kw: "dummy.gguf"
    )
    monkeypatch.setattr(
        _loader_mod, "get_diffusion_gguf_adapter", lambda *a, **kw: _Adapter()
    )

    hf_calls: list[str] = []

    def hf_fn(source: DiffusionWeightSource):
        hf_calls.append(source.subfolder)
        if source.subfolder == "vae":
            yield "vae.weight", torch.full((2, 2), 2.0)
        else:
            yield from ()

    sources = _make_sources()
    loaded = load_diffusion_gguf_weights(
        gguf_model="dummy.gguf",
        model=model,
        model_class_name=None,
        model_type=None,
        sources=sources,
        hf_weights_fn=hf_fn,
    )

    assert "transformer.weight" in loaded
    assert "transformer.bias" in loaded
    assert "vae.weight" in loaded
    assert hf_calls == ["vae"]


def test_load_gguf_does_not_fallback_to_hf_for_missing_transformer_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    """A GGUF transformer source never loads missing weights from HF."""
    model = _FakeModel()

    class _Adapter:
        def weights_iterator(self):
            yield "weight", torch.ones((2, 2))

    import vllm_gguf_plugin.weights_adapter.diffusion.loader as _loader_mod

    monkeypatch.setattr(
        _loader_mod, "resolve_gguf_model_path", lambda **kw: "dummy.gguf"
    )
    monkeypatch.setattr(
        _loader_mod, "get_diffusion_gguf_adapter", lambda *a, **kw: _Adapter()
    )

    hf_seen: list[str] = []

    def hf_fn(source: DiffusionWeightSource):
        hf_seen.append(source.subfolder)
        yield "transformer.bias", torch.zeros(2, dtype=torch.float32)
        yield "vae.weight", torch.full((2, 2), 2.0, dtype=torch.float32)

    sources = _make_sources()
    loaded = load_diffusion_gguf_weights(
        gguf_model="dummy.gguf",
        model=model,
        model_class_name=None,
        model_type=None,
        sources=sources,
        hf_weights_fn=hf_fn,
    )

    assert "transformer.weight" in loaded
    assert "transformer.bias" not in loaded
    assert "vae.weight" in loaded
    assert hf_seen == ["vae"]


def test_load_gguf_restores_plain_weight_from_gguf_qweight(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _FakeModel()

    class _Adapter:
        def weights_iterator(self):
            yield "qweight_type", torch.tensor(WeightType.F32)
            yield "qweight", torch.full((2, 2), 3.0)

    import vllm_gguf_plugin.weights_adapter.diffusion.loader as _loader_mod

    monkeypatch.setattr(
        _loader_mod, "resolve_gguf_model_path", lambda **kw: "dummy.gguf"
    )
    monkeypatch.setattr(
        _loader_mod, "get_diffusion_gguf_adapter", lambda *a, **kw: _Adapter()
    )

    hf_calls: list[str] = []

    def hf_fn(source: DiffusionWeightSource):
        hf_calls.append(source.subfolder)
        yield "transformer.weight", torch.zeros((2, 2))

    loaded = load_diffusion_gguf_weights(
        gguf_model="dummy.gguf",
        model=model,
        model_class_name=None,
        model_type=None,
        sources=[DiffusionWeightSource(prefix="transformer.", subfolder="transformer")],
        hf_weights_fn=hf_fn,
    )

    assert "transformer.weight" in loaded
    assert torch.allclose(model.transformer.weight, torch.full((2, 2), 3.0))
    assert hf_calls == []


def test_load_gguf_source_is_selected_by_adapter(monkeypatch: pytest.MonkeyPatch):
    model = _FakeModel()

    class _Adapter:
        source_prefix = "vae."
        source_subfolder = "vae"

        def weights_iterator(self):
            yield "weight", torch.full((2, 2), 4.0)

    import vllm_gguf_plugin.weights_adapter.diffusion.loader as _loader_mod

    monkeypatch.setattr(
        _loader_mod, "resolve_gguf_model_path", lambda **kw: "dummy.gguf"
    )
    monkeypatch.setattr(
        _loader_mod, "get_diffusion_gguf_adapter", lambda *a, **kw: _Adapter()
    )

    hf_calls: list[str] = []

    def hf_fn(source: DiffusionWeightSource):
        hf_calls.append(source.subfolder)
        if source.subfolder == "transformer":
            yield "transformer.weight", torch.ones((2, 2))
            yield "transformer.bias", torch.zeros(2)

    loaded = load_diffusion_gguf_weights(
        gguf_model="dummy.gguf",
        model=model,
        model_class_name=None,
        model_type=None,
        sources=_make_sources(),
        hf_weights_fn=hf_fn,
    )

    assert "vae.weight" in loaded
    assert "transformer.weight" in loaded
    assert hf_calls == ["transformer"]
    assert torch.allclose(model.vae.weight, torch.full((2, 2), 4.0))


def test_load_gguf_skips_hf_when_complete(monkeypatch: pytest.MonkeyPatch):
    """No HF fallback when GGUF covers all transformer weights."""
    model = _FakeModel()

    class _Adapter:
        def weights_iterator(self):
            yield "weight", torch.ones((2, 2))
            yield "bias", torch.zeros(2, dtype=torch.float32)

    import vllm_gguf_plugin.weights_adapter.diffusion.loader as _loader_mod

    monkeypatch.setattr(
        _loader_mod, "resolve_gguf_model_path", lambda **kw: "dummy.gguf"
    )
    monkeypatch.setattr(
        _loader_mod, "get_diffusion_gguf_adapter", lambda *a, **kw: _Adapter()
    )

    hf_calls: list[str] = []

    def hf_fn(source: DiffusionWeightSource):
        hf_calls.append(source.subfolder)
        yield from ()

    sources = _make_sources()
    loaded = load_diffusion_gguf_weights(
        gguf_model="dummy.gguf",
        model=model,
        model_class_name=None,
        model_type=None,
        sources=sources,
        hf_weights_fn=hf_fn,
    )

    assert "transformer.weight" in loaded
    assert "transformer.bias" in loaded
    # HF should be called for vae, but transformer hf should be skipped
    assert "vae" in hf_calls
