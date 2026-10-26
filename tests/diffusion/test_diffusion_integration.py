# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import ModuleType

import pytest
import torch.nn as nn

from vllm_gguf_plugin.weights_adapter.diffusion.integration import (
    _patch_diffusers_loader,
)


@pytest.mark.cpu
@pytest.mark.parametrize("method", [None, "bitsandbytes", "fp8"])
@pytest.mark.parametrize("stream", [False, True])
def test_non_gguf_weight_loading_preserves_keywords(monkeypatch, method, stream):
    calls = []

    class Loader:
        quant_config = {"method": method} if method else None

        def load_weights(self, model, *, stream_online_quant_to_cpu=False):
            calls.append((model, stream_online_quant_to_cpu))

        def load_model(self, model):
            self.load_weights(model, stream_online_quant_to_cpu=stream)
            return model

    module = ModuleType("vllm_omni.diffusion.model_loader.diffusers_loader")
    module.DiffusersPipelineLoader = Loader
    monkeypatch.setitem(sys.modules, module.__name__, module)
    _patch_diffusers_loader()
    _patch_diffusers_loader()
    model = nn.Linear(2, 2)
    assert Loader().load_model(model) is model
    assert calls == [(model, stream)]
