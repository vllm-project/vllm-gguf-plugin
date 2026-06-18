# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_smoke_module():
    smoke_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "plugin_gguf_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("plugin_gguf_smoke", smoke_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_parse_args_accepts_runtime_controls(monkeypatch):
    smoke = _load_smoke_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plugin_gguf_smoke.py",
            "--max-num-seqs",
            "1",
            "--moe-backend",
            "marlin",
            "--min-generated-tokens",
            "4",
            "--disable-v1-multiprocessing",
        ],
    )

    args = smoke.parse_args()

    assert args.max_num_seqs == 1
    assert args.moe_backend == "marlin"
    assert args.min_generated_tokens == 4
    assert args.disable_v1_multiprocessing is True


def test_smoke_build_llm_kwargs_includes_large_model_controls():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        model="/models/gpt-oss.gguf",
        tokenizer="openai/gpt-oss-20b",
        dtype="bfloat16",
        max_model_len=64,
        max_num_seqs=1,
        gpu_memory_utilization=0.2,
        enforce_eager=True,
        trust_remote_code=False,
        hf_config_path=None,
        moe_backend="marlin",
    )

    kwargs = smoke.build_llm_kwargs(args)

    assert kwargs == {
        "model": "/models/gpt-oss.gguf",
        "tokenizer": "openai/gpt-oss-20b",
        "load_format": "gguf",
        "quantization": "gguf",
        "dtype": "bfloat16",
        "max_model_len": 64,
        "gpu_memory_utilization": 0.2,
        "enforce_eager": True,
        "trust_remote_code": False,
        "max_num_seqs": 1,
        "moe_backend": "marlin",
    }


def test_smoke_build_llm_kwargs_omits_unset_optional_controls():
    smoke = _load_smoke_module()
    args = argparse.Namespace(
        model="/models/model.gguf",
        tokenizer="/models/tokenizer",
        dtype="float16",
        max_model_len=1024,
        max_num_seqs=None,
        gpu_memory_utilization=0.25,
        enforce_eager=False,
        trust_remote_code=False,
        hf_config_path=None,
        moe_backend=None,
    )

    kwargs = smoke.build_llm_kwargs(args)

    assert "max_num_seqs" not in kwargs
    assert "moe_backend" not in kwargs
    assert "hf_config_path" not in kwargs
