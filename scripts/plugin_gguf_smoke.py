#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test vllm-gguf-plugin registration and optional GGUF generation.

The default mode is intentionally lightweight: it imports vLLM, calls the
plugin's register hook, and reports whether GGUF is served by in-tree vLLM or by
this out-of-tree plugin. Pass --generate with a small local GGUF plus tokenizer
and config to exercise the real vLLM load/generate path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def object_module(value: Any) -> str:
    return getattr(value, "__module__", type(value).__module__)


def object_name(value: Any) -> str:
    return getattr(value, "__name__", type(value).__name__)


def plugin_quantization_imported() -> bool:
    return any(
        name == "vllm_gguf_plugin.quantization"
        or name.startswith("vllm_gguf_plugin.quantization.")
        for name in sys.modules
    )


def ensure_python_bin_on_path() -> None:
    python_bin = Path(sys.executable).parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"


def plugin_version() -> str:
    try:
        return metadata.version("vllm-gguf-plugin")
    except metadata.PackageNotFoundError:
        return "source-checkout"


def import_state() -> dict[str, Any]:
    import vllm
    from vllm.config.load import LoadConfig
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.transformers_utils.config import get_config_parser

    in_tree = "gguf" in QUANTIZATION_METHODS
    quant_config: Any | None = None
    model_loader: Any | None = None
    config_parser: Any | None = None
    errors: dict[str, str] = {}

    if in_tree:
        try:
            quant_config = get_quantization_config("gguf")
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors["quant_config"] = f"{type(exc).__name__}: {exc}"
        try:
            model_loader = get_model_loader(LoadConfig(load_format="gguf"))
        except Exception as exc:
            errors["model_loader"] = f"{type(exc).__name__}: {exc}"
        try:
            config_parser = get_config_parser("gguf")
        except Exception as exc:
            errors["config_parser"] = f"{type(exc).__name__}: {exc}"

    return {
        "vllmVersion": getattr(vllm, "__version__", None),
        "hasGgufQuantization": in_tree,
        "quantizationMethodsHasGguf": in_tree,
        "quantConfig": None
        if quant_config is None
        else f"{object_module(quant_config)}.{object_name(quant_config)}",
        "modelLoader": None
        if model_loader is None
        else f"{object_module(type(model_loader))}.{object_name(type(model_loader))}",
        "configParser": None
        if config_parser is None
        else f"{object_module(type(config_parser))}.{object_name(type(config_parser))}",
        "pluginQuantizationImported": plugin_quantization_imported(),
        "errors": errors,
    }


def expected_verdict(args: argparse.Namespace, before: dict[str, Any]) -> str:
    if args.expect != "auto":
        return args.expect
    if before["hasGgufQuantization"] and not args.override_in_tree:
        return "skipped"
    return "active"


def actual_verdict(after: dict[str, Any]) -> str:
    quant = after.get("quantConfig") or ""
    loader = after.get("modelLoader") or ""
    parser = after.get("configParser") or ""
    plugin_owned = (
        quant.startswith("vllm_gguf_plugin.")
        and loader.startswith("vllm_gguf_plugin.")
        and parser.startswith("vllm_gguf_plugin.")
    )
    return "active" if plugin_owned else "skipped"


def run_generation(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.generate:
        return None
    if not args.model:
        raise SystemExit("--generate requires --model")
    if not args.tokenizer:
        raise SystemExit("--generate requires --tokenizer")

    from vllm import LLM, SamplingParams

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "load_format": "gguf",
        "quantization": "gguf",
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.hf_config_path:
        kwargs["hf_config_path"] = args.hf_config_path

    llm = LLM(**kwargs)
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
        ),
    )
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    return {
        "model": args.model,
        "prompt": args.prompt,
        "text": text,
        "tokens": len(outputs[0].outputs[0].token_ids)
        if outputs and outputs[0].outputs
        else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=["auto", "active", "skipped"], default="auto")
    parser.add_argument("--override-in-tree", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--hf-config-path")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Reply with exactly: gguf ok")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_python_bin_on_path()
    if args.override_in_tree:
        os.environ["VLLM_GGUF_PLUGIN_OVERRIDE_IN_TREE"] = "1"

    before = import_state()

    import vllm_gguf_plugin

    imported_without_quantization = not plugin_quantization_imported()
    vllm_gguf_plugin.register()
    after = import_state()

    expected = expected_verdict(args, before)
    actual = actual_verdict(after)
    ok = actual == expected and imported_without_quantization

    generation = None
    if ok:
        generation = run_generation(args)

    result = {
        "ok": ok,
        "expected": expected,
        "actual": actual,
        "python": sys.executable,
        "pluginVersion": plugin_version(),
        "importedWithoutQuantization": imported_without_quantization,
        "before": before,
        "after": after,
        "generation": generation,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"vllm-gguf-plugin smoke: expected={expected} actual={actual} "
            f"ok={str(ok).lower()}"
        )
        print(
            f"vllm={after['vllmVersion']} plugin={result['pluginVersion']} "
            f"quant={after['quantConfig']} loader={after['modelLoader']} "
            f"parser={after['configParser']}"
        )
        if generation:
            print(f"generation tokens={generation['tokens']} text={generation['text']!r}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
