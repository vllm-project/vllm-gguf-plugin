#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a matrix of isolated vllm-gguf-plugin smoke cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_SCRIPT = REPO_ROOT / "scripts" / "plugin_gguf_smoke.py"


@dataclass
class SmokeCase:
    name: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int | None = None
    skip_if_missing: list[str] = field(default_factory=list)


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _load_cases(config_path: Path) -> list[SmokeCase]:
    with config_path.open() as f:
        data = json.load(f)
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{config_path} must contain a 'cases' list")

    loaded: list[SmokeCase] = []
    for idx, case_data in enumerate(cases):
        if not isinstance(case_data, dict):
            raise ValueError(f"case #{idx} must be an object")
        name = case_data.get("name")
        args = case_data.get("args")
        if not isinstance(name, str) or not name:
            raise ValueError(f"case #{idx} must have a non-empty string name")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"case {name!r} must have a string args list")
        env = case_data.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError(f"case {name!r} env must be a string map")
        timeout_seconds = case_data.get("timeoutSeconds")
        if timeout_seconds is not None and not isinstance(timeout_seconds, int):
            raise ValueError(f"case {name!r} timeoutSeconds must be an integer")
        skip_if_missing = case_data.get("skipIfMissing", [])
        if not isinstance(skip_if_missing, list) or not all(
            isinstance(path, str) for path in skip_if_missing
        ):
            raise ValueError(f"case {name!r} skipIfMissing must be a string list")
        loaded.append(
            SmokeCase(
                name=name,
                args=[_expand(arg) for arg in args],
                env={key: _expand(value) for key, value in env.items()},
                timeout_seconds=timeout_seconds,
                skip_if_missing=[_expand(path) for path in skip_if_missing],
            )
        )
    return loaded


def _gpt_oss_mxfp4_case(model_path: str) -> SmokeCase:
    return SmokeCase(
        name="gpt_oss_mxfp4_marlin",
        args=[
            "--override-in-tree",
            "--expect",
            "active",
            "--generate",
            "--model",
            model_path,
            "--tokenizer",
            "openai/gpt-oss-20b",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "64",
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.20",
            "--moe-backend",
            "marlin",
            "--max-tokens",
            "4",
            "--min-generated-tokens",
            "4",
            "--disable-v1-multiprocessing",
            "--enforce-eager",
            "--json",
        ],
        env={"VLLM_LOGGING_LEVEL": "INFO"},
        timeout_seconds=900,
        skip_if_missing=[model_path],
    )


def _registration_case() -> SmokeCase:
    return SmokeCase(name="registration", args=["--json"])


def _parse_smoke_result(output: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    result: dict[str, Any] | None = None
    for idx, char in enumerate(output):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(output[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "ok" in candidate and "actual" in candidate:
            result = candidate
    return result


def _missing_paths(case: SmokeCase) -> list[str]:
    return [path for path in case.skip_if_missing if not Path(path).exists()]


def run_case(
    case: SmokeCase,
    smoke_script: Path,
    default_timeout_seconds: int | None,
) -> dict[str, Any]:
    missing_paths = _missing_paths(case)
    if missing_paths:
        return {
            "name": case.name,
            "ok": True,
            "skipped": True,
            "missing": missing_paths,
            "returncode": None,
            "smoke": None,
            "outputTail": "",
        }

    env = os.environ.copy()
    env.update(case.env)
    timeout_seconds = case.timeout_seconds or default_timeout_seconds
    command = [sys.executable, str(smoke_script), *case.args]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout
        smoke = _parse_smoke_result(output)
        ok = completed.returncode == 0 and bool(smoke and smoke.get("ok"))
        return {
            "name": case.name,
            "ok": ok,
            "skipped": False,
            "returncode": completed.returncode,
            "smoke": smoke,
            "outputTail": output[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {
            "name": case.name,
            "ok": False,
            "skipped": False,
            "returncode": None,
            "timeoutSeconds": timeout_seconds,
            "smoke": _parse_smoke_result(output),
            "outputTail": output[-8000:],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--case", action="append", dest="case_names")
    parser.add_argument("--smoke-script", type=Path, default=DEFAULT_SMOKE_SCRIPT)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait before each case so prior GPU memory releases settle.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--gpt-oss-mxfp4-model")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _select_cases(cases: list[SmokeCase], names: list[str] | None) -> list[SmokeCase]:
    if not names:
        return cases
    selected = [case for case in cases if case.name in set(names)]
    missing = sorted(set(names) - {case.name for case in selected})
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(missing)}")
    return selected


def main() -> int:
    args = parse_args()
    cases = _load_cases(args.config) if args.config else [_registration_case()]
    if args.gpt_oss_mxfp4_model:
        cases.append(_gpt_oss_mxfp4_case(_expand(args.gpt_oss_mxfp4_model)))
    cases = _select_cases(cases, args.case_names)

    results: list[dict[str, Any]] = []
    for case in cases:
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)
        result = run_case(case, args.smoke_script, args.timeout_seconds)
        results.append(result)
        if args.fail_fast and not result["ok"]:
            break

    ok = all(result["ok"] for result in results)
    payload = {"ok": ok, "cases": results}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            status = "SKIP" if result["skipped"] else "PASS" if result["ok"] else "FAIL"
            print(f"{status} {result['name']}")
        print(f"matrix ok={str(ok).lower()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
