# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path


def _load_matrix_module():
    matrix_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "plugin_gguf_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("plugin_gguf_matrix", matrix_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_loads_config_with_expansion(tmp_path, monkeypatch):
    matrix = _load_matrix_module()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    config = tmp_path / "matrix.json"
    config.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "tiny",
                        "args": ["--model", "$MODEL_DIR/model.gguf"],
                        "env": {"VLLM_LOGGING_LEVEL": "INFO"},
                        "timeoutSeconds": 7,
                        "skipIfMissing": ["$MODEL_DIR/model.gguf"],
                    }
                ]
            }
        )
    )

    cases = matrix._load_cases(config)

    assert len(cases) == 1
    assert cases[0].name == "tiny"
    assert cases[0].args == ["--model", str(tmp_path / "model.gguf")]
    assert cases[0].env == {"VLLM_LOGGING_LEVEL": "INFO"}
    assert cases[0].timeout_seconds == 7
    assert cases[0].skip_if_missing == [str(tmp_path / "model.gguf")]


def test_matrix_parses_smoke_json_amid_logs():
    matrix = _load_matrix_module()
    output = """
INFO loading model
{"not": "the smoke payload"}
{
  "actual": "active",
  "ok": true,
  "generation": {"tokens": 4}
}
[rank0] cleanup warning
"""

    result = matrix._parse_smoke_result(output)

    assert result == {
        "actual": "active",
        "ok": True,
        "generation": {"tokens": 4},
    }


def test_matrix_skips_case_when_artifact_missing(tmp_path):
    matrix = _load_matrix_module()
    case = matrix.SmokeCase(
        name="missing",
        args=["--json"],
        skip_if_missing=[str(tmp_path / "missing.gguf")],
    )

    result = matrix.run_case(case, tmp_path / "unused.py", default_timeout_seconds=1)

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["missing"] == [str(tmp_path / "missing.gguf")]


def test_matrix_run_case_parses_successful_smoke_subprocess(tmp_path):
    matrix = _load_matrix_module()
    smoke_script = tmp_path / "smoke.py"
    smoke_script.write_text(
        "\n".join(
            [
                "print('INFO before json')",
                'print(\'{"actual": "active", "ok": true}\')',
                "print('INFO after json')",
            ]
        )
    )
    case = matrix.SmokeCase(name="ok", args=["--ignored"])

    result = matrix.run_case(case, smoke_script, default_timeout_seconds=5)

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["returncode"] == 0
    assert result["smoke"]["ok"] is True


def test_matrix_run_case_fails_on_nonzero_returncode(tmp_path):
    matrix = _load_matrix_module()
    smoke_script = tmp_path / "smoke.py"
    smoke_script.write_text(
        "\n".join(
            [
                'print(\'{"actual": "active", "ok": true}\')',
                "raise SystemExit(2)",
            ]
        )
    )
    case = matrix.SmokeCase(name="bad", args=[])

    result = matrix.run_case(case, smoke_script, default_timeout_seconds=5)

    assert result["ok"] is False
    assert result["returncode"] == 2
    assert result["smoke"]["ok"] is True


def test_matrix_gpt_oss_case_uses_reproducible_runtime_shape():
    matrix = _load_matrix_module()

    case = matrix._gpt_oss_mxfp4_case("/models/gpt-oss.gguf")

    assert case.name == "gpt_oss_mxfp4_marlin"
    assert "--moe-backend" in case.args
    assert case.args[case.args.index("--moe-backend") + 1] == "marlin"
    assert "--max-num-seqs" in case.args
    assert case.args[case.args.index("--max-num-seqs") + 1] == "1"
    assert case.skip_if_missing == ["/models/gpt-oss.gguf"]


def test_matrix_main_settles_before_each_case(monkeypatch, capsys):
    matrix = _load_matrix_module()
    sleeps = []
    calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plugin_gguf_matrix.py",
            "--settle-seconds",
            "1.5",
            "--json",
        ],
    )
    monkeypatch.setattr(matrix.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        matrix,
        "_registration_case",
        lambda: matrix.SmokeCase(name="registration", args=["--json"]),
    )

    def fake_run_case(case, smoke_script, default_timeout_seconds):
        calls.append((case.name, smoke_script, default_timeout_seconds))
        return {
            "name": case.name,
            "ok": True,
            "skipped": False,
            "returncode": 0,
            "smoke": {"ok": True},
            "outputTail": "",
        }

    monkeypatch.setattr(matrix, "run_case", fake_run_case)

    assert matrix.main() == 0

    assert sleeps == [1.5]
    assert calls[0][0] == "registration"
    assert json.loads(capsys.readouterr().out)["ok"] is True
