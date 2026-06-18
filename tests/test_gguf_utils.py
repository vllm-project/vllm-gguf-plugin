# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from unittest.mock import patch

import pytest

import vllm_gguf_plugin.gguf_utils as gguf_utils_module
from vllm_gguf_plugin.gguf_utils import (
    detect_gguf_multimodal,
    extract_lm_head_from_gguf,
    get_gguf_file_path_from_hf,
    is_gguf,
    is_local_gguf_quant,
    is_remote_gguf,
    resolve_gguf_config_source,
    split_remote_gguf,
)


class TestIsRemoteGGUF:
    """Test is_remote_gguf utility function."""

    def test_is_remote_gguf_with_colon_and_slash(self):
        """Test is_remote_gguf with repo_id:quant_type format."""
        assert is_remote_gguf("unsloth/Qwen3-0.6B-GGUF:IQ1_S")
        assert is_remote_gguf("user/repo:Q2_K")
        assert is_remote_gguf("repo/model:Q4_K")
        assert is_remote_gguf("repo/model:Q8_0")

        assert not is_remote_gguf("repo/model:quant")
        assert not is_remote_gguf("repo/model:INVALID")
        assert not is_remote_gguf("repo/model:invalid_type")

    def test_is_remote_gguf_extended_quant_types(self):
        """Test is_remote_gguf with extended quant type naming conventions."""
        assert is_remote_gguf("repo/model:Q4_K_M")
        assert is_remote_gguf("repo/model:Q4_K_S")
        assert is_remote_gguf("repo/model:Q3_K_L")
        assert is_remote_gguf("repo/model:Q5_K_M")
        assert is_remote_gguf("repo/model:Q3_K_S")

        assert is_remote_gguf("repo/model:Q5_K_XL")
        assert is_remote_gguf("repo/model:IQ4_XS")
        assert is_remote_gguf("repo/model:IQ3_XXS")

        assert not is_remote_gguf("repo/model:INVALID_M")
        assert not is_remote_gguf("repo/model:Q9_K_M")

    def test_is_remote_gguf_file_type_only_quants(self):
        """Test is_remote_gguf with file-type-only quants (LlamaFileType).

        IQ2_M / IQ3_M / IQ3_XS / MXFP4_MOE exist only as GGUF file types
        (LlamaFileType), not as GGML tensor types. Regression test for
        https://github.com/vllm-project/vllm/issues/42734.
        """
        assert is_remote_gguf("unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ2_M")
        assert is_remote_gguf("repo/model:IQ2_M")
        assert is_remote_gguf("repo/model:IQ3_M")
        assert is_remote_gguf("repo/model:IQ3_XS")
        assert is_remote_gguf("repo/model:MXFP4_MOE")
        assert is_remote_gguf("user/Model-GGUF:UD-IQ3_XS")

        assert not is_remote_gguf("repo/model:IQ9_M")
        assert not is_remote_gguf("repo/model:NOTATYPE")

    def test_is_remote_gguf_nonstandard_quant_type(self):
        """Test is_remote_gguf with non-standard quant types containing
        a known GGML type."""
        assert is_remote_gguf("unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL")
        assert is_remote_gguf("user/Model:UD-Q4_K_M")
        assert is_remote_gguf("user/SomeModel:Custom-Q8_0")

        assert is_remote_gguf("user/Model-GGUF:UD-IQ4_NL")
        assert is_remote_gguf("user/Model-GGUF:UD-Q8_0")

        assert not is_remote_gguf("repo/model:TOTALLY-RANDOM")
        assert not is_remote_gguf("user/Model:UD-INVALID")
        assert not is_remote_gguf("repo/model:UDIQ4NL")

    def test_is_remote_gguf_without_colon(self):
        """Test is_remote_gguf without colon."""
        assert not is_remote_gguf("repo/model")
        assert not is_remote_gguf("unsloth/Qwen3-0.6B-GGUF")

    def test_is_remote_gguf_without_slash(self):
        """Test is_remote_gguf without slash."""
        assert not is_remote_gguf("model.gguf")
        assert not is_remote_gguf("model:IQ1_S")
        assert not is_remote_gguf("model:quant")

    def test_is_remote_gguf_local_path(self):
        """Test is_remote_gguf with local file path."""
        assert not is_remote_gguf("/path/to/model.gguf")
        assert not is_remote_gguf("./model.gguf")

    def test_is_remote_gguf_with_path_object(self):
        """Test is_remote_gguf with Path object."""
        assert is_remote_gguf(Path("unsloth/Qwen3-0.6B-GGUF:IQ1_S"))
        assert not is_remote_gguf(Path("repo/model"))

    def test_is_remote_gguf_with_http_https(self):
        """Test is_remote_gguf with HTTP/HTTPS URLs."""
        assert not is_remote_gguf("http://example.com/repo/model:IQ1_S")
        assert not is_remote_gguf("https://huggingface.co/repo/model:Q2_K")
        assert not is_remote_gguf("http://repo/model:Q4_K")
        assert not is_remote_gguf("https://repo/model:Q8_0")

    def test_is_remote_gguf_with_cloud_storage(self):
        """Test is_remote_gguf with cloud storage paths."""
        assert not is_remote_gguf("s3://bucket/repo/model:IQ1_S")
        assert not is_remote_gguf("gs://bucket/repo/model:Q2_K")
        assert not is_remote_gguf("s3://repo/model:Q4_K")
        assert not is_remote_gguf("gs://repo/model:Q8_0")


class TestIsLocalGGUFQuant:
    """Test is_local_gguf_quant utility function."""

    @patch("vllm_gguf_plugin.gguf_utils.Path")
    def test_is_local_gguf_quant_valid(self, mock_path_cls):
        """Test with valid local dir:quant_type."""
        mock_path_inst = mock_path_cls.return_value
        mock_path_inst.is_dir.return_value = True
        assert is_local_gguf_quant("/some/dir:Q8_0")
        assert is_local_gguf_quant("/mnt/data/model-gguf:Q4_K_M")

    def test_is_local_gguf_quant_no_colon(self):
        """Test without colon."""
        assert not is_local_gguf_quant("/some/dir")
        assert not is_local_gguf_quant("model.gguf")

    def test_is_local_gguf_quant_invalid_quant(self):
        """Test with invalid quant type."""
        assert not is_local_gguf_quant("/some/dir:INVALID")
        assert not is_local_gguf_quant("/some/dir:random_type")

    @patch("vllm_gguf_plugin.gguf_utils.Path")
    def test_is_local_gguf_quant_not_dir(self, mock_path_cls):
        """Test with non-directory path."""
        mock_path_inst = mock_path_cls.return_value
        mock_path_inst.is_dir.return_value = False
        assert not is_local_gguf_quant("/some/file.txt:Q8_0")


class TestSplitRemoteGGUF:
    """Test split_remote_gguf utility function."""

    def test_split_remote_gguf_valid(self):
        """Test split_remote_gguf with valid repo_id:quant_type format."""
        repo_id, quant_type = split_remote_gguf("unsloth/Qwen3-0.6B-GGUF:IQ1_S")
        assert repo_id == "unsloth/Qwen3-0.6B-GGUF"
        assert quant_type == "IQ1_S"

        repo_id, quant_type = split_remote_gguf("repo/model:Q2_K")
        assert repo_id == "repo/model"
        assert quant_type == "Q2_K"

    def test_split_remote_gguf_extended_quant_types(self):
        """Test split_remote_gguf with extended quant type naming conventions."""
        repo_id, quant_type = split_remote_gguf("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
        assert repo_id == "unsloth/Qwen3-0.6B-GGUF"
        assert quant_type == "Q4_K_M"

        repo_id, quant_type = split_remote_gguf("repo/model:Q3_K_S")
        assert repo_id == "repo/model"
        assert quant_type == "Q3_K_S"

    def test_split_remote_gguf_nonstandard_quant_type(self):
        """Test split_remote_gguf with non-standard quant types in GGUF repos."""
        repo_id, quant_type = split_remote_gguf(
            "unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL"
        )
        assert repo_id == "unsloth/Qwen3.5-35B-A3B-GGUF"
        assert quant_type == "UD-Q4_K_XL"

    def test_split_remote_gguf_with_path_object(self):
        """Test split_remote_gguf with Path object."""
        repo_id, quant_type = split_remote_gguf(Path("unsloth/Qwen3-0.6B-GGUF:IQ1_S"))
        assert repo_id == "unsloth/Qwen3-0.6B-GGUF"
        assert quant_type == "IQ1_S"

    def test_split_remote_gguf_invalid(self):
        """Test split_remote_gguf with invalid format."""
        with pytest.raises(ValueError, match="Wrong GGUF model"):
            split_remote_gguf("repo/model")

        with pytest.raises(ValueError, match="Wrong GGUF model"):
            split_remote_gguf("repo/model:INVALID_TYPE")

        with pytest.raises(ValueError, match="Wrong GGUF model"):
            split_remote_gguf("http://repo/model:IQ1_S")

        with pytest.raises(ValueError, match="Wrong GGUF model"):
            split_remote_gguf("s3://bucket/repo/model:Q2_K")


class TestIsGGUF:
    """Test is_gguf utility function."""

    @patch("vllm_gguf_plugin.gguf_utils.check_gguf_file", return_value=True)
    def test_is_gguf_with_local_file(self, mock_check_gguf):
        """Test is_gguf with local GGUF file."""
        assert is_gguf("/path/to/model.gguf")
        assert is_gguf("./model.gguf")

    def test_is_gguf_with_remote_gguf(self):
        """Test is_gguf with remote GGUF format."""
        assert is_gguf("unsloth/Qwen3-0.6B-GGUF:IQ1_S")
        assert is_gguf("repo/model:Q2_K")
        assert is_gguf("repo/model:Q4_K")

        assert is_gguf("repo/model:Q4_K_M")
        assert is_gguf("repo/model:Q3_K_S")
        assert is_gguf("repo/model:Q5_K_L")

        assert not is_gguf("repo/model:quant")
        assert not is_gguf("repo/model:INVALID")

    def test_is_gguf_with_exact_remote_file(self):
        """Test is_gguf with exact remote GGUF file references."""
        assert is_gguf("org/repo/model.gguf")
        assert is_gguf("org/repo/subdir/model.gguf")

    @patch("vllm_gguf_plugin.gguf_utils.check_gguf_file", return_value=False)
    def test_is_gguf_false(self, mock_check_gguf):
        """Test is_gguf returns False for non-GGUF models."""
        assert not is_gguf("unsloth/Qwen3-0.6B")
        assert not is_gguf("repo/model")
        assert not is_gguf("model")

    def test_is_gguf_edge_cases(self):
        """Test is_gguf with edge cases."""
        assert not is_gguf("")
        assert not is_gguf("model:IQ1_S")
        assert not is_gguf("repo/model")
        assert not is_gguf("http://repo/model:IQ1_S")
        assert not is_gguf("https://repo/model:Q2_K")
        assert not is_gguf("s3://bucket/repo/model:IQ1_S")
        assert not is_gguf("gs://bucket/repo/model:Q2_K")


class TestDetectGGUFMultimodal:
    def test_prefers_quant_matched_mmproj(self, tmp_path):
        model = tmp_path / "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
        model.touch()
        (tmp_path / "mmproj-BF16.gguf").touch()
        (tmp_path / "mmproj-F16.gguf").touch()
        quant_mmproj = tmp_path / "mmproj-Q4_K_XL.gguf"
        quant_mmproj.touch()

        assert detect_gguf_multimodal(str(model)) == quant_mmproj

    def test_prefers_quant_matched_mmproj_for_dot_style_model(self, tmp_path):
        model = tmp_path / "model.q4_k_m.gguf"
        model.touch()
        (tmp_path / "mmproj-F16.gguf").touch()
        quant_mmproj = tmp_path / "mmproj-Q4_K_M.gguf"
        quant_mmproj.touch()

        assert detect_gguf_multimodal(str(model)) == quant_mmproj

    def test_finds_hf_snapshot_root_mmproj_for_subdir_model(self, tmp_path):
        snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
        model_dir = snapshot / "Q4_K_M"
        model_dir.mkdir(parents=True)
        model = model_dir / "Qwen-Q4_K_M.gguf"
        model.touch()
        mmproj = snapshot / "mmproj-BF16.gguf"
        mmproj.touch()

        assert detect_gguf_multimodal(str(model)) == mmproj

    def test_finds_intermediate_snapshot_ancestor_mmproj(self, tmp_path):
        snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
        model_dir = snapshot / "Q4_K_M" / "nested" / "deep"
        model_dir.mkdir(parents=True)
        model = model_dir / "Qwen-Q4_K_M.gguf"
        model.touch()
        snapshot_mmproj = snapshot / "mmproj-F16.gguf"
        ancestor_mmproj = snapshot / "Q4_K_M" / "mmproj-F16.gguf"
        snapshot_mmproj.touch()
        ancestor_mmproj.touch()

        assert detect_gguf_multimodal(str(model)) == ancestor_mmproj

    def test_prefers_nearest_mmproj_when_rank_ties(self, tmp_path):
        model_dir = tmp_path / "nested"
        model_dir.mkdir()
        model = model_dir / "model-Q4_K_M.gguf"
        model.touch()
        parent_mmproj = tmp_path / "mmproj-F16.gguf"
        local_mmproj = model_dir / "mmproj-F16.gguf"
        parent_mmproj.touch()
        local_mmproj.touch()

        assert detect_gguf_multimodal(str(model)) == local_mmproj


class TestResolveGGUFConfigSource:
    def test_local_file_uses_hf_snapshot_root_config(self, tmp_path, monkeypatch):
        snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
        model_dir = snapshot / "Q8_0" / "nested"
        model_dir.mkdir(parents=True)
        model = model_dir / "model.gguf"
        model.touch()
        calls = []

        def fake_file_or_path_exists(source, filename, revision):
            calls.append((Path(source), filename, revision))
            return Path(source) == snapshot and filename == "config.json"

        def fail_base_model_lookup(model_path):
            raise AssertionError(
                "base model lookup should not run when snapshot root has config"
            )

        monkeypatch.setattr(
            gguf_utils_module,
            "_get_local_gguf_base_model_ids",
            fail_base_model_lookup,
        )
        monkeypatch.setattr(
            gguf_utils_module,
            "file_or_path_exists",
            fake_file_or_path_exists,
        )

        assert resolve_gguf_config_source(model) == snapshot
        assert calls == [
            (model_dir, "config.json", None),
            (snapshot / "Q8_0", "config.json", None),
            (snapshot, "config.json", None),
        ]

    def test_local_file_uses_intermediate_snapshot_config(self, tmp_path, monkeypatch):
        snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
        config_dir = snapshot / "Q8_0"
        model_dir = config_dir / "nested" / "deep"
        model_dir.mkdir(parents=True)
        model = model_dir / "model.gguf"
        model.touch()
        calls = []

        def fake_file_or_path_exists(source, filename, revision):
            calls.append((Path(source), filename, revision))
            return Path(source) == config_dir and filename == "config.json"

        def fail_base_model_lookup(model_path):
            raise AssertionError(
                "base model lookup should not run when ancestor has config"
            )

        monkeypatch.setattr(
            gguf_utils_module,
            "_get_local_gguf_base_model_ids",
            fail_base_model_lookup,
        )
        monkeypatch.setattr(
            gguf_utils_module,
            "file_or_path_exists",
            fake_file_or_path_exists,
        )

        assert resolve_gguf_config_source(model) == config_dir
        assert calls == [
            (model_dir, "config.json", None),
            (config_dir / "nested", "config.json", None),
            (config_dir, "config.json", None),
        ]

    def test_exact_remote_file_uses_repo_config(self, monkeypatch):
        calls = []

        def fake_file_or_path_exists(model, filename, revision):
            calls.append((model, filename, revision))
            return model == "org/repo" and filename == "config.json"

        def fail_base_model_lookup(repo_id, revision=None):
            raise AssertionError(
                "base model lookup should not run when repo has config"
            )

        monkeypatch.setattr(
            gguf_utils_module,
            "_get_remote_gguf_base_model_ids",
            fail_base_model_lookup,
        )
        monkeypatch.setattr(
            gguf_utils_module,
            "file_or_path_exists",
            fake_file_or_path_exists,
        )

        assert (
            resolve_gguf_config_source(
                "org/repo/subdir/model.gguf",
                revision="main",
            )
            == "org/repo"
        )
        assert calls == [("org/repo", "config.json", "main")]

    def test_exact_remote_file_can_redirect_to_base_model(self, monkeypatch):
        def fake_file_or_path_exists(model, filename, revision):
            return model == "base/model" and filename == "config.json"

        monkeypatch.setattr(
            gguf_utils_module,
            "_get_remote_gguf_base_model_ids",
            lambda repo_id, revision=None: ("base/model",),
        )
        monkeypatch.setattr(
            gguf_utils_module,
            "file_or_path_exists",
            fake_file_or_path_exists,
        )

        assert (
            resolve_gguf_config_source(
                "org/repo/subdir/model.gguf",
                revision="main",
            )
            == "base/model"
        )


class TestGetGGUFFilePathFromHF:
    def test_uses_download_quant_patterns(self, monkeypatch):
        calls = {}

        def fake_list_filtered_repo_files(repo_id, allow_patterns, revision):
            calls["repo_id"] = repo_id
            calls["allow_patterns"] = allow_patterns
            calls["revision"] = revision
            return ["Q4_K_M/model.q4_k_m.gguf"]

        monkeypatch.setattr(
            gguf_utils_module,
            "list_filtered_repo_files",
            fake_list_filtered_repo_files,
        )

        assert (
            get_gguf_file_path_from_hf(
                "org/repo",
                "Q4_K_M",
                revision="abc123",
            )
            == "Q4_K_M/model.q4_k_m.gguf"
        )
        assert calls["repo_id"] == "org/repo"
        assert calls["revision"] == "abc123"
        assert "*.Q4_K_M.gguf" in calls["allow_patterns"]
        assert "*/*.Q4_K_M.gguf" in calls["allow_patterns"]
        assert "*.q4_k_m.gguf" in calls["allow_patterns"]
        assert "*/*.q4_k_m.gguf" in calls["allow_patterns"]

    def test_uses_download_candidate_order(self, monkeypatch):
        monkeypatch.setattr(
            gguf_utils_module,
            "list_filtered_repo_files",
            lambda repo_id, allow_patterns, revision: [
                "model-Q4_K_M-00002-of-00002.gguf",
                "model-Q4_K_M-00001-of-00002.gguf",
            ],
        )

        assert (
            get_gguf_file_path_from_hf("org/repo", "Q4_K_M")
            == "model-Q4_K_M-00001-of-00002.gguf"
        )

    def test_ignores_mmproj_candidates(self, monkeypatch):
        monkeypatch.setattr(
            gguf_utils_module,
            "list_filtered_repo_files",
            lambda repo_id, allow_patterns, revision: [
                "mmproj-Q4_K_M.gguf",
                "model-Q4_K_M.gguf",
            ],
        )

        assert get_gguf_file_path_from_hf("org/repo", "Q4_K_M") == ("model-Q4_K_M.gguf")

    def test_rejects_mmproj_only_candidates(self, monkeypatch):
        monkeypatch.setattr(
            gguf_utils_module,
            "list_filtered_repo_files",
            lambda repo_id, allow_patterns, revision: ["mmproj-Q4_K_M.gguf"],
        )

        with pytest.raises(ValueError, match="model GGUF file"):
            get_gguf_file_path_from_hf("org/repo", "Q4_K_M")


class TestExtractLMHeadFromGGUF:
    @patch("vllm_gguf_plugin.gguf_utils.check_gguf_file", return_value=True)
    @patch("vllm_gguf_plugin.gguf_utils.gguf.GGUFReader")
    def test_matches_only_exact_output_weight(self, mock_reader_cls, _mock_check):
        mock_reader_cls.return_value.tensors = [
            type("Tensor", (), {"name": "blk.0.attn_output.weight"})(),
            type("Tensor", (), {"name": "output_norm.weight"})(),
        ]

        assert not extract_lm_head_from_gguf("/tmp/model.gguf")

    @patch("vllm_gguf_plugin.gguf_utils.check_gguf_file", return_value=True)
    @patch("vllm_gguf_plugin.gguf_utils.gguf.GGUFReader")
    def test_detects_exact_output_weight(self, mock_reader_cls, _mock_check):
        mock_reader_cls.return_value.tensors = [
            type("Tensor", (), {"name": "blk.0.attn_output.weight"})(),
            type("Tensor", (), {"name": "output.weight"})(),
        ]

        assert extract_lm_head_from_gguf("/tmp/model.gguf")
