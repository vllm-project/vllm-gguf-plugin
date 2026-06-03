# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_gguf_plugin.gguf_utils import (
    extract_lm_head_from_gguf,
    is_gguf,
    is_local_gguf_quant,
    is_remote_gguf,
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
