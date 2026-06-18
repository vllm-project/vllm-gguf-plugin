# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
from vllm.config.load import LoadConfig

from vllm_gguf_plugin.loader import GGUFModelLoader
from vllm_gguf_plugin.weight_utils import download_gguf, resolve_gguf_file_set


class TestSplitGGUFResolution:
    """Test split GGUF shard discovery and validation."""

    def test_non_split_file_returns_single_path(self):
        assert resolve_gguf_file_set("/models/model-Q4_K_M.gguf") == [
            "/models/model-Q4_K_M.gguf"
        ]

    def test_split_file_resolves_all_shards_from_any_entry_shard(self, tmp_path):
        for idx in range(1, 4):
            (tmp_path / f"model-Q4_K_M-0000{idx}-of-00003.gguf").touch()

        result = resolve_gguf_file_set(
            tmp_path / "model-Q4_K_M-00002-of-00003.gguf"
        )

        assert result == [
            str(tmp_path / "model-Q4_K_M-00001-of-00003.gguf"),
            str(tmp_path / "model-Q4_K_M-00002-of-00003.gguf"),
            str(tmp_path / "model-Q4_K_M-00003-of-00003.gguf"),
        ]

    def test_split_file_missing_shard_fails_closed(self, tmp_path):
        (tmp_path / "model-Q4_K_M-00001-of-00003.gguf").touch()
        (tmp_path / "model-Q4_K_M-00003-of-00003.gguf").touch()

        with pytest.raises(ValueError, match="Incomplete split GGUF model"):
            resolve_gguf_file_set(tmp_path / "model-Q4_K_M-00001-of-00003.gguf")


class TestGGUFDownload:
    """Test GGUF model downloading functionality."""

    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_single_file(self, mock_download):
        """Test downloading a single GGUF file."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [f"{mock_folder}/model-IQ1_S.gguf"] if "IQ1_S" in pattern else []
            )

            result = download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")

            mock_download.assert_called_once_with(
                repo_id="unsloth/Qwen3-0.6B-GGUF",
                cache_dir=None,
                allow_patterns=[
                    "*.IQ1_S-*.gguf",
                    "*/*.IQ1_S-*.gguf",
                    "*.IQ1_S.gguf",
                    "*/*.IQ1_S.gguf",
                    "*-IQ1_S-*.gguf",
                    "*/*-IQ1_S-*.gguf",
                    "*-IQ1_S.gguf",
                    "*/*-IQ1_S.gguf",
                    "*.iq1_s-*.gguf",
                    "*/*.iq1_s-*.gguf",
                    "*.iq1_s.gguf",
                    "*/*.iq1_s.gguf",
                    "*-iq1_s-*.gguf",
                    "*/*-iq1_s-*.gguf",
                    "*-iq1_s.gguf",
                    "*/*-iq1_s.gguf",
                ],
                revision=None,
                ignore_patterns=None,
            )

            assert result == f"{mock_folder}/model-IQ1_S.gguf"

    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_sharded_files(self, mock_download, tmp_path):
        """Test downloading sharded GGUF files."""
        mock_folder = str(tmp_path)
        mock_download.return_value = mock_folder
        (tmp_path / "model-Q2_K-00001-of-00002.gguf").touch()
        (tmp_path / "model-Q2_K-00002-of-00002.gguf").touch()

        result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

        assert result == f"{mock_folder}/model-Q2_K-00001-of-00002.gguf"

    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_subdir(self, mock_download, tmp_path):
        """Test downloading GGUF files from subdirectory."""
        mock_folder = str(tmp_path)
        mock_download.return_value = mock_folder
        (tmp_path / "Q2_K").mkdir()
        (tmp_path / "Q2_K" / "model-Q2_K.gguf").touch()

        result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

        assert result == f"{mock_folder}/Q2_K/model-Q2_K.gguf"

    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    @patch("glob.glob", return_value=[])
    def test_download_gguf_no_files_found(self, mock_glob, mock_download):
        """Test error when no GGUF files are found."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with pytest.raises(ValueError, match="Downloaded GGUF files not found"):
            download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")


class TestGGUFModelLoader:
    """Test GGUFModelLoader class methods."""

    @patch("os.path.isfile", return_value=True)
    def test_prepare_weights_local_file(self, mock_isfile):
        """Test _prepare_weights with local file."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        model_config = MagicMock()
        model_config.model_weights = "/path/to/model.gguf"
        model_config.model = "/path/to/hf"

        result = loader._prepare_weights(model_config)
        assert result == "/path/to/model.gguf"
        mock_isfile.assert_called_once_with("/path/to/model.gguf")

    @patch("vllm_gguf_plugin.loader.hf_hub_download")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_filename(self, mock_isfile, mock_hf_download):
        """Test _prepare_weights with repo_id/filename.gguf format."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_hf_download.return_value = "/downloaded/model.gguf"

        model_config = MagicMock()
        model_config.model_weights = "unsloth/Qwen3-0.6B-GGUF/model.gguf"
        model_config.model = "unsloth/Qwen3-0.6B-GGUF"

        result = loader._prepare_weights(model_config)
        assert result == "/downloaded/model.gguf"
        mock_hf_download.assert_called_once_with(
            repo_id="unsloth/Qwen3-0.6B-GGUF", filename="model.gguf"
        )

    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    @patch("glob.glob")
    @patch("os.path.isdir", return_value=False)
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_remote_repo_quant_type(
        self, mock_isfile, mock_isdir, mock_glob, mock_download
    ):
        """Test _prepare_weights with remote repo_id:quant_type format."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder
        mock_glob.side_effect = lambda pattern, **kwargs: (
            [f"{mock_folder}/model-IQ1_S.gguf"] if "IQ1_S" in pattern else []
        )

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        model_config = MagicMock()
        model_config.model_weights = "unsloth/Qwen3-0.6B-GGUF:IQ1_S"
        model_config.model = "unsloth/Qwen3-0.6B-GGUF"
        model_config.revision = None

        result = loader._prepare_weights(model_config)
        assert result == f"{mock_folder}/model-IQ1_S.gguf"
        mock_download.assert_called_once()

    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_invalid_format(self, mock_isfile):
        """Test _prepare_weights with invalid format."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        model_config = MagicMock()
        model_config.model_weights = "invalid-format"
        model_config.model = "invalid-format"

        with pytest.raises(ValueError, match="Unrecognised GGUF reference"):
            loader._prepare_weights(model_config)
