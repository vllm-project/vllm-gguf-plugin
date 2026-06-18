# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
from vllm.config.load import LoadConfig

from vllm_gguf_plugin.loader import GGUFModelLoader
from vllm_gguf_plugin.weight_utils import (
    download_gguf,
    download_gguf_file,
    resolve_gguf_file_set,
    resolve_local_gguf,
    split_remote_gguf_file_ref,
)


class TestSplitGGUFResolution:
    """Test split GGUF shard discovery and validation."""

    def test_non_split_file_returns_single_path(self):
        assert resolve_gguf_file_set("/models/model-Q4_K_M.gguf") == [
            "/models/model-Q4_K_M.gguf"
        ]

    def test_split_file_resolves_all_shards_from_any_entry_shard(self, tmp_path):
        for idx in range(1, 4):
            (tmp_path / f"model-Q4_K_M-0000{idx}-of-00003.gguf").touch()

        result = resolve_gguf_file_set(tmp_path / "model-Q4_K_M-00002-of-00003.gguf")

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


class TestRemoteGGUFFileRefs:
    def test_split_remote_gguf_file_ref_root_file(self):
        assert split_remote_gguf_file_ref("unsloth/Qwen-GGUF/model.gguf") == (
            "unsloth/Qwen-GGUF",
            "model.gguf",
        )

    def test_split_remote_gguf_file_ref_subdir_file(self):
        assert split_remote_gguf_file_ref("org/repo/Q4_K_M/model.gguf") == (
            "org/repo",
            "Q4_K_M/model.gguf",
        )

    def test_split_remote_gguf_file_ref_rejects_local_or_unsafe_paths(self):
        assert split_remote_gguf_file_ref("/tmp/model.gguf") is None
        assert split_remote_gguf_file_ref("org/repo/../model.gguf") is None
        assert split_remote_gguf_file_ref("repo/model.gguf") is None


class TestGGUFDownload:
    """Test GGUF model downloading functionality."""

    @patch("vllm_gguf_plugin.weight_utils.list_repo_files", return_value=[])
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_single_file(self, mock_download, mock_list_repo_files):
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
            mock_list_repo_files.assert_called_once_with(
                "unsloth/Qwen3-0.6B-GGUF",
                revision=None,
            )

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "Qwen3.5-0.8B-Q4_K_M.gguf",
            "mmproj-BF16.gguf",
            "mmproj-F16.gguf",
            "mmproj-F32.gguf",
            "processor_config.json",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_downloads_preferred_f16_mmproj(
        self,
        mock_download,
        mock_list_repo_files,
        tmp_path,
    ):
        mock_download.return_value = str(tmp_path)
        (tmp_path / "Qwen3.5-0.8B-Q4_K_M.gguf").touch()
        (tmp_path / "mmproj-F16.gguf").touch()
        (tmp_path / "processor_config.json").touch()

        result = download_gguf("unsloth/Qwen3.5-0.8B-GGUF", "Q4_K_M")

        assert result == str(tmp_path / "Qwen3.5-0.8B-Q4_K_M.gguf")
        assert mock_download.call_args.kwargs["allow_patterns"] == [
            "Qwen3.5-0.8B-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "processor_config.json",
        ]
        mock_list_repo_files.assert_called_once_with(
            "unsloth/Qwen3.5-0.8B-GGUF",
            revision=None,
        )

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "gemma-4-E2B-it-Q8_0.gguf",
            "mmproj-gemma-4-E2B-it-Q8_0.gguf",
            "mmproj-gemma-4-E2B-it-bf16.gguf",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_prefers_quant_matched_mmproj(
        self,
        mock_download,
        mock_list_repo_files,
        tmp_path,
    ):
        mock_download.return_value = str(tmp_path)
        (tmp_path / "gemma-4-E2B-it-Q8_0.gguf").touch()
        (tmp_path / "mmproj-gemma-4-E2B-it-Q8_0.gguf").touch()

        result = download_gguf("ggml-org/gemma-4-E2B-it-GGUF", "Q8_0")

        assert result == str(tmp_path / "gemma-4-E2B-it-Q8_0.gguf")
        assert (
            "mmproj-gemma-4-E2B-it-Q8_0.gguf"
            in mock_download.call_args.kwargs["allow_patterns"]
        )
        assert (
            "mmproj-gemma-4-E2B-it-bf16.gguf"
            not in mock_download.call_args.kwargs["allow_patterns"]
        )
        mock_list_repo_files.assert_called_once_with(
            "ggml-org/gemma-4-E2B-it-GGUF",
            revision=None,
        )

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
            "mmproj-Q4_K_XL.gguf",
            "mmproj-F16.gguf",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_matches_mmproj_for_prefixed_quant_type(
        self,
        mock_download,
        mock_list_repo_files,
        tmp_path,
    ):
        mock_download.return_value = str(tmp_path)
        (tmp_path / "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf").touch()
        (tmp_path / "mmproj-Q4_K_XL.gguf").touch()

        result = download_gguf("unsloth/Qwen3.5-35B-A3B-GGUF", "UD-Q4_K_XL")

        assert result == str(tmp_path / "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf")
        assert "mmproj-Q4_K_XL.gguf" in mock_download.call_args.kwargs["allow_patterns"]
        assert "mmproj-F16.gguf" not in mock_download.call_args.kwargs["allow_patterns"]
        mock_list_repo_files.assert_called_once_with(
            "unsloth/Qwen3.5-35B-A3B-GGUF",
            revision=None,
        )

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "Q4_K_M/model-Q4_K_M.gguf",
            "unrelated/mmproj-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "Q4_K_M/mmproj-F16.gguf",
            "Q4_K_M/processor_config.json",
            "unrelated/processor_config.json",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_ignores_unrelated_mmproj_sidecars(
        self,
        mock_download,
        mock_list_repo_files,
        tmp_path,
    ):
        mock_download.return_value = str(tmp_path)
        (tmp_path / "Q4_K_M").mkdir()
        (tmp_path / "Q4_K_M" / "model-Q4_K_M.gguf").touch()
        (tmp_path / "Q4_K_M" / "mmproj-F16.gguf").touch()
        (tmp_path / "Q4_K_M" / "processor_config.json").touch()

        result = download_gguf("org/repo", "Q4_K_M")

        assert result == str(tmp_path / "Q4_K_M" / "model-Q4_K_M.gguf")
        assert mock_download.call_args.kwargs["allow_patterns"] == [
            "Q4_K_M/model-Q4_K_M.gguf",
            "Q4_K_M/mmproj-F16.gguf",
            "Q4_K_M/processor_config.json",
        ]
        mock_list_repo_files.assert_called_once_with("org/repo", revision=None)

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "model-Q2_K-00001-of-00002.gguf",
            "model-Q2_K-00002-of-00002.gguf",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_sharded_files(
        self,
        mock_download,
        mock_list_repo_files,
        tmp_path,
    ):
        """Test downloading sharded GGUF files."""
        mock_folder = str(tmp_path)
        mock_download.return_value = mock_folder
        (tmp_path / "model-Q2_K-00001-of-00002.gguf").touch()
        (tmp_path / "model-Q2_K-00002-of-00002.gguf").touch()

        result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

        assert result == f"{mock_folder}/model-Q2_K-00001-of-00002.gguf"
        assert mock_download.call_args.kwargs["allow_patterns"] == [
            "model-Q2_K-00001-of-00002.gguf",
            "model-Q2_K-00002-of-00002.gguf",
        ]
        mock_list_repo_files.assert_called_once_with(
            "unsloth/gpt-oss-120b-GGUF",
            revision=None,
        )

    @patch("vllm_gguf_plugin.weight_utils.list_repo_files", return_value=[])
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_subdir(self, mock_download, mock_list_repo_files, tmp_path):
        """Test downloading GGUF files from subdirectory."""
        mock_folder = str(tmp_path)
        mock_download.return_value = mock_folder
        (tmp_path / "Q2_K").mkdir()
        (tmp_path / "Q2_K" / "model-Q2_K.gguf").touch()

        result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

        assert result == f"{mock_folder}/Q2_K/model-Q2_K.gguf"
        mock_list_repo_files.assert_called_once_with(
            "unsloth/gpt-oss-120b-GGUF",
            revision=None,
        )

    @patch("vllm_gguf_plugin.weight_utils.list_repo_files", return_value=[])
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    @patch("glob.glob", return_value=[])
    def test_download_gguf_no_files_found(
        self,
        mock_glob,
        mock_download,
        mock_list_repo_files,
    ):
        """Test error when no GGUF files are found."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with pytest.raises(ValueError, match="Downloaded GGUF files not found"):
            download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")
        mock_list_repo_files.assert_called_once_with(
            "unsloth/Qwen3-0.6B-GGUF",
            revision=None,
        )

    @patch("vllm_gguf_plugin.weight_utils.list_repo_files", return_value=[])
    @patch("vllm_gguf_plugin.weight_utils.hf_hub_download")
    def test_download_gguf_file_single_exact_file(
        self,
        mock_hf_download,
        mock_list_repo_files,
    ):
        mock_hf_download.return_value = "/downloaded/Qwen.gguf"

        result = download_gguf_file(
            "unsloth/Qwen3-0.6B-GGUF",
            "Qwen3-0.6B-Q8_0.gguf",
            cache_dir="/cache",
            revision="abc123",
        )
        mock_list_repo_files.assert_called_once_with(
            "unsloth/Qwen3-0.6B-GGUF",
            revision="abc123",
        )

        assert result == "/downloaded/Qwen.gguf"
        mock_hf_download.assert_called_once_with(
            repo_id="unsloth/Qwen3-0.6B-GGUF",
            filename="Qwen3-0.6B-Q8_0.gguf",
            cache_dir="/cache",
            revision="abc123",
        )

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "Qwen3.6-35B-A3B-NVFP4.gguf",
            "mmproj-BF16.gguf",
            "processor_config.json",
            "preprocessor_config.json",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.hf_hub_download")
    def test_download_gguf_file_single_exact_file_downloads_mmproj(
        self,
        mock_hf_download,
        mock_list_repo_files,
    ):
        def fake_hf_download(**kwargs):
            return f"/downloaded/{kwargs['filename']}"

        mock_hf_download.side_effect = fake_hf_download

        result = download_gguf_file(
            "knoopx/Qwen3.6-35B-A3B-NVFP4-GGUF",
            "Qwen3.6-35B-A3B-NVFP4.gguf",
            cache_dir="/cache",
            revision="abc123",
        )

        assert result == "/downloaded/Qwen3.6-35B-A3B-NVFP4.gguf"
        mock_list_repo_files.assert_called_once_with(
            "knoopx/Qwen3.6-35B-A3B-NVFP4-GGUF",
            revision="abc123",
        )
        assert [
            call.kwargs["filename"] for call in mock_hf_download.call_args_list
        ] == [
            "Qwen3.6-35B-A3B-NVFP4.gguf",
            "mmproj-BF16.gguf",
            "processor_config.json",
            "preprocessor_config.json",
        ]

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "Q8_0/nested/model.gguf",
            "mmproj-BF16.gguf",
            "Q8_0/nested/processor_config.json",
            "Q8_0/preprocessor_config.json",
            "video_preprocessor_config.json",
            "unrelated/processor_config.json",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.hf_hub_download")
    def test_download_gguf_file_downloads_relevant_processor_sidecars(
        self,
        mock_hf_download,
        mock_list_repo_files,
    ):
        def fake_hf_download(**kwargs):
            return f"/downloaded/{kwargs['filename']}"

        mock_hf_download.side_effect = fake_hf_download

        result = download_gguf_file(
            "org/repo",
            "Q8_0/nested/model.gguf",
            cache_dir="/cache",
            revision="abc123",
        )

        assert result == "/downloaded/Q8_0/nested/model.gguf"
        mock_list_repo_files.assert_called_once_with("org/repo", revision="abc123")
        assert [
            call.kwargs["filename"] for call in mock_hf_download.call_args_list
        ] == [
            "Q8_0/nested/model.gguf",
            "mmproj-BF16.gguf",
            "Q8_0/nested/processor_config.json",
            "Q8_0/preprocessor_config.json",
            "video_preprocessor_config.json",
        ]

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "Q8_0/nested/model-Q4_K_M.gguf",
            "unrelated/mmproj-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "Q8_0/mmproj-F16.gguf",
            "Q8_0/nested/mmproj-F16.gguf",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.hf_hub_download")
    def test_download_gguf_file_ignores_unrelated_mmproj_sidecars(
        self,
        mock_hf_download,
        mock_list_repo_files,
    ):
        def fake_hf_download(**kwargs):
            return f"/downloaded/{kwargs['filename']}"

        mock_hf_download.side_effect = fake_hf_download

        result = download_gguf_file(
            "org/repo",
            "Q8_0/nested/model-Q4_K_M.gguf",
            cache_dir="/cache",
            revision="abc123",
        )

        assert result == "/downloaded/Q8_0/nested/model-Q4_K_M.gguf"
        mock_list_repo_files.assert_called_once_with("org/repo", revision="abc123")
        assert [
            call.kwargs["filename"] for call in mock_hf_download.call_args_list
        ] == [
            "Q8_0/nested/model-Q4_K_M.gguf",
            "Q8_0/nested/mmproj-F16.gguf",
        ]

    @patch(
        "vllm_gguf_plugin.weight_utils.list_repo_files",
        return_value=[
            "Qwen-Q4_K_M-00001-of-00002.gguf",
            "Qwen-Q4_K_M-00002-of-00002.gguf",
            "mmproj-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "processor_config.json",
        ],
    )
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_file_split_exact_file_set(
        self,
        mock_download,
        mock_list_repo_files,
        tmp_path,
    ):
        mock_download.return_value = str(tmp_path)
        (tmp_path / "Qwen-Q4_K_M-00001-of-00002.gguf").touch()
        (tmp_path / "Qwen-Q4_K_M-00002-of-00002.gguf").touch()
        (tmp_path / "mmproj-Q4_K_M.gguf").touch()
        (tmp_path / "processor_config.json").touch()

        result = download_gguf_file(
            "org/repo",
            "Qwen-Q4_K_M-00002-of-00002.gguf",
            cache_dir="/cache",
            revision="abc123",
        )

        assert result == str(tmp_path / "Qwen-Q4_K_M-00001-of-00002.gguf")
        mock_download.assert_called_once_with(
            repo_id="org/repo",
            cache_dir="/cache",
            allow_patterns=[
                "Qwen-Q4_K_M-00001-of-00002.gguf",
                "Qwen-Q4_K_M-00002-of-00002.gguf",
                "mmproj-Q4_K_M.gguf",
                "processor_config.json",
            ],
            revision="abc123",
        )
        mock_list_repo_files.assert_called_once_with("org/repo", revision="abc123")

    @patch("vllm_gguf_plugin.weight_utils.list_repo_files", return_value=[])
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_file_split_missing_shard_fails(
        self,
        mock_download,
        mock_list_repo_files,
        tmp_path,
    ):
        mock_download.return_value = str(tmp_path)
        (tmp_path / "Qwen-Q4_K_M-00001-of-00002.gguf").touch()

        with pytest.raises(ValueError, match="Incomplete split GGUF model"):
            download_gguf_file("org/repo", "Qwen-Q4_K_M-00001-of-00002.gguf")
        mock_list_repo_files.assert_called_once_with("org/repo", revision=None)

    def test_resolve_local_gguf_validates_split_shards(self, tmp_path):
        (tmp_path / "model-Q4_K_M-00001-of-00002.gguf").touch()
        (tmp_path / "model-Q4_K_M-00002-of-00002.gguf").touch()

        assert resolve_local_gguf(str(tmp_path), "Q4_K_M") == str(
            tmp_path / "model-Q4_K_M-00001-of-00002.gguf"
        )


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

    @patch("vllm_gguf_plugin.loader.download_gguf_file")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_filename(self, mock_isfile, mock_download_file):
        """Test _prepare_weights with repo_id/filename.gguf format."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_download_file.return_value = "/downloaded/model.gguf"

        model_config = MagicMock()
        model_config.model_weights = "unsloth/Qwen3-0.6B-GGUF/model.gguf"
        model_config.model = "unsloth/Qwen3-0.6B-GGUF"
        model_config.revision = "abc123"

        result = loader._prepare_weights(model_config)
        assert result == "/downloaded/model.gguf"
        mock_download_file.assert_called_once_with(
            repo_id="unsloth/Qwen3-0.6B-GGUF",
            filename="model.gguf",
            cache_dir=None,
            revision="abc123",
        )

    @patch("vllm_gguf_plugin.loader.download_gguf_file")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_subdir_filename(
        self, mock_isfile, mock_download_file
    ):
        """Test _prepare_weights with repo_id/subdir/filename.gguf format."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_download_file.return_value = "/downloaded/model.gguf"

        model_config = MagicMock()
        model_config.model_weights = "org/repo/Q8_0/model.gguf"
        model_config.model = "org/repo"
        model_config.revision = None

        result = loader._prepare_weights(model_config)
        assert result == "/downloaded/model.gguf"
        mock_download_file.assert_called_once_with(
            repo_id="org/repo",
            filename="Q8_0/model.gguf",
            cache_dir=None,
            revision=None,
        )

    @patch("vllm_gguf_plugin.weight_utils.list_repo_files", return_value=[])
    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    @patch("glob.glob")
    @patch("os.path.isdir", return_value=False)
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_remote_repo_quant_type(
        self, mock_isfile, mock_isdir, mock_glob, mock_download, mock_list_repo_files
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
        mock_list_repo_files.assert_called_once_with(
            "unsloth/Qwen3-0.6B-GGUF",
            revision=None,
        )

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
