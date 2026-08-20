# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import ResolvedRevision
from vllm.config.load import LoadConfig

import vllm_gguf_plugin.loader as loader_module
from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.loader import GGUFModelLoader
from vllm_gguf_plugin.weight_utils import (
    download_gguf,
    download_mmproj,
    resolve_local_gguf,
)


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
                    "*.IQ1_S.gguf",
                    "*-IQ1_S-*.gguf",
                    "*-IQ1_S.gguf",
                    "*.iq1_s-*.gguf",
                    "*.iq1_s.gguf",
                    "*-iq1_s-*.gguf",
                    "*-iq1_s.gguf",
                ],
                revision=None,
                ignore_patterns=None,
            )

            assert result == f"{mock_folder}/model-IQ1_S.gguf"

    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_sharded_files(self, mock_download):
        """Test downloading sharded GGUF files."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [
                    f"{mock_folder}/model-Q2_K-00001-of-00002.gguf",
                    f"{mock_folder}/model-Q2_K-00002-of-00002.gguf",
                ]
                if "Q2_K" in pattern
                else []
            )

            result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

            assert result == f"{mock_folder}/model-Q2_K-00001-of-00002.gguf"

    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    def test_download_gguf_subdir(self, mock_download, tmp_path):
        """Test downloading GGUF files from subdirectory."""
        quant_dir = tmp_path / "Q2_K"
        quant_dir.mkdir()
        model = quant_dir / "model-Q2_K.gguf"
        model.touch()
        mock_download.return_value = str(tmp_path)

        result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

        assert result == str(model)

    def test_resolve_local_gguf_subdir(self, tmp_path):
        quant_dir = tmp_path / "UD-Q2_K_XL"
        quant_dir.mkdir()
        model = quant_dir / "Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf"
        model.touch()
        assert resolve_local_gguf(str(tmp_path), "UD-Q2_K_XL") == str(model)

    @patch("vllm_gguf_plugin.weight_utils.snapshot_download")
    @patch("glob.glob", return_value=[])
    def test_download_gguf_no_files_found(self, mock_glob, mock_download):
        """Test error when no GGUF files are found."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with pytest.raises(ValueError, match="Downloaded GGUF files not found"):
            download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")

    @patch("vllm_gguf_plugin.weight_utils.hf_hub_download")
    @patch("vllm_gguf_plugin.weight_utils.list_filtered_repo_files")
    def test_download_mmproj_prefers_bfloat16(self, mock_list, mock_download):
        mock_list.return_value = [
            "mmproj-F32.gguf",
            "mmproj-F16.gguf",
            "mmproj-BF16.gguf",
        ]
        mock_download.return_value = "/cache/mmproj-BF16.gguf"

        result = download_mmproj("org/model", revision="main")

        assert result == "/cache/mmproj-BF16.gguf"
        mock_download.assert_called_once_with(
            repo_id="org/model",
            filename="mmproj-BF16.gguf",
            cache_dir=None,
            revision="main",
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

    @pytest.mark.parametrize(
        ("initial_revision", "resolved_revision", "expected_revision"),
        [
            (None, "b" * 40, None),
            ("weights-branch", "b" * 40, "weights-branch"),
            ("a" * 40, "a" * 40, "a" * 40),
        ],
    )
    @patch("vllm_gguf_plugin.loader.download_gguf")
    @patch("os.path.isdir", return_value=False)
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_detaches_revision_resolved_for_another_repo(
        self,
        mock_isfile,
        mock_isdir,
        mock_download,
        initial_revision,
        resolved_revision,
        expected_revision,
    ):
        mock_download.return_value = "/downloaded/model-Q4_0.gguf"
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        model_config = MagicMock()
        model_config.model = "org/config-repo"
        model_config.model_weights = "org/weights-repo:Q4_0"
        model_config.revision = ResolvedRevision(
            initial=initial_revision,
            resolved=resolved_revision,
        )

        result = loader._prepare_weights(model_config)

        assert result == "/downloaded/model-Q4_0.gguf"
        mock_download.assert_called_once_with(
            "org/weights-repo",
            "Q4_0",
            cache_dir=None,
            revision=expected_revision,
            ignore_patterns=load_config.ignore_patterns,
        )
        revision = mock_download.call_args.kwargs["revision"]
        assert revision is None or type(revision) is str

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

    def test_rejects_unknown_extra_config(self):
        load_config = LoadConfig(
            load_format="gguf",
            model_loader_extra_config={"unknown": "value"},
        )

        with pytest.raises(ValueError, match="Unsupported GGUF"):
            GGUFModelLoader(load_config)

    def test_prepare_model_files_uses_explicit_mmproj(self, tmp_path):
        backbone = tmp_path / "model.gguf"
        mm_proj = tmp_path / "projector-custom.gguf"
        backbone.write_bytes(b"GGUF")
        mm_proj.write_bytes(b"GGUF")

        loader = GGUFModelLoader(
            LoadConfig(
                load_format="gguf",
                model_loader_extra_config={"mm_proj": str(mm_proj)},
            )
        )
        model_config = SimpleNamespace(
            model_weights=str(backbone),
            model=str(backbone),
            revision=None,
            hf_config=SimpleNamespace(),
        )

        files = loader._prepare_model_files(model_config)

        assert files == GGUFModelFiles((str(backbone),), str(mm_proj))

    def test_prepare_model_files_downloads_remote_mmproj(self, monkeypatch):
        loader = GGUFModelLoader(LoadConfig(load_format="gguf"))
        model_config = SimpleNamespace(
            model_weights="org/model:Q4_0",
            model="org/model",
            revision="main",
            hf_config=SimpleNamespace(vision_config=object()),
        )

        monkeypatch.setattr(
            loader,
            "_prepare_weights",
            lambda config: "/cache/model-Q4_0.gguf",
        )
        download_calls = []

        def fake_download(repo_id, cache_dir=None, revision=None):
            download_calls.append((repo_id, cache_dir, revision))
            return "/cache/mmproj-BF16.gguf"

        monkeypatch.setattr(loader_module, "download_mmproj", fake_download)

        files = loader._prepare_model_files(model_config)

        assert files == GGUFModelFiles(
            ("/cache/model-Q4_0.gguf",), "/cache/mmproj-BF16.gguf"
        )
        assert download_calls == [("org/model", None, "main")]
