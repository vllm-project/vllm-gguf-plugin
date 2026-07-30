# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from transformers import PretrainedConfig

from vllm_gguf_plugin.weights_adapter.default import GGUFWeightsAdapter


def make_adapter(files: list[str]) -> GGUFWeightsAdapter:
    adapter = object.__new__(GGUFWeightsAdapter)
    adapter._get_all_gguf_files = MagicMock(return_value=files)
    return adapter


def test_lm_head_present_in_one_shard_is_not_tied():
    adapter = make_adapter(["model-00001.gguf", "model-00002.gguf"])
    config = PretrainedConfig(tie_word_embeddings=True)
    name_map = {
        "output.weight": "lm_head.weight",
        "token_embd.weight": "model.embed_tokens.weight",
    }

    with patch(
        "vllm_gguf_plugin.weights_adapter.default.get_gguf_extra_tensor_names",
        side_effect=[
            ["model.embed_tokens.weight"],
            ["lm_head.weight"],
        ],
    ):
        adapter.update_tie_word_embeddings("model-00001.gguf", config, name_map)

    assert config.tie_word_embeddings is False


def test_lm_head_absent_from_every_shard_is_tied():
    adapter = make_adapter(["model-00001.gguf", "model-00002.gguf"])
    config = PretrainedConfig(tie_word_embeddings=False)
    name_map = {
        "output.weight": "lm_head.weight",
        "token_embd.weight": "model.embed_tokens.weight",
    }

    with patch(
        "vllm_gguf_plugin.weights_adapter.default.get_gguf_extra_tensor_names",
        side_effect=[
            ["lm_head.weight", "model.embed_tokens.weight"],
            ["lm_head.weight"],
        ],
    ):
        adapter.update_tie_word_embeddings("model-00001.gguf", config, name_map)

    assert config.tie_word_embeddings is True
