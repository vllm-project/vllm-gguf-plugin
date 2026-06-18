# SPDX-License-Identifier: Apache-2.0

import sys

import vllm_gguf_plugin.plugin as plugin_module


def test_import_does_not_eagerly_import_quantization():
    assert "vllm_gguf_plugin.quantization.linear" not in sys.modules
    assert "vllm_gguf_plugin.quantization.fused_moe" not in sys.modules


def test_register_noops_when_vllm_already_has_gguf(monkeypatch):
    imported = False

    def fail_if_loaded():
        nonlocal imported
        imported = True
        raise AssertionError("plugin classes should not load for in-tree GGUF")

    monkeypatch.setattr(plugin_module, "_load_oot_gguf_classes", fail_if_loaded)
    monkeypatch.setattr(plugin_module, "QUANTIZATION_METHODS", ["gguf"])

    plugin_module.register()

    assert imported is False


def test_register_can_override_in_tree_when_explicit(monkeypatch):
    calls = []

    class FakeConfig:
        pass

    class FakeParser:
        pass

    class FakeLoader:
        pass

    def fake_load():
        calls.append("load")
        return FakeConfig, FakeParser, FakeLoader

    def fake_register_quantization_config(name):
        assert name == "gguf"

        def inner(config):
            calls.append(("quant", config))
            return config

        return inner

    def fake_register_model_loader(name):
        assert name == "gguf"

        def inner(loader):
            calls.append(("loader", loader))
            return loader

        return inner

    def fake_register_config_parser(name):
        assert name == "gguf"

        def inner(parser):
            calls.append(("parser", parser))
            return parser

        return inner

    monkeypatch.setenv("VLLM_GGUF_PLUGIN_OVERRIDE_IN_TREE", "1")
    monkeypatch.setattr(plugin_module, "_load_oot_gguf_classes", fake_load)
    monkeypatch.setattr(plugin_module, "QUANTIZATION_METHODS", ["gguf"])
    monkeypatch.setattr(
        plugin_module,
        "register_quantization_config",
        fake_register_quantization_config,
    )
    monkeypatch.setattr(
        plugin_module,
        "_patch_quantization_config_lookup",
        lambda: calls.append("quant_lookup"),
    )
    monkeypatch.setattr(plugin_module, "_LOAD_FORMAT_TO_MODEL_LOADER", {})
    monkeypatch.setattr(
        plugin_module,
        "register_model_loader",
        fake_register_model_loader,
    )
    monkeypatch.setattr(plugin_module, "get_config_parser", lambda name: None)
    monkeypatch.setattr(
        plugin_module,
        "register_config_parser",
        fake_register_config_parser,
    )
    monkeypatch.setattr(
        plugin_module,
        "_patch_engine_args",
        lambda: calls.append("engine_args"),
    )
    monkeypatch.setattr(
        plugin_module,
        "_patch_speculator_probe",
        lambda: calls.append("speculator"),
    )

    plugin_module.register()

    assert calls == [
        "load",
        ("quant", FakeConfig),
        "quant_lookup",
        ("loader", FakeLoader),
        ("parser", FakeParser),
        "engine_args",
        "speculator",
    ]
