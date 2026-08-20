# SPDX-License-Identifier: Apache-2.0
"""Tests for the Muse Glimmer DFlash draft GGUF adapter.

The draft's adapter is almost entirely negative space: none of the four
conversions the backbone needs applies to it.  Its Q/K rows are already in NEOX
order, its norms are stored without the folded offset, and its Q/K norms are
learned rather than synthesized.  Applying the backbone's rules here would
rewrite correct weights, and nothing downstream would say so -- the target
verifies every token, so the output stays fluent and correct while the draft's
proposals quietly stop being accepted.  Several tests below therefore assert
that a transformation did *not* happen.

The rest covers the two places where a draft differs structurally from a target
model, both of which produced silent-wrong-weights bugs while this was written:

  · A draft resolves its config and its quantization config separately from the
    target, so declarations recorded on the shared objects never reach it.

  · A draft's layers are numbered after the target's, so a module path that
    looks right in isolation matches nothing at runtime.

Everything runs on synthetic names and tensors; none of it needs a checkpoint.
"""

from types import SimpleNamespace

import pytest
import torch
from gguf import GGMLQuantizationType, dequantize
from transformers import PretrainedConfig
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

from vllm_gguf_plugin.plugin import _redirect_draft_to_its_config_source
from vllm_gguf_plugin.quantization.config import GGUFConfig
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod
from vllm_gguf_plugin.quantization.params import (
    _gguf_replicated_weight_loader,
    _resolve_gguf_weight_loader,
)
from vllm_gguf_plugin.weights_adapter import (
    get_adapter_architecture,
    get_weights_adapter,
)
from vllm_gguf_plugin.weights_adapter.muse_glimmer import (
    MUSE_GLIMMER_DRAFT_ARCHITECTURE,
    MuseGlimmerDraftGGUFAdapter,
    MuseGlimmerGGUFAdapter,
    build_muse_glimmer_draft_name_map,
)

# One layer's worth of GGUF tensor names, plus the three that live outside the
# blocks.  Taken from the shipped dflash checkpoint.
DRAFT_LAYER_TENSORS = [
    "blk.{i}.attn_norm.weight",
    "blk.{i}.attn_q.weight",
    "blk.{i}.attn_k.weight",
    "blk.{i}.attn_v.weight",
    "blk.{i}.attn_output.weight",
    "blk.{i}.attn_q_norm.weight",
    "blk.{i}.attn_k_norm.weight",
    "blk.{i}.ffn_norm.weight",
    "blk.{i}.ffn_gate.weight",
    "blk.{i}.ffn_up.weight",
    "blk.{i}.ffn_down.weight",
]
DRAFT_TOP_LEVEL_TENSORS = [
    "fc.weight",
    "enc.output_norm.weight",
    "output_norm.weight",
]


def draft_tensor_names(num_layers: int = 5) -> list[str]:
    names = list(DRAFT_TOP_LEVEL_TENSORS)
    for layer in range(num_layers):
        names += [name.format(i=layer) for name in DRAFT_LAYER_TENSORS]
    return names


def draft_config(model_type: str = "muse_glimmer_assistant") -> PretrainedConfig:
    return PretrainedConfig(model_type=model_type)


def eagle_wrapped(inner: PretrainedConfig) -> PretrainedConfig:
    """Mimic how EAGLEConfig presents a draft config to the loader.

    It reports ``model_type == "eagle"`` and keeps the real config on ``.model``,
    and it deliberately does not copy the inner ``model_type`` up.
    """
    return PretrainedConfig(model_type="eagle", model=inner)


# --------------------------------------------------------------------------
# Adapter selection
# --------------------------------------------------------------------------
def test_the_draft_adapter_claims_the_assistant_config():
    assert isinstance(get_weights_adapter(draft_config()), MuseGlimmerDraftGGUFAdapter)


def test_the_draft_adapter_is_still_found_through_the_eagle_wrapper():
    """The wrapper goes on after the config parser has run.

    An adapter that only checks the bare model type matches while the
    architecture is being chosen and stops matching by the time weights are
    mapped, at which point the fallback adapter takes over and fails on an
    architecture it has never heard of.
    """
    wrapped = eagle_wrapped(draft_config())
    assert isinstance(get_weights_adapter(wrapped), MuseGlimmerDraftGGUFAdapter)


def test_the_draft_adapter_declares_the_bare_architecture():
    """EAGLEConfig rewrites this to DFlash{arch}; it must not be pre-rewritten."""
    assert get_adapter_architecture(draft_config()) == MUSE_GLIMMER_DRAFT_ARCHITECTURE
    assert not MUSE_GLIMMER_DRAFT_ARCHITECTURE.startswith("DFlash")


@pytest.mark.parametrize("model_type", ["muse_glimmer", "muse_glimmer_text"])
def test_the_draft_adapter_leaves_the_backbone_alone(model_type):
    config = PretrainedConfig(model_type=model_type)
    assert not MuseGlimmerDraftGGUFAdapter.matches(config)
    assert isinstance(get_weights_adapter(config), MuseGlimmerGGUFAdapter)


def test_the_backbone_adapter_does_not_claim_the_draft():
    assert not MuseGlimmerGGUFAdapter.matches(draft_config())


def test_an_unrelated_eagle_draft_is_not_claimed():
    """The wrapper is generic, so unwrapping must not widen what matches."""
    wrapped = eagle_wrapped(PretrainedConfig(model_type="llama"))
    assert not MuseGlimmerDraftGGUFAdapter.matches(wrapped)


# --------------------------------------------------------------------------
# Name mapping
# --------------------------------------------------------------------------
def test_the_name_map_is_a_bijection_over_the_checkpoint():
    names = draft_tensor_names()
    name_map = build_muse_glimmer_draft_name_map(names)

    assert set(name_map) == set(names), "some GGUF tensor went unmapped"
    assert len(set(name_map.values())) == len(names), "two tensors share a name"


def test_the_draft_discards_nothing():
    """Unlike the backbone, which drops 104 synthesized Q/K norms.

    Reusing that rule here would remove real learned weights.
    """
    names = draft_tensor_names()
    name_map = build_muse_glimmer_draft_name_map(names)

    assert len(name_map) == len(names)
    for layer in range(5):
        for kind in ("q", "k"):
            assert f"blk.{layer}.attn_{kind}_norm.weight" in name_map


@pytest.mark.parametrize(
    "gguf_name,hf_name",
    [
        ("fc.weight", "encoder.fc.weight"),
        ("enc.output_norm.weight", "encoder.output_norm_enc.weight"),
        ("output_norm.weight", "norm.weight"),
    ],
)
def test_the_tensors_outside_the_blocks_are_renamed(gguf_name, hf_name):
    """llama.cpp abbreviates these three; nothing about them is derivable."""
    name_map = build_muse_glimmer_draft_name_map(draft_tensor_names())
    assert name_map[gguf_name] == hf_name


@pytest.mark.parametrize(
    "suffix,hf_suffix",
    [
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
    ],
)
def test_the_within_layer_renames(suffix, hf_suffix):
    name_map = build_muse_glimmer_draft_name_map(draft_tensor_names())
    assert name_map[f"blk.3.{suffix}.weight"] == f"layers.3.{hf_suffix}.weight"


def test_an_unrecognized_tensor_is_skipped_rather_than_guessed(caplog):
    name_map = build_muse_glimmer_draft_name_map(
        [*draft_tensor_names(1), "blk.0.something_new.weight"]
    )
    assert "blk.0.something_new.weight" not in name_map
    assert "something_new" in caplog.text


# --------------------------------------------------------------------------
# What the draft must *not* do to its weights
# --------------------------------------------------------------------------
def build_adapter(num_layers: int = 2) -> MuseGlimmerDraftGGUFAdapter:
    """Create an adapter in the order the loader calls it."""
    adapter = MuseGlimmerDraftGGUFAdapter()
    adapter.build_name_map(
        SimpleNamespace(all_files=(), backbone=(), primary_backbone=None),
        SimpleNamespace(hf_config=draft_config(), dtype=torch.bfloat16),
    )
    return adapter


@pytest.fixture
def draft_adapter(monkeypatch):
    monkeypatch.setattr(
        "vllm_gguf_plugin.weights_adapter.muse_glimmer.get_gguf_tensor_names",
        lambda _files: draft_tensor_names(2),
    )
    return build_adapter()


def transformed(adapter, weights):
    model_config = SimpleNamespace(hf_config=draft_config(), dtype=torch.bfloat16)
    return dict(adapter.transform_weights(iter(weights), model_config))


# A Q4_K super-block spends 144 bytes on 256 weights: the fp16 pair `d`/`dmin`,
# then 6-bit scales and 4-bit quants, both integer fields.
Q4_K_WEIGHTS_PER_BLOCK = 256
Q4_K_BLOCK_BYTES = 144


def packed_q4_k(rows: int, cols: int, *, seed: int) -> torch.Tensor:
    """Random Q4_K bytes whose super-block scales are finite.

    The opening pair is the only part read as floating point, and uniform
    random bytes give one of them a NaN exponent for about one super-block in
    twenty.  That would be a test failing on its own data: a NaN compares
    unequal to itself, so the dequantized rows would not match a reference
    dequantization of the very same bytes.
    """
    blocks = cols // Q4_K_WEIGHTS_PER_BLOCK
    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0,
        256,
        (rows, blocks * Q4_K_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    scales = torch.tensor([1.0, 0.5], dtype=torch.float16).view(torch.uint8)
    for block in range(blocks):
        start = block * Q4_K_BLOCK_BYTES
        packed[:, start : start + scales.numel()] = scales
    return packed


def test_the_norms_pass_through_untouched(draft_adapter):
    """The backbone subtracts one here.  The draft must not.

    Its norms are plain RMSNorm weights, so an offset that the backbone's
    checkpoint folds in was never folded in to begin with.
    """
    norm = torch.tensor([1.5, 0.5, 2.0], dtype=torch.bfloat16)
    out = transformed(draft_adapter, [("layers.0.input_layernorm.weight", norm)])

    torch.testing.assert_close(out["layers.0.input_layernorm.weight"], norm)


def test_the_final_norm_passes_through_untouched(draft_adapter):
    norm = torch.tensor([1.5, 0.5, 2.0], dtype=torch.bfloat16)
    out = transformed(draft_adapter, [("norm.weight", norm)])

    torch.testing.assert_close(out["norm.weight"], norm)


def test_the_qk_norms_survive(draft_adapter):
    """They are learned weights here, not the synthesized ones the backbone drops."""
    weight = torch.tensor([1.25, 0.75], dtype=torch.bfloat16)
    out = transformed(draft_adapter, [("layers.1.self_attn.q_norm.weight", weight)])

    torch.testing.assert_close(out["layers.1.self_attn.q_norm.weight"], weight)


def test_the_output_projection_keeps_its_packed_bytes(draft_adapter):
    """Only Q/K/V are unpacked; everything else stays quantized.

    Unpacking more than necessary is not a correctness bug, which is exactly why
    it needs a test -- it would just quietly cost memory.
    """
    packed = torch.arange(16, dtype=torch.uint8)
    out = transformed(
        draft_adapter,
        [
            ("layers.0.self_attn.o_proj.qweight_type", torch.tensor(14)),
            ("layers.0.self_attn.o_proj.qweight", packed),
        ],
    )

    assert "layers.0.self_attn.o_proj.weight" not in out
    torch.testing.assert_close(out["layers.0.self_attn.o_proj.qweight"], packed)


@pytest.mark.parametrize("kind", ["q", "k", "v"])
def test_the_qkv_projections_are_unpacked(kind, draft_adapter):
    """The head reads qkv_proj.weight to build its fused KV buffer."""
    rows, cols = 4, 256
    block = GGMLQuantizationType.Q4_K
    packed = packed_q4_k(rows, cols, seed=0)
    name = f"layers.0.self_attn.{kind}_proj"
    out = transformed(
        draft_adapter,
        [
            (f"{name}.qweight_type", torch.tensor(int(block))),
            (f"{name}.qweight", packed),
        ],
    )

    assert f"{name}.qweight" not in out, "the packed form must not also be yielded"
    assert out[f"{name}.weight"].shape == (rows, cols)
    assert out[f"{name}.weight"].dtype == torch.bfloat16


@pytest.mark.parametrize("kind", ["q", "k"])
def test_the_qk_rows_keep_the_order_they_were_stored_in(kind, draft_adapter):
    """The backbone reorders these rows.  The draft's are already NEOX.

    Unpacking is the only thing allowed to happen to them, so the result has to
    equal a plain dequantization of the same bytes.  A permutation added here
    would not fail loudly: the target verifies every token, so the output would
    stay fluent while the draft's proposals stopped being accepted.
    """
    rows, cols = 8, 256
    block = GGMLQuantizationType.Q4_K
    packed = packed_q4_k(rows, cols, seed=1)
    name = f"layers.0.self_attn.{kind}_proj"
    out = transformed(
        draft_adapter,
        [
            (f"{name}.qweight_type", torch.tensor(int(block))),
            (f"{name}.qweight", packed),
        ],
    )

    reference = torch.from_numpy(
        dequantize(packed.numpy(), block).reshape(rows, cols)
    ).to(torch.bfloat16)
    assert reference.isfinite().all(), (
        "a NaN here would compare unequal to itself and fail the assertion below "
        "regardless of what the adapter did"
    )
    torch.testing.assert_close(out[f"{name}.weight"], reference)


# --------------------------------------------------------------------------
# Reaching a draft's own quantization config
# --------------------------------------------------------------------------
@pytest.fixture
def single_rank(monkeypatch):
    """Let linear layers be built without a distributed group.

    Only the layer's type matters to the code under test, but constructing one
    still asks for the tensor-parallel rank.
    """
    import vllm.model_executor.layers.linear as linear_module
    import vllm.model_executor.parameter as parameter_module

    for module in (linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)


def dense_config() -> GGUFConfig:
    config = GGUFConfig(
        dense_module_suffixes=list(MuseGlimmerDraftGGUFAdapter.dense_module_suffixes)
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    return config


def test_a_declaration_survives_being_rebuilt_from_the_config_dict():
    """A draft's layers are built against a config rebuilt from this dict.

    Anything the loader records on the shared object reaches the target model
    and never the draft, so dropping these keys would leave a draft with an
    empty declaration and no sign of why.
    """
    rebuilt = GGUFConfig.from_config(
        {
            "quant_method": "gguf",
            "unquantized_modules": ["embed_tokens"],
            "dense_module_suffixes": ["self_attn.qkv_proj"],
        }
    )

    assert rebuilt.unquantized_modules == ["embed_tokens"]
    assert rebuilt.dense_module_suffixes == ["self_attn.qkv_proj"]


def test_a_config_dict_without_the_keys_still_builds():
    """Target models supply neither key."""
    rebuilt = GGUFConfig.from_config({"quant_method": "gguf"})

    assert rebuilt.unquantized_modules == []
    assert rebuilt.dense_module_suffixes == []


def test_a_suffix_declaration_survives_the_layer_renumbering(single_rank):
    """vLLM numbers a draft's layers after the target's.

    A five-layer draft behind a sixty-two-layer target is asked about
    ``model.layers.62..66``.  Declaring the layers by name means predicting that
    offset; matching on the suffix does not.
    """
    config = dense_config()
    layer = QKVParallelLinear(
        hidden_size=64,
        head_size=16,
        total_num_heads=4,
        quant_config=None,
        disable_tp=True,
    )

    for index in (0, 62, 66, 999):
        method = config.get_quant_method(
            layer, prefix=f"model.layers.{index}.self_attn.qkv_proj"
        )
        assert isinstance(method, UnquantizedLinearMethod), (
            f"layer {index} was left quantized"
        )


def test_the_suffix_declaration_does_not_leak_to_other_layers(single_rank):
    """Only the fused attention projection is exempt."""
    config = dense_config()
    layer = RowParallelLinear(
        input_size=64, output_size=64, quant_config=None, disable_tp=True
    )

    for suffix in ("self_attn.o_proj", "mlp.down_proj", "mlp.gate_up_proj", "fc"):
        method = config.get_quant_method(layer, prefix=f"model.layers.3.{suffix}")
        assert isinstance(method, GGUFLinearMethod), f"{suffix} should stay packed"


def test_a_config_without_suffixes_quantizes_everything(single_rank):
    """Negative control: the exemption comes from the declaration, not the path."""
    config = GGUFConfig()
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    layer = QKVParallelLinear(
        hidden_size=64,
        head_size=16,
        total_num_heads=4,
        quant_config=None,
        disable_tp=True,
    )

    method = config.get_quant_method(layer, prefix="model.layers.62.self_attn.qkv_proj")
    assert isinstance(method, GGUFLinearMethod)


# --------------------------------------------------------------------------
# The one linear layer GGUF had never had to load
# --------------------------------------------------------------------------
def test_an_unsharded_linear_gets_a_loader_that_can_size_itself(single_rank):
    """ReplicatedLinear has no v2 loader, and its v1 loader asserts the shape.

    GGUF parameters start empty and take their shape from the packed bytes, so
    the assertion fires on the first tensor.  The draft's ``fc`` is the first
    ReplicatedLinear any GGUF model has needed.
    """
    layer = ReplicatedLinear(
        input_size=32, output_size=8, quant_config=None, disable_tp=True
    )

    assert not hasattr(layer, "weight_loader_v2")
    resolved = _resolve_gguf_weight_loader(layer, layer.weight_loader)
    assert resolved is _gguf_replicated_weight_loader


def test_a_sharded_linear_still_uses_the_v2_loader(single_rank):
    layer = RowParallelLinear(
        input_size=32, output_size=8, quant_config=None, disable_tp=True
    )

    resolved = _resolve_gguf_weight_loader(layer, layer.weight_loader)
    assert resolved == layer.weight_loader_v2


# --------------------------------------------------------------------------
# Pointing a separate-file draft at its own config
# --------------------------------------------------------------------------
def engine_args(speculative_config):
    return SimpleNamespace(speculative_config=speculative_config)


def test_a_draft_is_redirected_to_the_config_directory_it_was_given(tmp_path):
    config_dir = tmp_path / "assistant"
    config_dir.mkdir()
    (config_dir / "config.json").write_text("{}")
    draft = str(tmp_path / "draft.gguf")

    args = engine_args({"model": draft, "hf_config_path": str(config_dir)})
    weights = _redirect_draft_to_its_config_source(args)

    assert weights == draft, "the file has to come back as the weights source"
    assert args.speculative_config["model"] == str(config_dir)
    assert args.speculative_config["quantization"] == "gguf"
    assert "hf_config_path" not in args.speculative_config, (
        "SpeculativeConfig rejects fields it does not declare"
    )


def test_redirecting_twice_still_reports_the_weights(tmp_path):
    """The rewrite edits the caller's dict, so a second pass sees no GGUF path.

    Losing the weights path there is not an error: the draft keeps the config
    directory as its weights source and loads whatever checkpoint is sitting in
    it, which for this layout is the unquantized one.
    """
    config_dir = tmp_path / "assistant"
    config_dir.mkdir()
    (config_dir / "config.json").write_text("{}")
    draft = str(tmp_path / "draft.gguf")
    args = engine_args({"model": draft, "hf_config_path": str(config_dir)})

    first = _redirect_draft_to_its_config_source(args)
    second = _redirect_draft_to_its_config_source(args)

    assert first == second == draft


def test_a_draft_without_a_config_says_what_to_pass(tmp_path):
    """The directory a draft sits in belongs to the model it drafts for."""
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"")
    args = engine_args({"model": str(draft)})

    with pytest.raises(ValueError, match="hf_config_path"):
        _redirect_draft_to_its_config_source(args)


def test_an_unquantized_draft_is_left_alone():
    args = engine_args({"model": "/models/some-draft", "num_speculative_tokens": 3})

    assert _redirect_draft_to_its_config_source(args) is None
    assert args.speculative_config["model"] == "/models/some-draft"
    assert "quantization" not in args.speculative_config


def test_no_speculative_config_is_left_alone():
    assert _redirect_draft_to_its_config_source(engine_args(None)) is None


def test_an_explicit_quantization_choice_is_respected(tmp_path):
    config_dir = tmp_path / "assistant"
    config_dir.mkdir()
    (config_dir / "config.json").write_text("{}")
    args = engine_args(
        {
            "model": str(tmp_path / "draft.gguf"),
            "hf_config_path": str(config_dir),
            "quantization": "awq",
        }
    )

    _redirect_draft_to_its_config_source(args)

    assert args.speculative_config["quantization"] == "awq"
