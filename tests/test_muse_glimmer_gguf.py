# SPDX-License-Identifier: Apache-2.0
"""Tests for the Muse Glimmer GGUF adapter.

Every conversion this adapter undoes fails quietly when it is wrong: the weights
load without complaint and the model generates fluent, incorrect text.  So these
tests pin the conversions themselves rather than that loading succeeds.  They run
on synthetic tensors and synthetic tensor names, so none of them needs the real
30B checkpoint.

Muse Glimmer GGUF checkpoints store Q/K with llama.cpp's interleaved rotary
layout while vLLM's implementation hardcodes NEOX, so the adapter permutes rows
on the way in.  The first group of tests pins the three properties that make the
permutation safe to apply to packed quantized bytes.

Quantized coverage uses the small synthetic sample tensors from
``Isotr0py/test-gguf-sample`` rather than a real checkpoint, since the property
under test is generic to any K-quant tensor.

The dequantization reference is the Triton kernel, deliberately: it computes in
fp32 and matches ``gguf.dequantize`` bit-for-bit, whereas the native CUDA/HIP
kernel computes in fp16 (see ``csrc/gguf/dequantize.cuh``) and would turn these
exact comparisons into tolerance comparisons.
"""

from types import SimpleNamespace

import pytest
import torch
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from transformers import PretrainedConfig

from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton
from vllm_gguf_plugin.weights_adapter import muse_glimmer as adapter_module
from vllm_gguf_plugin.weights_adapter.muse_glimmer import (
    MUSE_GLIMMER_ARCHITECTURES,
    MuseGlimmerGGUFAdapter,
    has_vision,
    interleaved_to_neox_row_index,
    neox_to_interleaved_row_index,
    reconstruct_patch_embedding,
    undo_rope_interleave,
)

from .utils import get_gguf_sample_tensors

# 96 and 128 are the vision and text head_dim of the shipped Muse Glimmer
# checkpoints; the rest are there to catch off-by-one indexing. 4 is excluded on
# purpose: the permutation is accidentally self-inverse there, which hides bugs.
HEAD_DIMS = [8, 16, 64, 96, 128]
K_QUANT_TYPES = [
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
]
HIDDEN_SIZES = [256, 1024]


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_heads", [1, 2, 32])
def test_forward_and_inverse_index_compose_to_identity(num_heads, head_dim):
    forward = neox_to_interleaved_row_index(num_heads, head_dim)
    inverse = interleaved_to_neox_row_index(num_heads, head_dim)
    identity = torch.arange(num_heads * head_dim)

    torch.testing.assert_close(forward[inverse], identity)
    torch.testing.assert_close(inverse[forward], identity)


@pytest.mark.parametrize("head_dim", [8, 16, 64, 128])
def test_inverse_index_is_not_self_inverse(head_dim):
    """Guards the most likely way to get this wrong.

    Applying the permutation twice looks like an inverse for head_dim == 4 but
    diverges for every larger size, so a helper that calls the forward function
    twice passes toy tests and corrupts real checkpoints.
    """
    inverse = interleaved_to_neox_row_index(2, head_dim)
    identity = torch.arange(2 * head_dim)

    assert not torch.equal(inverse[inverse], identity)


def test_head_dim_four_is_the_misleading_case():
    """Documents why head_dim == 4 must not be used as the only test size."""
    inverse = interleaved_to_neox_row_index(1, 4)

    assert torch.equal(inverse[inverse], torch.arange(4))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_undo_rope_interleave_inverts_the_conversion(head_dim):
    """End-to-end on float rows: forward re-layout then undo is the identity."""
    num_heads = 4
    original = torch.randn(num_heads * head_dim, 7)

    converted = original.index_select(
        0, neox_to_interleaved_row_index(num_heads, head_dim)
    )
    assert not torch.equal(converted, original)

    torch.testing.assert_close(
        undo_rope_interleave(converted, num_heads, head_dim), original
    )


def test_undo_rope_interleave_rejects_row_count_mismatch():
    with pytest.raises(ValueError, match="expected 256 rows"):
        undo_rope_interleave(torch.zeros(255, 4), num_heads=2, head_dim=128)


def test_odd_head_dim_is_rejected():
    with pytest.raises(ValueError, match="head_dim must be even"):
        interleaved_to_neox_row_index(1, 7)


@pytest.mark.parametrize("quant_type", K_QUANT_TYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_qweight_rows_are_self_contained_byte_runs(quant_type, hidden_size):
    """Super-blocks split along the input dim, so a row is whole blocks.

    This is the precondition that lets the adapter permute packed bytes at all.
    """
    block_size, type_size = GGML_QUANT_SIZES[quant_type]

    for tensor in get_gguf_sample_tensors(hidden_size, quant_type):
        qweight = torch.tensor(tensor.data)
        assert qweight.ndim == 2
        row_bytes = qweight.shape[1]

        assert row_bytes % type_size == 0, (
            f"{tensor.name}: row of {row_bytes} bytes is not a whole number of "
            f"{type_size}-byte blocks"
        )
        assert (row_bytes // type_size) * block_size == hidden_size


@pytest.mark.parametrize("quant_type", K_QUANT_TYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_qweight_byte_permutation_round_trips(quant_type, hidden_size):
    """Permuting packed bytes forward then back recovers the original bytes."""
    head_dim = 128

    for tensor in get_gguf_sample_tensors(hidden_size, quant_type):
        qweight = torch.tensor(tensor.data)
        num_rows = qweight.shape[0]
        if num_rows % head_dim:
            continue
        num_heads = num_rows // head_dim

        forward = qweight.index_select(
            0, neox_to_interleaved_row_index(num_heads, head_dim)
        )
        restored = undo_rope_interleave(forward, num_heads, head_dim)

        assert torch.equal(restored, qweight), f"{tensor.name}: bytes changed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("quant_type", K_QUANT_TYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_permute_then_dequantize_equals_dequantize_then_permute(
    quant_type, hidden_size
):
    """The load-bearing equivalence: reordering bytes is safe.

    If this holds, the adapter can permute rows in the quantized domain at zero
    memory cost and zero precision loss, instead of dequantizing Q/K (which
    would drag the whole fused QKV projection to bf16).
    """
    head_dim = 128
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    checked = 0

    for tensor in get_gguf_sample_tensors(hidden_size, quant_type):
        qweight = torch.tensor(tensor.data).cuda()
        num_rows, row_bytes = qweight.shape
        if num_rows % head_dim:
            continue
        num_heads = num_rows // head_dim
        shape = (num_rows, row_bytes // type_size * block_size)

        straight = ggml_dequantize_triton(
            qweight, int(quant_type), *shape, torch.float32
        )
        permuted_first = ggml_dequantize_triton(
            undo_rope_interleave(qweight, num_heads, head_dim),
            int(quant_type),
            *shape,
            torch.float32,
        )
        dequantized_first = undo_rope_interleave(straight, num_heads, head_dim)

        # Without this the equality below would also hold for a no-op permutation.
        assert not torch.equal(permuted_first, straight), (
            f"{tensor.name}: permutation had no effect, equality is vacuous"
        )
        assert torch.equal(permuted_first, dequantized_first), (
            f"{tensor.name}: permuting bytes and permuting floats disagree"
        )
        checked += 1

    assert checked, "no sample tensor had a row count divisible by head_dim"


# --------------------------------------------------------------------------
# Which weights the adapter loads at all
# --------------------------------------------------------------------------
VISION_LAYERS = (0, 1)
# The GGUF side of the naming, which belongs to the file format rather than to
# this plugin, so it is spelled out here instead of read back off the mapper.
_VISION_LEAVES = ("attn_q", "attn_k", "attn_v", "attn_out")


def _vision_config(**kwargs) -> PretrainedConfig:
    config = PretrainedConfig()
    config.vision_config = PretrainedConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def test_has_vision_matches_the_rule_vllm_applies():
    """The two have to stay the same rule, so compare against vLLM's own.

    The adapter decides which weights to produce and vLLM decides which modules
    to build.  Whenever the two disagree, one side has weights the other has
    nowhere to put -- so this pins them together rather than trusting that a
    copied predicate stays a copy.
    """
    from vllm.model_executor.models.muse_glimmer import _muse_glimmer_has_vision

    configs = [
        _vision_config(),
        _vision_config(has_vision=True),
        _vision_config(has_vision=False),
        PretrainedConfig(),
    ]

    for config in configs:
        assert has_vision(config) == _muse_glimmer_has_vision(config)

    # Spelled out as well, so a change to both at once still has to be deliberate.
    assert [has_vision(config) for config in configs] == [True, True, False, False]


def test_architecture_names_the_class_vllm_will_build():
    """Crossing the two model types is another failure that does not raise.

    Neither type can be looked up in the mapping the config parser consults, so
    the adapter has to name the class itself.  Pointing the multimodal type at
    the causal-LM class loads and generates perfectly well -- as a model with no
    vision tower at all -- so this checks both against their real sources rather
    than against a copy of the same dict.
    """
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )
    from vllm.model_executor.models.registry import ModelRegistry

    assert (
        MUSE_GLIMMER_ARCHITECTURES["muse_glimmer"]
        == (MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES["muse_glimmer"])
    )
    assert (
        MUSE_GLIMMER_ARCHITECTURES["muse_glimmer_text"]
        != (MUSE_GLIMMER_ARCHITECTURES["muse_glimmer"])
    )

    supported = ModelRegistry.get_supported_archs()
    for model_type, architecture in MUSE_GLIMMER_ARCHITECTURES.items():
        config = SimpleNamespace(model_type=model_type)
        assert MuseGlimmerGGUFAdapter.matches(config)
        assert MuseGlimmerGGUFAdapter.architecture(config) == architecture
        assert architecture in supported


def _build(monkeypatch, config, *, projector: str | None, gguf_names=()):
    """Drive the adapter through the loader's sequence, without real files."""
    monkeypatch.setattr(
        adapter_module,
        "maybe_patch_hf_config_from_gguf",
        lambda _path, cfg, mmproj_path=None: cfg,
    )
    monkeypatch.setattr(
        adapter_module, "get_gguf_tensor_names", lambda _files: list(gguf_names)
    )

    files = GGUFModelFiles(backbone=("backbone.gguf",), mm_proj=projector)
    adapter = MuseGlimmerGGUFAdapter()
    patched = adapter.patch_hf_config(files, config)
    name_map = adapter.build_name_map(files, SimpleNamespace(hf_config=patched))
    return adapter, name_map


_PROJECTOR_NAMES = ("v.blk.0.attn_q.weight", "mm.2.weight")


def test_projector_weights_are_mapped_when_vision_is_on(monkeypatch):
    _, name_map = _build(
        monkeypatch,
        _vision_config(),
        projector="mmproj.gguf",
        gguf_names=_PROJECTOR_NAMES,
    )

    assert sorted(name_map) == sorted(_PROJECTOR_NAMES)


def test_missing_projector_fails_instead_of_loading_half_a_model(monkeypatch):
    """A silent partial load is the failure worth preventing here.

    Requesting one quantization of a repo does not bring the projector along,
    since it is quantized separately.  The config still describes a vision tower,
    so vLLM builds one and simply never receives its weights -- which text
    prompts do not reveal, and image prompts reveal only as bad output.
    """
    with pytest.raises(RuntimeError, match="mmproj"):
        _build(monkeypatch, _vision_config(), projector=None)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_vision_config(has_vision=False), id="vision_turned_off"),
        pytest.param(PretrainedConfig(), id="text_only_config"),
    ],
)
def test_projector_weights_are_left_out_when_vision_is_off(monkeypatch, config):
    """Text-only runs must not pick the projector up off the directory.

    The loader resolves the projector from the directory, and downloads one when
    the config carries a ``vision_config``, so it arrives whether or not this run
    wants it.  vLLM builds no vision tower for either config here, so leaving
    those tensors out of the map is what keeps 809 weights from arriving with
    nowhere to go.
    """
    _, name_map = _build(
        monkeypatch, config, projector="mmproj.gguf", gguf_names=_PROJECTOR_NAMES
    )

    assert name_map == {}


# --------------------------------------------------------------------------
# Name mapping
# --------------------------------------------------------------------------
def test_synthesized_qk_norms_are_discarded(monkeypatch, caplog):
    """These are not learned parameters and loading them applies a factor twice.

    The converter synthesizes them as constants from the config's scale factor,
    which vLLM already applies itself.  They are also checked not to be reported
    as unmapped: that warning is for names the mapping failed to cover, and
    listing a deliberate omission there would send the next reader after a
    mapping bug that does not exist.
    """
    names = [f"blk.0.attn_{proj}_norm.weight" for proj in ("q", "k")]

    _, name_map = _build(
        monkeypatch, _vision_config(), projector="mmproj.gguf", gguf_names=names
    )

    assert name_map == {}
    assert "No HF name" not in caplog.text


def test_text_and_vision_projections_are_told_apart(monkeypatch):
    """Both towers call the projection ``attn_q`` in GGUF; the targets differ.

    The text module ``self_attn.q_proj`` ends with the vision module's
    ``attn.q_proj``, so a substring rule alone would rewrite vision names into
    text ones.
    """
    _, name_map = _build(
        monkeypatch,
        _vision_config(),
        projector="mmproj.gguf",
        gguf_names=["blk.0.attn_q.weight", "v.blk.0.attn_q.weight"],
    )

    assert name_map == {
        "blk.0.attn_q.weight": (
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ),
        "v.blk.0.attn_q.weight": "model.vision_tower.layers.0.attn.q_proj.weight",
    }


# --------------------------------------------------------------------------
# Declaring the dequantized modules unquantized
# --------------------------------------------------------------------------
def test_declaration_cannot_be_read_before_the_name_map_is_built():
    """It is derived from the name map, so reading it early would under-declare.

    The loader builds the map first today.  Were the declaration assigned during
    that call rather than derived from it, a reordering on the loader's side
    would quietly return an empty list -- and an under-declared fused layer
    allocates packed buffers that no incoming tensor fills.
    """
    adapter = MuseGlimmerGGUFAdapter()

    with pytest.raises(RuntimeError, match="before build_name_map"):
        _ = adapter.extra_unquantized_modules


def _declaration_as_vllm_sees_it(monkeypatch) -> tuple[list[str], dict[str, list[str]]]:
    """What the adapter declared, in vLLM's namespace.

    Taken from the adapter rather than assembled here, so that dropping either
    half of the declaration shows up as a failure instead of leaving the test
    agreeing with itself.

    Three steps at runtime: the loader records HF names, vLLM rewrites them with
    ``apply_vllm_mapper``, and the prefix ``get_quant_method`` receives is a vLLM
    name too.  Nearly every name changes on the way through -- the text tower
    loses ``model.language_model.`` and ``model.vision_tower.`` becomes
    ``vision_encoder.`` -- so a declaration that only agrees with itself in HF
    names says nothing about what happens at load time.

    The loader's own half of the declaration -- the modules it derives from the
    tensors a GGUF stores unquantized -- is left out on purpose, which is what
    passing only synthetic names achieves.  Real checkpoints carry attention
    biases among those, whose full paths cover the fused ``qkv_proj`` for free;
    relying on that is what kept ``o_proj`` looking covered while it was not.
    """
    from vllm.model_executor.models.muse_glimmer import MuseGlimmerForCausalLM

    from vllm_gguf_plugin.quantization.config import GGUFConfig

    gguf_names = [
        f"v.blk.{layer}.{leaf}.weight"
        for layer in VISION_LAYERS
        for leaf in _VISION_LEAVES
    ] + ["token_embd.weight", "mm.2.weight"]

    adapter, _ = _build(
        monkeypatch,
        _vision_config(),
        projector="mmproj.gguf",
        gguf_names=gguf_names,
    )

    config = GGUFConfig(unquantized_modules=list(adapter.extra_unquantized_modules))
    config.apply_vllm_mapper(
        MuseGlimmerForCausalLM.hf_to_vllm_mapper.get_unstacked_mapper()
    )
    return config.unquantized_modules, MuseGlimmerForCausalLM.packed_modules_mapping


@pytest.mark.parametrize("layer", VISION_LAYERS)
@pytest.mark.parametrize("projection", ["qkv_proj", "o_proj"])
def test_dequantized_vision_layers_are_declared_unquantized(
    monkeypatch, layer, projection
):
    """Whatever is dequantized has to be judged unquantized -- one decision.

    The adapter hands these modules plain float ``weight``.  If
    ``get_quant_method`` thinks they are quantized it allocates ``qweight``
    buffers that no incoming tensor fills.

    Nothing reaches that branch today, because vLLM builds these layers without
    forwarding ``quant_config`` -- which is what forced the dequantization in the
    first place.  So this guards the day vLLM forwards it, when a missing
    declaration turns into a hard error.

    The two projections are covered by different halves of the declaration, which
    is why both are checked: ``qkv_proj`` is fused, and the fused branch of
    ``is_layer_skipped_gguf`` asks whether a declared name *contains* the shard's
    full path, so only a full-depth name matches it -- while ``o_proj`` is
    renamed on vLLM's way in (from ``attn.proj``) and is reached only by the
    coarse prefix.
    """
    from vllm_gguf_plugin.quantization.utils import is_layer_skipped_gguf

    declared, fused_mapping = _declaration_as_vllm_sees_it(monkeypatch)
    prefix = f"vision_encoder.transformer.{layer}.attn.{projection}"

    assert is_layer_skipped_gguf(prefix, declared, fused_mapping), (
        f"{prefix} is handed a dequantized float weight but is not declared "
        "unquantized; once vLLM forwards quant_config it would allocate qweight "
        "buffers nothing fills"
    )


def test_embedding_is_declared_unquantized(monkeypatch):
    from vllm_gguf_plugin.quantization.utils import is_layer_skipped_gguf

    declared, fused_mapping = _declaration_as_vllm_sees_it(monkeypatch)

    assert is_layer_skipped_gguf("model.embed_tokens", declared, fused_mapping)


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ],
)
def test_text_layers_stay_quantized(monkeypatch, prefix):
    """The declaration is matched by substring, which is easy to overreach with.

    One declared name that is too short also marks the text tower unquantized,
    and vLLM would then build plain float weights for layers the adapter feeds
    packed bytes -- losing the whole 30B backbone.
    """
    from vllm_gguf_plugin.quantization.utils import is_layer_skipped_gguf

    declared, fused_mapping = _declaration_as_vllm_sees_it(monkeypatch)

    assert not is_layer_skipped_gguf(prefix, declared, fused_mapping)


# --------------------------------------------------------------------------
# Patch embedding
# --------------------------------------------------------------------------
@pytest.mark.parametrize("patch_temporal", [1, 2, 4])
def test_patch_embedding_split_adds_back_to_the_stored_sum(patch_temporal):
    """Only the sum survives conversion, and still images depend on just the sum.

    The encoder feeds a still image to every time step by expanding the same
    patch, so any split that adds back to the stored sum is equivalent for
    images -- which is what makes the image path exact rather than approximate.
    """
    out_channels = 3
    summed = torch.randn(out_channels, 5)

    blocks = reconstruct_patch_embedding(summed, patch_temporal)

    assert blocks.shape == (out_channels, 5 * patch_temporal)
    torch.testing.assert_close(
        blocks.reshape(out_channels, patch_temporal, 5).sum(dim=1), summed
    )
