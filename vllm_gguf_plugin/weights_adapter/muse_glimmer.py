# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight adapter for Muse Glimmer GGUF checkpoints.

Two of the conversions this adapter performs are invisible if you get them
wrong: the weights load without complaint and the model generates fluent-looking
but wrong output.  Both are called out where they are implemented.

The first is the Q/K row layout.  The conversion script that produces these GGUF
files re-lays out the Q and K projections from the half-split ("NEOX") ordering
that HF checkpoints use into llama.cpp's interleaved ordering.  vLLM's Muse
Glimmer implementation hardcodes NEOX rotary embeddings, so the adapter has to
undo that re-layout.

The re-layout only reorders rows of the output dimension; it never changes a
value.  Because GGUF splits quantized super-blocks along the *input* dimension,
every output row is a self-contained run of quantized bytes, so the inverse can
be applied directly to the packed ``qweight`` bytes instead of dequantizing
first.  ``test_muse_glimmer_gguf.py`` pins that equivalence bit-for-bit.

The second is the norm offset, which applies to the per-layer norms but not to
the final one -- see :data:`NORM_OFFSET_SUFFIXES`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

MUSE_GLIMMER_MODEL_TYPES = ("muse_glimmer", "muse_glimmer_text")

# Neither entry can be looked up in the auto-model mappings: the text-only config
# is in no mapping at all, and while the multimodal one is registered for
# image-text-to-text, the config parser only consults the causal-LM mapping.
# Getting these two crossed does not raise -- pointing ``muse_glimmer`` at the
# causal-LM class quietly builds a text-only model that loads and generates.
MUSE_GLIMMER_ARCHITECTURES = {
    "muse_glimmer": "MuseGlimmerForConditionalGeneration",
    "muse_glimmer_text": "MuseGlimmerForCausalLM",
}


def interleaved_to_neox_row_index(
    num_heads: int,
    head_dim: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Row gather index turning llama.cpp interleaved order back into NEOX.

    The forward direction (what the conversion script did) writes NEOX row ``i``
    to interleaved row ``2i`` for the first half of a head and to ``2i + 1`` for
    the second half.  Inverting it means gathering the even rows first, then the
    odd rows, within each head::

        [0, 2, 4, ..., head_dim - 2, 1, 3, 5, ..., head_dim - 1]

    This permutation is **not** self-inverse for ``head_dim >= 8``, so applying
    the forward helper twice does not undo it.  ``head_dim == 4`` happens to be
    self-inverse, which makes it a misleading size to test against.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    within_head = torch.cat(
        (
            torch.arange(0, head_dim, 2, device=device),
            torch.arange(1, head_dim, 2, device=device),
        )
    )
    head_offsets = torch.arange(num_heads, device=device) * head_dim
    return (head_offsets[:, None] + within_head[None, :]).reshape(-1)


def neox_to_interleaved_row_index(
    num_heads: int,
    head_dim: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Row gather index for the forward direction, for tests and debugging."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    half = head_dim // 2
    within_head = torch.empty(head_dim, dtype=torch.int64, device=device)
    within_head[0::2] = torch.arange(0, half, device=device)
    within_head[1::2] = torch.arange(half, head_dim, device=device)
    head_offsets = torch.arange(num_heads, device=device) * head_dim
    return (head_offsets[:, None] + within_head[None, :]).reshape(-1)


def undo_rope_interleave(
    tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Restore NEOX row order on a Q or K tensor.

    Works on packed ``qweight`` bytes (``[num_rows, row_bytes]`` uint8), on
    dequantized weights (``[num_rows, input_dim]``) and on 1-D biases alike:
    all three only need the leading dimension permuted.
    """
    num_rows = tensor.shape[0]
    expected = num_heads * head_dim
    if num_rows != expected:
        raise ValueError(
            f"expected {expected} rows for {num_heads} heads of {head_dim}, "
            f"got {num_rows}"
        )

    index = interleaved_to_neox_row_index(num_heads, head_dim, tensor.device)
    # index_select copies, so the result stays contiguous even though the gather
    # is non-monotonic; downstream TP sharding and QKV fusion rely on that.
    return tensor.index_select(0, index)


TEXT_LAYER_PREFIX = "model.language_model.layers."
VISION_LAYER_PREFIX = "model.vision_tower.layers."
PATCH_EMBEDDING = "model.vision_tower.patch_embedder.patch_embedding.weight"

# llama.cpp's converter folds the ``1 +`` from this architecture's norm into the
# stored weight, so it has to be taken back out.  Only the per-layer norms carry
# the offset; the final norm is stored as-is.  Its weights sit close to 1.0, so
# subtracting from it as well would leave values close to 0 and scale the last
# hidden state away entirely.  Hence an explicit list of the four per-layer norms
# rather than a test on the ``norm.weight`` suffix, which the final norm matches
# too.
NORM_OFFSET_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
)

# The Q/K norms in these files are not learned parameters: the converter
# synthesizes them as constant tensors from the config's scale factor.  vLLM
# applies that factor itself from the config, so loading them would apply it
# twice.  Dropping them is what makes the rest of the mapping a bijection onto
# the HF checkpoint.
#
# Skipped before the name map is built rather than mapped to ``None``, so that
# they do not land in the list of tensors reported as unmapped: that list is for
# names the mapping failed to cover, and these are left out on purpose.
SYNTHETIC_QK_NORM_SUBSTRINGS = ("attn_q_norm.", "attn_k_norm.")

# Modules that have to be handed plain weights rather than packed bytes, for two
# unrelated reasons -- both on the vLLM side, neither fixable from here:
#
#   * ``embed_tokens`` is built without forwarding ``quant_config``, so the layer
#     only ever owns a plain ``weight`` and rejects packed bytes outright ("no
#     module or parameter named model.embed_tokens.qweight_type").
#   * the vision tower's linear layers are either plain ``nn.Linear`` or are also
#     built without ``quant_config``, so they have nowhere to put packed bytes.
#
# Unpacking costs roughly 1.9 GB for the embedding and 2.5 GB for the vision
# tower.  Both shrink back once vLLM forwards ``quant_config`` in those places.
#
# Every prefix here is also declared unquantized by
# :attr:`MuseGlimmerGGUFAdapter.extra_unquantized_modules`.  That is what keeps
# the two decisions from drifting apart: whichever way vLLM builds these layers,
# ``get_quant_method`` sees them as unquantized and expects the plain ``weight``
# this adapter hands over.  Without the declaration, vLLM forwarding
# ``quant_config`` here would make the layer allocate ``qweight`` buffers that no
# incoming tensor matches.
DEQUANTIZED_MODULE_PREFIXES = (
    "model.language_model.embed_tokens",
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection",
)

_VISION_GGUF_PREFIXES = ("v.", "mm.")


def has_vision(config: PretrainedConfig) -> bool:
    """Whether the vision tower is part of this model.

    Deliberately the same rule vLLM's Muse Glimmer implementation applies, so
    that the weights this adapter produces and the modules vLLM builds are
    decided by one predicate rather than two.  Reading it off the directory
    instead -- taking the vision tower to be present whenever a projector file
    happens to be resolvable -- decides it a second time, and the two answers
    disagree as soon as a config turns vision off.

    The shipped ``config.json`` carries no ``has_vision`` key, so the fallback is
    the usual path; the attribute is there for turning vision off explicitly.
    """
    configured = getattr(config, "has_vision", None)
    if configured is not None:
        return bool(configured)
    return hasattr(config, "vision_config")


def _module_of(param: str) -> str:
    return param.rsplit(".", 1)[0] if param.endswith((".weight", ".bias")) else param


def dequantized_module_names(mapped_names: Iterable[str]) -> set[str]:
    """Full-depth module names for every weight :meth:`transform_weights` unpacks.

    Takes names that have already been mapped into vLLM's namespace, so the rule
    can be tested against a handful of synthetic names.

    Declaring only the prefixes above would read as equivalent and is not.
    ``is_layer_skipped_gguf`` handles a fused layer by asking whether some
    declared name *contains* the full path of each shard, so a prefix -- being
    shorter than the path it is a prefix of -- never matches one.  The vision
    tower's attention is fused into ``qkv_proj``, and these full paths are what
    covers it.
    """
    return {
        _module_of(name)
        for name in mapped_names
        if name.startswith(DEQUANTIZED_MODULE_PREFIXES)
    }


def dequantize_packed_rows(
    qweight: torch.Tensor,
    quant_type: int,
    dtype: torch.dtype,
    rows_per_chunk: int = 4096,
) -> torch.Tensor:
    """Unpack GGUF bytes straight into *dtype*.

    ``gguf.quants.dequantize`` hands back float32, which for a vocabulary this
    size is several gigabytes on top of the result.  Rows are self-contained --
    GGUF blocks never straddle one -- so converting a block of rows at a time
    keeps the transient down to the block rather than the whole tensor.
    """
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
    from gguf.quants import dequantize

    ggml_type = GGMLQuantizationType(quant_type)
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]
    num_rows, row_bytes = qweight.shape
    num_cols = row_bytes // type_size * block_size

    packed = qweight.numpy()
    out = torch.empty((num_rows, num_cols), dtype=dtype)
    for start in range(0, num_rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, num_rows)
        values = dequantize(packed[start:stop], ggml_type)
        out[start:stop] = torch.from_numpy(values.reshape(stop - start, num_cols)).to(
            dtype
        )
    return out


def reconstruct_patch_embedding(
    summed: torch.Tensor,
    patch_temporal: int,
) -> torch.Tensor:
    """Undo the converter's sum over the patch embedding's time axis.

    The checkpoint keeps one weight block per time step, laid out with time as
    the outermost axis of the flattened patch dimension; the converter stores
    only their sum, so the individual blocks are gone for good.

    Splitting the sum evenly is exact for still images and is also the closest
    available guess at the original blocks.  The encoder feeds a still image to
    every time step by expanding the same patch, so the output only ever depends
    on the sum -- any split reproducing it is equivalent.  And the reference
    blocks turn out to be near-copies of each other (equal mean magnitude to
    within 0.1%, correlation 0.99), so half the sum is close to each of them.

    Video feeds distinct frames per time step, which does depend on the blocks
    individually, and is rejected for that reason.
    """
    flat = summed.reshape(summed.shape[0], -1)
    return flat.div(patch_temporal).repeat(1, patch_temporal)


# The vision tower cannot reuse the text substring rules: GGUF calls both
# projections ``attn_q``, while the checkpoint calls the text one
# ``self_attn.q_proj`` and the vision one ``attn.q_proj``.  Anchored regexes keep
# the two apart -- they run before the substring pass, so a rewritten vision name
# no longer matches any text rule.
_VISION_BLOCK_LEAVES = {
    "attn_q": "attn.q_proj",
    "attn_k": "attn.k_proj",
    "attn_v": "attn.v_proj",
    "attn_out": "attn.proj",
    # Straight, not crossed: GGUF ``ffn_up`` is (8960, 1536), which is ``fc1``.
    "ffn_up": "mlp.fc1",
    "ffn_down": "mlp.fc2",
    # These carry no offset, unlike the text norms -- they are stored as-is, so
    # they must not appear in NORM_OFFSET_SUFFIXES.
    "ln1": "norm1",
    "ln2": "norm2",
}


def build_muse_glimmer_mapper() -> WeightsMapper:
    orig_to_new_regex: dict[re.Pattern, str | None] = {
        re.compile(rf"^v\.blk\.(\d+)\.{gguf}\."): (f"{VISION_LAYER_PREFIX}\\1.{hf}.")
        for gguf, hf in _VISION_BLOCK_LEAVES.items()
    }
    orig_to_new_substr: dict[str, str | None] = {
        "attn_norm.": "input_layernorm.",
        "post_attention_norm.": "post_attention_layernorm.",
        "ffn_norm.": "pre_feedforward_layernorm.",
        "post_ffw_norm.": "post_feedforward_layernorm.",
        "attn_q.": "self_attn.q_proj.",
        "attn_k.": "self_attn.k_proj.",
        "attn_v.": "self_attn.v_proj.",
        "attn_output.": "self_attn.o_proj.",
        "attn_gate.": "self_attn.gate_proj.",
        "ffn_gate.": "mlp.gate_proj.",
        "ffn_up.": "mlp.up_proj.",
        "ffn_down.": "mlp.down_proj.",
    }
    orig_to_new_prefix: dict[str, str | None] = {
        "token_embd.": "model.language_model.embed_tokens.",
        "blk.": TEXT_LAYER_PREFIX,
        "output_norm.": "model.language_model.norm.",
        "output.": "lm_head.",
        "v.patch_embd.": "model.vision_tower.patch_embedder.patch_embedding.",
        "v.position_embd.": (
            "model.vision_tower.patch_embedder.position_embedding_table."
        ),
        "v.pre_ln.": "model.vision_tower.ln_pre.",
        "v.post_ln.": "model.vision_tower.ln_post.",
        # Shapes pin these three down: (4096, 6144), (4096, 4096) and
        # (6656, 4096) are all distinct and each matches exactly one target.
        "mm.0.": "model.vision_adapter.fc1.",
        "mm.1.": "model.vision_adapter.fc2.",
        "mm.2.": "model.vision_projection.",
    }

    return WeightsMapper(
        orig_to_new_regex=orig_to_new_regex,
        orig_to_new_prefix=orig_to_new_prefix,
        orig_to_new_substr=orig_to_new_substr,
    )


class MuseGlimmerGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for Muse Glimmer GGUF models."""

    # Only the sum of the patch embedding's per-time-step blocks survives the
    # conversion; see reconstruct_patch_embedding. Still images depend on that
    # sum alone, so they are exact, while video depends on the blocks
    # individually and would run about 7% off in the one channel that carries
    # frame-to-frame motion -- wrong in a way that still looks plausible.
    UNSUPPORTED_MODALITIES = ("video",)

    def __init__(self) -> None:
        self._name_map: dict[str, str] | None = None

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in MUSE_GLIMMER_MODEL_TYPES

    @classmethod
    def architecture(cls, config) -> str | None:
        return MUSE_GLIMMER_ARCHITECTURES.get(config.model_type)

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        patched = maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )
        if has_vision(patched) and files.mm_proj is None:
            raise RuntimeError(
                "The vision tower needs the multimodal projector, and no mmproj "
                f"file could be resolved for {files.primary_backbone}.  "
                "Requesting one quantization does not bring it along, since the "
                "projector is quantized separately.  Place *mmproj*.gguf beside "
                "the backbone, pass model_loader_extra_config={'mm_proj': ...}, "
                "or set has_vision=false to run text-only.  Continuing without "
                "it would leave the vision weights unloaded, which text prompts "
                "would not reveal."
            )
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        mapper = build_muse_glimmer_mapper()
        vision = has_vision(model_config.hf_config)

        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            if any(substr in name for substr in SYNTHETIC_QK_NORM_SUBSTRINGS):
                continue
            # The projector is resolved by the loader from the directory, so it
            # can turn up even for a config that runs text-only.  Leaving its
            # tensors out of the map is what keeps them from being loaded.
            if not vision and name.startswith(_VISION_GGUF_PREFIXES):
                continue
            mapped = mapper.apply_list([name])
            if not mapped or mapped[0] == name:
                unmapped.append(name)
            else:
                name_map[name] = mapped[0]

        if unmapped:
            logger.warning(
                "No HF name for %d Muse Glimmer GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        self._name_map = name_map
        return name_map

    @property
    def extra_unquantized_modules(self) -> tuple[str, ...]:
        """Modules this adapter unpacks, which the GGUF stores quantized.

        The loader derives its own list from the tensors a GGUF stores
        unquantized, which these are not -- they are quantized in the file and
        dequantized here -- so they have to be declared separately.

        Derived from the name map rather than assigned during
        :meth:`build_name_map`, so that reading it too early fails loudly.  Were
        it assigned, a reordering on the loader's side would silently under-
        declare, and an under-declared fused layer allocates packed buffers that
        no incoming tensor matches.
        """
        if self._name_map is None:
            raise RuntimeError(
                "extra_unquantized_modules was read before build_name_map; the "
                "declaration is derived from the name map"
            )
        # Both forms are needed.  The full paths are what a fused layer matches
        # against; the prefixes cover the layers vLLM's own mapper renames on the
        # way in (``attn.proj`` becomes ``attn.o_proj``, and a full path recorded
        # under the old name stops matching).
        return tuple(
            sorted(
                dequantized_module_names(self._name_map.values())
                | set(DEQUANTIZED_MODULE_PREFIXES)
            )
        )

    def _rope_layout(
        self,
        name: str,
        config: PretrainedConfig,
    ) -> tuple[int, int] | None:
        """``(num_heads, head_dim)`` to un-interleave *name* with, else ``None``.

        Only Q and K are rotated, so only they were re-laid out.  V shares their
        shape and quantization, so permuting it as well is not caught by any
        shape check -- it just corrupts every value it touches.

        The text and vision towers have to be told apart before the projection
        name is inspected, because the text module ``self_attn.q_proj`` also ends
        with the vision module's ``attn.q_proj``.
        """
        # ``.qweight_type`` is a scalar tag rather than a weight; it shares the
        # module prefix, so match on the payload suffixes instead.
        if not name.endswith((".qweight", ".weight", ".bias")):
            return None
        module = name.rsplit(".", 1)[0]

        if name.startswith(TEXT_LAYER_PREFIX):
            text_config = config.get_text_config()
            if module.endswith("self_attn.q_proj"):
                return text_config.num_attention_heads, text_config.head_dim
            if module.endswith("self_attn.k_proj"):
                return text_config.num_key_value_heads, text_config.head_dim
            return None

        if name.startswith(VISION_LAYER_PREFIX):
            if not module.endswith(("attn.q_proj", "attn.k_proj")):
                return None
            # The vision tower is not grouped-query, so Q and K share a count.
            vision_config = config.vision_config
            num_heads = vision_config.num_attention_heads
            return num_heads, vision_config.hidden_size // num_heads

        return None

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Transform mapped GGUF weights into HF-style weights."""
        config = model_config.hf_config
        dtype = model_config.dtype

        # The iterator emits a module's ``qweight_type`` immediately before its
        # ``qweight``, so a single slot is enough to rejoin the two.
        quant_types: dict[str, int] = {}
        for name, weight in weights:
            module, _, leaf = name.rpartition(".")
            if leaf in ("qweight", "qweight_type") and module.startswith(
                DEQUANTIZED_MODULE_PREFIXES
            ):
                if leaf == "qweight_type":
                    quant_types[module] = int(weight.item())
                    continue
                name = f"{module}.weight"
                weight = dequantize_packed_rows(weight, quant_types.pop(module), dtype)

            if name == PATCH_EMBEDDING:
                weight = reconstruct_patch_embedding(
                    weight, config.vision_config.patch_temporal
                )
            elif name.endswith(NORM_OFFSET_SUFFIXES):
                weight = weight - 1
            elif (layout := self._rope_layout(name, config)) is not None:
                weight = undo_rope_interleave(weight, *layout)
            yield name, weight
