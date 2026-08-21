# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconcile GGUF tensor names against a target model's parameter names.

Adding an architecture to this plugin starts with the same three questions:
which GGUF tensors map to which parameters, what is left over on each side,
and where do shapes disagree (i.e. where a transform is needed). Answering
them by hand means diffing two lists of ~1000 names.

This reads GGUF headers and instantiates the target on meta tensors, so it
needs no GPU and loads no weights.

    python -m vllm_gguf_plugin.tools.gguf_map \\
        --gguf unsloth/Qwen3.5-27B-GGUF:Qwen3.5-27B-Q4_K_M.gguf \\
        --target-model Qwen/Qwen3.5-27B [--target vllm] [--emit-adapter]

``--gguf`` takes a local path, an ``https://`` URL or ``repo_id:filename``;
the latter two read the header over an HTTP range request.

``--target hf`` maps to ``transformers`` state-dict names. ``--target vllm``
maps to the names vLLM's ``load_weights`` accepts, which differ for fused and
MoE layers -- vLLM stacks experts into ``w13_weight`` but its loader consumes
per-expert ``experts.{i}.gate_proj.weight``.

For a sharded GGUF, point at the first shard: each shard carries its own
header, and shard 1 holds every distinct name pattern because layers repeat.

What it cannot see: value-level transforms (e.g. a checkpoint storing
``-exp(A_log)``) produce neither a name nor a shape mismatch. Confirm numerics
separately.
"""

from __future__ import annotations

import argparse
import difflib
import re
import struct
import urllib.request

import gguf
import torch

# GGUF metadata scalar type -> byte width.
_KV_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_KV_STRING = 8
_KV_ARRAY = 9

DEFAULT_HEADER_BYTES = 64 << 20


class GGUFHeaderTruncated(Exception):
    """Fetched prefix was too short to contain the whole GGUF header."""


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise GGUFHeaderTruncated(
                f"header exceeds the {len(self.data)} bytes fetched; "
                f"retry with a larger --header-bytes"
            )
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")

    def skip_value(self, kind: int) -> None:
        if kind == _KV_STRING:
            self.string()
        elif kind == _KV_ARRAY:
            elem, count = self.u32(), self.u64()
            if elem == _KV_STRING:
                # Length-prefixed, so each element must be walked.
                for _ in range(count):
                    self.string()
            elif elem == _KV_ARRAY:
                raise ValueError("nested GGUF arrays are not supported")
            else:
                self.take(_KV_WIDTH[elem] * count)
        else:
            self.take(_KV_WIDTH[kind])


def parse_gguf_header(data: bytes) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape, from a GGUF header prefix.

    Shapes are returned in torch order; ggml stores dimensions
    fastest-varying first.
    """
    cur = _Cursor(data)
    if cur.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    cur.u32()  # format version
    n_tensors, n_kv = cur.u64(), cur.u64()
    for _ in range(n_kv):
        cur.string()
        cur.skip_value(cur.u32())
    tensors = {}
    for _ in range(n_tensors):
        name = cur.string()
        dims = [cur.u64() for _ in range(cur.u32())]
        cur.u32()  # ggml dtype
        cur.u64()  # data offset
        tensors[name] = tuple(reversed(dims))
    return tensors


def collect_gguf_remote(
    ref: str, header_bytes: int = DEFAULT_HEADER_BYTES
) -> dict[str, tuple[int, ...]]:
    """Read tensor metadata over HTTP range, without downloading weights.

    *ref* is a URL or ``repo_id:filename``. A few MB is enough to reconcile a
    model whose weights are hundreds of GB.
    """
    if ref.startswith(("http://", "https://")):
        url = ref
    else:
        repo_id, filename = ref.rsplit(":", 1)
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    request = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{header_bytes - 1}"}
    )
    with urllib.request.urlopen(request) as response:
        return parse_gguf_header(response.read())


def is_remote_ref(ref: str) -> bool:
    if ref.startswith(("http://", "https://")):
        return True
    return ref.count(":") == 1 and not ref.startswith(("/", "."))


def collect_gguf(ref: str, header_bytes: int = DEFAULT_HEADER_BYTES):
    if is_remote_ref(ref):
        return collect_gguf_remote(ref, header_bytes)
    reader = gguf.GGUFReader(ref)
    return {
        tensor.name: tuple(int(d) for d in reversed(tensor.shape))
        for tensor in reader.tensors
    }


def collect_hf(model_ref: str, trust_remote_code: bool = False):
    """``transformers`` state-dict names and shapes, via a meta instantiation."""
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # pragma: no cover - older transformers
        AutoModelForImageTextToText = None

    config = AutoConfig.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    multimodal = getattr(config, "vision_config", None) is not None
    auto_cls = AutoModelForCausalLM
    if multimodal and AutoModelForImageTextToText is not None:
        auto_cls = AutoModelForImageTextToText
    with torch.device("meta"):
        model = auto_cls.from_config(config, trust_remote_code=trust_remote_code)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def collect_vllm(
    model_ref: str,
    trust_remote_code: bool = False,
    dtype: str = "bfloat16",
    port: int = 29591,
):
    """Checkpoint names vLLM's ``load_weights`` accepts, with their shapes.

    vLLM fuses linears (``qkv_proj``, ``gate_up_proj``) and stacks MoE experts
    (``w13_weight``), but its loader still consumes the unfused, per-expert
    names and splits them on the way in. Reconstruct that contract from the
    instantiated module tree rather than from the parameter names.
    """
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    model_config = ModelConfig(
        model=model_ref,
        tokenizer=model_ref,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        seed=0,
        max_model_len=128,
    )
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        with set_default_torch_dtype(getattr(torch, dtype)), torch.device("meta"):
            model = initialize_model(vllm_config=vllm_config)
    return expand_vllm_params(model)


def expand_vllm_params(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    """Expand a vLLM module tree into the checkpoint names it accepts."""
    packed = getattr(type(model), "packed_modules_mapping", None) or {}
    names: dict[str, tuple[int, ...]] = {}

    for path, module in model.named_modules():
        own = dict(module.named_parameters(recurse=False))
        if not own:
            continue
        kind = type(module).__name__
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        leaf = path.rsplit(".", 1)[-1]

        if kind in ("FusedMoE", "RoutedExperts") and hasattr(
            module, "global_num_experts"
        ):
            base = path.removesuffix(".routed_experts")
            if base.rsplit(".", 1)[-1] != "experts":
                base += ".experts"
            inter = int(module.intermediate_size_per_partition)
            hidden = int(module.hidden_size)
            for expert in range(int(module.global_num_experts)):
                names[f"{base}.{expert}.gate_proj.weight"] = (inter, hidden)
                names[f"{base}.{expert}.up_proj.weight"] = (inter, hidden)
                names[f"{base}.{expert}.down_proj.weight"] = (hidden, inter)
            continue

        weight = own.get("weight")
        in_features = (
            int(weight.shape[1]) if weight is not None and weight.dim() > 1 else None
        )
        sizes = getattr(module, "output_sizes", None)
        parts = packed.get(leaf)
        if sizes and parts and len(sizes) == len(parts) and in_features:
            for size, part in zip(sizes, parts):
                names[f"{parent}.{part}.weight"] = (int(size), in_features)
            continue
        if kind == "QKVParallelLinear" and hasattr(module, "num_heads") and in_features:
            head = int(module.head_size)
            for part, count in (
                ("q_proj", int(module.num_heads)),
                ("k_proj", int(module.num_kv_heads)),
                ("v_proj", int(module.num_kv_heads)),
            ):
                names[f"{parent}.{part}.weight"] = (count * head, in_features)
            continue

        for param_name, param in own.items():
            names[f"{path}.{param_name}" if path else param_name] = tuple(param.shape)
    return names


# ---------------------------------------------------------------------------
# templating
# ---------------------------------------------------------------------------


def templatize(name: str) -> tuple[str, tuple[int, ...]]:
    """``blk.3.attn_q.weight`` -> ``blk.{i}.attn_q.weight``, ``(3,)``."""
    parts, out, indices = name.split("."), [], []
    for part in parts:
        if part.isdigit():
            out.append("{i}")
            indices.append(int(part))
        else:
            out.append(part)
    return ".".join(out), tuple(indices)


class Template:
    """One repeated name pattern, with the index values it occurs at.

    ``slots`` is decisive for hybrid stacks: Qwen3.5 interleaves attention and
    gated-delta-net layers, so ``attn_output`` occupies layers [3, 7, 11, ...]
    and ``ssm_out`` [0, 1, 2, 4, ...]. Those sets are disjoint and line up with
    ``self_attn.o_proj`` vs ``linear_attn.out_proj`` -- which identical shapes
    and equally plausible names cannot distinguish.
    """

    __slots__ = ("count", "example", "shape", "slots")

    def __init__(self, shape: tuple[int, ...], example: str, n_slots: int) -> None:
        self.shape = shape
        self.example = example
        self.count = 0
        self.slots: list[set[int]] = [set() for _ in range(n_slots)]

    @property
    def layers(self) -> frozenset[int]:
        return frozenset(self.slots[0]) if self.slots else frozenset()

    def add(self, indices: tuple[int, ...]) -> None:
        self.count += 1
        for slot, value in enumerate(indices):
            if slot < len(self.slots):
                self.slots[slot].add(value)


def group(names: dict[str, tuple[int, ...]]) -> dict[str, Template]:
    """Collapse per-layer repetition: ~1000 names become ~25 templates."""
    grouped: dict[str, Template] = {}
    for name, shape in names.items():
        key, indices = templatize(name)
        if key not in grouped:
            grouped[key] = Template(shape, name, len(indices))
        grouped[key].add(indices)
    return grouped


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

# Terms that denote the same module on either side. Used only to rank
# candidates, never to assert a match alone.
SYNONYMS: tuple[frozenset[str], ...] = tuple(
    frozenset(group)
    for group in (
        ("attn_q", "q_proj"),
        ("attn_k", "k_proj"),
        ("attn_v", "v_proj"),
        ("attn_output", "attn_out", "o_proj"),
        ("ssm_out", "out_proj"),
        ("attn_qkv", "qkv_proj", "qkv"),
        ("ffn_gate", "gate_proj"),
        ("ffn_up", "up_proj"),
        ("ffn_down", "down_proj"),
        ("ffn_gate_exps", "gate_proj"),
        ("ffn_up_exps", "up_proj"),
        ("ffn_down_exps", "down_proj"),
        ("ffn_gate_inp", "gate"),
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("attn_q_norm", "q_norm"),
        ("attn_k_norm", "k_norm"),
        ("token_embd", "embed_tokens"),
        ("output_norm", "norm"),
        ("output", "lm_head"),
        ("blk", "layers"),
        ("ssm_conv1d", "conv1d"),
        ("ssm_norm", "norm"),
        ("ssm_dt", "dt_bias"),
        ("ssm_a", "A_log"),
        ("ssm_alpha", "in_proj_a"),
        ("ssm_beta", "in_proj_b"),
        ("exp_probs_b", "e_score_correction_bias"),
    )
)

_SPLIT_RE = re.compile(r"[.\-_]|(?<=[a-z])(?=[A-Z])")


def _tokens(name: str) -> set[str]:
    return {t for t in _SPLIT_RE.split(name) if t and t != "{i}"}


def synonym_score(left: str, right: str) -> float:
    lt, rt = _tokens(left), _tokens(right)
    ls, rs = set(left.split(".")), set(right.split("."))
    score = 0.0
    for group_ in SYNONYMS:
        if (lt & group_) and (rt & group_):
            score += 1.0
        if (ls & group_) and (rs & group_):
            score += 1.0
    return score


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def shape_affinity(src: tuple[int, ...], dst: tuple[int, ...]) -> float | None:
    """Score a shape pairing, or ``None`` when the two cannot be reconciled.

    ``None`` disqualifies the candidate outright. Without that, a strong name
    signal (``blk`` vs ``layers``) plus identical layer occupancy can carry a
    pairing whose shapes have nothing in common, which surfaces as a bogus
    "transform needed" rather than an honest unmatched tensor.
    """
    if src == dst:
        return 3.0
    if sorted(src) == sorted(dst):
        return 1.5  # transposed or permuted
    if _numel(src) == _numel(dst):
        return 1.0  # reshape-compatible, e.g. conv1d 2-D vs 3-D
    if len(src) == len(dst) + 1 and src[1:] == dst:
        return 0.5  # stacked: leading dim is an expert/group count
    if len(src) == len(dst):
        differing = [d for d in range(len(src)) if src[d] != dst[d]]
        if len(differing) == 1:
            return 0.0  # one axis differs: possible concat partner
    return None


def score(source: str, src: Template, target: str, dst: Template) -> float:
    """Rank one candidate pairing. Higher is better; ``-inf`` disqualifies."""
    affinity = shape_affinity(src.shape, dst.shape)
    if affinity is None:
        return float("-inf")
    value = affinity

    if src.layers and dst.layers:
        if src.layers == dst.layers:
            value += 4.0
        elif src.layers & dst.layers:
            overlap = len(src.layers & dst.layers) / len(src.layers | dst.layers)
            value += 4.0 * overlap
        else:
            value -= 4.0  # disjoint: almost certainly a different module

    if source.count("{i}") == target.count("{i}"):
        value += 0.5
    value += 2.0 * synonym_score(source, target)
    value += difflib.SequenceMatcher(None, source, target).ratio()
    src_suffix = source.rsplit(".", 1)[-1]
    if src_suffix == target.rsplit(".", 1)[-1] and src_suffix in ("weight", "bias"):
        value += 0.3
    return value


class Match:
    __slots__ = ("how", "note", "target")

    def __init__(self, target: str, how: str, note: str | None = None) -> None:
        self.target = target
        self.how = how
        self.note = note


class Reconciliation:
    __slots__ = (
        "ambiguous",
        "fusions",
        "matched",
        "unmatched_source",
        "unmatched_target",
    )

    def __init__(self) -> None:
        self.matched: dict[str, Match] = {}
        self.ambiguous: dict[str, list[tuple[str, float]]] = {}
        self.fusions: list[tuple[str, str, str, int]] = []
        self.unmatched_source: list[str] = []
        self.unmatched_target: list[str] = []

    @property
    def resolved(self) -> int:
        return len(self.matched) + len(self.fusions)


def reconcile(
    source: dict[str, Template],
    target: dict[str, Template],
    min_score: float = 3.0,
) -> Reconciliation:
    """Assign GGUF templates to target templates, best pairing first.

    Assignment is global rather than per-tensor. Locally, ``attn_output`` and
    ``ssm_out`` both look like ``linear_attn.out_proj``, and whichever is
    considered first takes it; resolving the highest-confidence pairs first
    leaves the loser its correct, uncontested second choice.
    """
    result = Reconciliation()
    ranked: dict[str, list[tuple[float, str]]] = {}
    best: list[tuple[float, str]] = []

    for name, template in sorted(source.items()):
        candidates = sorted(
            (
                (score(name, template, other, other_template), other)
                for other, other_template in target.items()
            ),
            reverse=True,
        )
        ranked[name] = candidates
        if candidates and candidates[0][0] >= min_score:
            best.append((candidates[0][0], name))

    taken: set[str] = set()
    for _, name in sorted(best, reverse=True):
        for value, candidate in ranked[name]:
            if value < min_score:
                break
            if candidate in taken:
                continue
            src_shape, dst_shape = source[name].shape, target[candidate].shape
            note = None
            if src_shape != dst_shape:
                if len(src_shape) == len(dst_shape) + 1 and src_shape[1:] == dst_shape:
                    note = f"SPLIT: unbind axis 0 into {src_shape[0]}"
                else:
                    note = f"shape {src_shape} -> {dst_shape}"
            result.matched[name] = Match(candidate, f"inferred({value:.1f})", note)
            taken.add(candidate)
            free = [(v, c) for v, c in ranked[name] if c not in taken]
            if free and abs(value - free[0][0]) < 0.5:
                result.ambiguous[name] = [(candidate, round(value, 2))] + [
                    (c, round(v, 2)) for v, c in free[:2]
                ]
            break

    result.unmatched_source = sorted(set(source) - set(result.matched))
    result.fusions, result.unmatched_source = detect_fusions(
        result, result.unmatched_source, source, target
    )
    result.unmatched_target = sorted(set(target) - taken)
    return result


def detect_fusions(
    result: Reconciliation,
    unmatched: list[str],
    source: dict[str, Template],
    target: dict[str, Template],
) -> tuple[list[tuple[str, str, str, int]], list[str]]:
    """Recover GGUF tensors that must be concatenated into one target param.

    A 1:1 assignment cannot express this, so it surfaces as one tensor matched
    with a shape mismatch and its sibling unmatched. DeepSeek's
    ``ffn_gate_exps`` and ``ffn_up_exps`` (both ``(E, I, H)``) concatenate into
    ``experts.gate_up_proj`` ``(E, 2I, H)`` on axis 1.
    """
    fusions, remaining = [], []
    for name in unmatched:
        shape = source[name].shape
        found = None
        for other, match in result.matched.items():
            if not match.note:
                continue
            other_shape = source[other].shape
            goal = target[match.target].shape
            if not len(shape) == len(other_shape) == len(goal):
                continue
            for axis in range(len(shape)):
                if any(
                    other_shape[d] != shape[d] for d in range(len(shape)) if d != axis
                ):
                    continue
                combined = list(other_shape)
                combined[axis] += shape[axis]
                if tuple(combined) == goal:
                    found = (other, name, match.target, axis)
                    break
            if found:
                break
        if found:
            fusions.append(found)
        else:
            remaining.append(name)
    return fusions, remaining


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _pin_expert_slot(target: str) -> str:
    """Pin a leftover index slot to 0.

    The layer index is consumed by the ``blk.`` -> ``model.layers.`` prefix
    rule. A second slot is an expert index, which a rename cannot express: the
    GGUF holds one stacked ``(E, ...)`` tensor. Naming slot 0 is what
    :func:`vllm_gguf_plugin.weight_utils.split_stacked_experts` keys on when it
    unbinds that tensor into per-expert weights.
    """
    return target.replace("{i}", "0")


def build_mapper_rules(
    matched: dict[str, Match],
) -> tuple[dict[str, str], dict[str, str]]:
    """Derive ``WeightsMapper`` prefix and substring rules from matches."""
    prefix: dict[str, str] = {}
    substr: dict[str, str] = {}
    for source, match in sorted(matched.items()):
        src_parts, dst_parts = source.split("."), match.target.split(".")
        if "{i}" in src_parts and "{i}" in dst_parts:
            si, di = src_parts.index("{i}"), dst_parts.index("{i}")
            src_head, dst_head = ".".join(src_parts[:si]), ".".join(dst_parts[:di])
            if src_head and dst_head:
                prefix[src_head + "."] = dst_head + "."
            src_tail = ".".join(src_parts[si + 1 :])
            dst_tail = ".".join(dst_parts[di + 1 :])
            if not src_tail or not dst_tail or src_tail == dst_tail:
                continue
            shared = ""
            for suffix in (".weight", ".bias"):
                if src_tail.endswith(suffix) and dst_tail.endswith(suffix):
                    src_tail = src_tail[: -len(suffix)]
                    dst_tail = dst_tail[: -len(suffix)]
                    shared = suffix
                    break
            # Only a shared .weight/.bias suffix makes these module prefixes.
            # Otherwise the target is a bare parameter (A_log, dt_bias) and the
            # rule must rewrite the whole name, with no trailing dot.
            if shared:
                substr[src_tail + "."] = _pin_expert_slot(dst_tail) + "."
            else:
                substr[src_tail] = _pin_expert_slot(dst_tail)
        else:
            src_base = source.removesuffix(".weight").removesuffix(".bias")
            dst_base = match.target.removesuffix(".weight").removesuffix(".bias")
            if src_base != dst_base:
                prefix[src_base + "."] = _pin_expert_slot(dst_base) + "."
    return prefix, substr


def format_adapter(matched: dict[str, Match]) -> str:
    prefix, substr = build_mapper_rules(matched)
    lines = ["from vllm.model_executor.models.utils import WeightsMapper", ""]
    stacked = sorted(k for k, m in matched.items() if m.target.count("{i}") > 1)
    if stacked:
        lines += [
            "# Stacked experts: these rename to slot 0 and still need",
            "# split_stacked_experts() in the adapter's weight stream.",
            *(f"#   {name}" for name in stacked),
        ]
    lines += [
        "MAPPER = WeightsMapper(",
        "    orig_to_new_prefix={",
    ]
    lines += [f'        "{k}": "{v}",' for k, v in sorted(prefix.items())]
    lines += ["    },", "    orig_to_new_substr={"]
    # Longest first: substring rules are order sensitive, and a short key can
    # be a prefix of a longer one (``ssm_a`` inside ``ssm_alpha``).
    lines += [
        f'        "{k}": "{v}",'
        for k, v in sorted(substr.items(), key=lambda kv: -len(kv[0]))
    ]
    lines += ["    },", ")"]
    return "\n".join(lines)


def format_report(result: Reconciliation, max_unmatched: int = 25) -> str:
    out: list[str] = ["MATCHED"]
    # A fused source carries a shape note that the concat already explains;
    # listing it under transforms as well reads as two separate fixups.
    fused = {first for first, _, _, _ in result.fusions}
    transforms = []
    for name, match in sorted(result.matched.items()):
        out.append(f"  {name}")
        detail = f" | {match.note}" if match.note else ""
        out.append(f"      -> {match.target}  [{match.how}{detail}]")
        if match.note and name not in fused:
            transforms.append((name, match.note))

    if result.ambiguous:
        out += ["", "AMBIGUOUS (runner-up within 0.5 - review)"]
        for name, candidates in sorted(result.ambiguous.items()):
            out.append(f"  {name}")
            out += [f"      {value:>7}  {target}" for target, value in candidates]

    if transforms:
        out += ["", f"TRANSFORMS NEEDED ({len(transforms)})"]
        out += [f"  {name}: {note}" for name, note in transforms]

    if result.fusions:
        out += ["", f"FUSIONS ({len(result.fusions)})"]
        for first, second, target, axis in result.fusions:
            out.append(f"  [{first}, {second}]")
            out.append(f"      -> {target}  (cat axis {axis})")

    out += ["", f"UNMATCHED GGUF ({len(result.unmatched_source)})"]
    out += [f"  {name}" for name in result.unmatched_source[:max_unmatched]]
    out += ["", f"UNCOVERED TARGET PARAMS ({len(result.unmatched_target)})"]
    out += [f"  {name}" for name in result.unmatched_target[:max_unmatched]]

    total = result.resolved + len(result.unmatched_source)
    pct = 100.0 * result.resolved / max(total, 1)
    out += [
        "",
        (
            f"SUMMARY: {result.resolved}/{total} GGUF templates mapped "
            f"({pct:.0f}%), {len(transforms)} transforms, "
            f"{len(result.fusions)} fusions, {len(result.ambiguous)} ambiguous, "
            f"{len(result.unmatched_target)} target params uncovered"
        ),
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gguf", required=True, help="local path, URL, or repo_id:filename"
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="HF model id or local path for the target config",
    )
    parser.add_argument("--target", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--emit-adapter", action="store_true")
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--max-unmatched", type=int, default=25)
    parser.add_argument("--header-bytes", type=int, default=DEFAULT_HEADER_BYTES)
    args = parser.parse_args(argv)

    gguf_names = collect_gguf(args.gguf, args.header_bytes)
    if args.target == "vllm":
        target_names = collect_vllm(args.target_model, args.trust_remote_code)
    else:
        target_names = collect_hf(args.target_model, args.trust_remote_code)

    source, target = group(gguf_names), group(target_names)
    print(f"GGUF tensors: {len(gguf_names)}  target params: {len(target_names)}")
    print(f"templates:    {len(source)} GGUF  vs  {len(target)} target\n")

    result = reconcile(source, target, args.min_score)
    print(format_report(result, args.max_unmatched))
    if args.emit_adapter:
        print("\nADAPTER SKELETON (review before use)\n")
        print(format_adapter(result.matched))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
