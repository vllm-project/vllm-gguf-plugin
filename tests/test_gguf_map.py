# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import struct

import pytest
import torch

from vllm_gguf_plugin.tools.gguf_map import (
    GGUFHeaderTruncated,
    Template,
    build_mapper_rules,
    detect_fusions,
    expand_vllm_params,
    group,
    is_remote_ref,
    parse_gguf_header,
    reconcile,
    score,
    templatize,
)


def _gguf_header(tensors: dict[str, tuple[int, ...]], kv: dict[str, str]) -> bytes:
    """Minimal GGUF header: shapes are given in torch order and written reversed."""

    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for key, value in kv.items():
        out += s(key) + struct.pack("<I", 8) + s(value)
    for name, shape in tensors.items():
        out += s(name) + struct.pack("<I", len(shape))
        for dim in reversed(shape):
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", 0) + struct.pack("<Q", 0)
    return out


class TestHeaderParsing:
    def test_roundtrip_restores_torch_order_shapes(self):
        want = {"token_embd.weight": (32, 8), "blk.0.attn_q.weight": (8, 8)}
        assert parse_gguf_header(_gguf_header(want, {"general.name": "x"})) == want

    def test_string_array_metadata_is_walked_not_skipped(self):
        # A tokenizer vocab is a string array; mis-skipping it desynchronises
        # the cursor and corrupts every tensor name that follows.
        header = b"GGUF" + struct.pack("<I", 3)
        header += struct.pack("<Q", 1) + struct.pack("<Q", 1)

        def s(text):
            raw = text.encode()
            return struct.pack("<Q", len(raw)) + raw

        header += s("tokenizer.tokens") + struct.pack("<I", 9)
        header += struct.pack("<I", 8) + struct.pack("<Q", 3)
        header += s("a") + s("bb") + s("ccc")
        header += s("output.weight") + struct.pack("<I", 2)
        header += struct.pack("<Q", 4) + struct.pack("<Q", 2)
        header += struct.pack("<I", 0) + struct.pack("<Q", 0)
        assert parse_gguf_header(header) == {"output.weight": (2, 4)}

    def test_truncated_header_raises(self):
        blob = _gguf_header({"a.weight": (4, 4)}, {})
        with pytest.raises(GGUFHeaderTruncated):
            parse_gguf_header(blob[: len(blob) // 2])

    def test_rejects_non_gguf(self):
        with pytest.raises(ValueError, match="not a GGUF"):
            parse_gguf_header(b"XXXX" + b"\x00" * 64)

    @pytest.mark.parametrize(
        ("ref", "remote"),
        [
            ("org/model:Q4_K_M", True),
            ("https://host/model.gguf", True),
            ("/abs/path/model.gguf", False),
            ("./model.gguf", False),
        ],
    )
    def test_remote_ref_detection(self, ref, remote):
        assert is_remote_ref(ref) is remote


class TestTemplating:
    def test_indices_become_placeholders(self):
        assert templatize("blk.3.attn_q.weight") == ("blk.{i}.attn_q.weight", (3,))

    def test_every_index_slot_is_captured(self):
        key, indices = templatize("model.layers.2.mlp.experts.7.gate_proj.weight")
        assert key == "model.layers.{i}.mlp.experts.{i}.gate_proj.weight"
        assert indices == (2, 7)

    def test_group_records_layer_occupancy(self):
        names = {f"blk.{i}.attn_q.weight": (8, 8) for i in (3, 7, 11)}
        names["output.weight"] = (32, 8)
        grouped = group(names)
        assert grouped["blk.{i}.attn_q.weight"].count == 3
        assert grouped["blk.{i}.attn_q.weight"].layers == frozenset({3, 7, 11})
        assert grouped["output.weight"].layers == frozenset()


class TestScoring:
    def test_shape_agreement_beats_mismatch(self):
        src = Template((8, 8), "blk.0.attn_q.weight", 1)
        same = Template((8, 8), "x", 1)
        other = Template((99, 5), "y", 1)
        a = score("blk.{i}.attn_q.weight", src, "m.{i}.self_attn.q_proj.weight", same)
        b = score("blk.{i}.attn_q.weight", src, "m.{i}.self_attn.q_proj.weight", other)
        assert a > b

    def test_disjoint_layer_occupancy_is_penalised(self):
        src = Template((8, 8), "blk.3.attn_output.weight", 1)
        src.add((3,))
        overlapping = Template((8, 8), "x", 1)
        overlapping.add((3,))
        disjoint = Template((8, 8), "y", 1)
        disjoint.add((0,))
        name = "blk.{i}.attn_output.weight"
        target = "m.{i}.self_attn.o_proj.weight"
        assert score(name, src, target, overlapping) > score(
            name, src, target, disjoint
        )


class TestReconcile:
    def _hybrid(self):
        """Qwen3.5-style stack: attention on layers 3/7, GDN on 0/1/2.

        Both projections share a shape and a plausible name, so only layer
        occupancy separates them.
        """
        gguf_names = {}
        for i in (3, 7):
            gguf_names[f"blk.{i}.attn_output.weight"] = (16, 16)
        for i in (0, 1, 2):
            gguf_names[f"blk.{i}.ssm_out.weight"] = (16, 16)
        target_names = {}
        for i in (3, 7):
            target_names[f"model.layers.{i}.self_attn.o_proj.weight"] = (16, 16)
        for i in (0, 1, 2):
            target_names[f"model.layers.{i}.linear_attn.out_proj.weight"] = (16, 16)
        return group(gguf_names), group(target_names)

    def test_layer_occupancy_disambiguates_identical_shapes(self):
        source, target = self._hybrid()
        result = reconcile(source, target)
        assert (
            result.matched["blk.{i}.attn_output.weight"].target
            == "model.layers.{i}.self_attn.o_proj.weight"
        )
        assert (
            result.matched["blk.{i}.ssm_out.weight"].target
            == "model.layers.{i}.linear_attn.out_proj.weight"
        )

    def test_each_target_is_claimed_at_most_once(self):
        source, target = self._hybrid()
        result = reconcile(source, target)
        claimed = [m.target for m in result.matched.values()]
        assert len(claimed) == len(set(claimed))

    def test_leading_expert_dim_is_reported_as_a_split(self):
        source = group({"blk.0.ffn_gate_exps.weight": (64, 14, 8)})
        target = group(
            {
                f"model.layers.0.mlp.experts.{e}.gate_proj.weight": (14, 8)
                for e in range(64)
            }
        )
        result = reconcile(source, target)
        note = result.matched["blk.{i}.ffn_gate_exps.weight"].note
        assert note == "SPLIT: unbind axis 0 into 64"

    def test_unmatched_are_reported_not_silently_dropped(self):
        source = group({"blk.0.mystery_tensor.weight": (3, 5)})
        target = group({"model.layers.0.self_attn.q_proj.weight": (128, 64)})
        result = reconcile(source, target)
        assert result.unmatched_source == ["blk.{i}.mystery_tensor.weight"]
        assert not result.matched


class TestFusionDetection:
    def test_two_sources_concatenating_into_one_target(self):
        # DeepSeek: gate_exps + up_exps -> experts.gate_up_proj on axis 1.
        source = group(
            {
                "blk.1.ffn_gate_exps.weight": (64, 14, 8),
                "blk.1.ffn_up_exps.weight": (64, 14, 8),
            }
        )
        target = group({"model.layers.1.mlp.experts.gate_up_proj": (64, 28, 8)})
        result = reconcile(source, target)
        fusions, remaining = detect_fusions(
            result, sorted(set(source) - set(result.matched)), source, target
        )
        assert not remaining
        assert len(fusions) == 1
        first, second, into, axis = fusions[0]
        assert {first, second} == {
            "blk.{i}.ffn_gate_exps.weight",
            "blk.{i}.ffn_up_exps.weight",
        }
        assert into == "model.layers.{i}.mlp.experts.gate_up_proj"
        assert axis == 1


class TestMapperRules:
    def test_module_rules_keep_the_trailing_dot(self):
        matched = reconcile(
            group({"blk.0.attn_q.weight": (8, 8)}),
            group({"model.layers.0.self_attn.q_proj.weight": (8, 8)}),
        ).matched
        prefix, substr = build_mapper_rules(matched)
        assert prefix["blk."] == "model.layers."
        assert substr["attn_q."] == "self_attn.q_proj."

    def test_bare_parameter_rules_drop_the_trailing_dot(self):
        # A_log is a parameter, not a module: a trailing dot would make the
        # rewritten name "...A_log..weight" and never match.
        matched = reconcile(
            group({"blk.0.ssm_a": (4,)}),
            group({"model.layers.0.linear_attn.A_log": (4,)}),
        ).matched
        _, substr = build_mapper_rules(matched)
        assert substr["ssm_a"] == "linear_attn.A_log"
        assert "ssm_a." not in substr

    def test_emitted_substr_order_puts_longer_keys_first(self):
        from vllm_gguf_plugin.tools.gguf_map import format_adapter

        matched = reconcile(
            group({"blk.0.ssm_a": (4,), "blk.0.ssm_alpha.weight": (4, 4)}),
            group(
                {
                    "model.layers.0.linear_attn.A_log": (4,),
                    "model.layers.0.linear_attn.in_proj_a.weight": (4, 4),
                }
            ),
        ).matched
        text = format_adapter(matched)
        # "ssm_a" is a prefix of "ssm_alpha", so it must be applied last.
        assert text.index('"ssm_alpha.') < text.index('"ssm_a"')


class _FakeMoE(torch.nn.Module):
    def __init__(self, experts, inter, hidden):
        super().__init__()
        self.global_num_experts = experts
        self.intermediate_size_per_partition = inter
        self.hidden_size = hidden
        self.w13_weight = torch.nn.Parameter(
            torch.empty(experts, 2 * inter, hidden, device="meta")
        )
        self.w2_weight = torch.nn.Parameter(
            torch.empty(experts, hidden, inter, device="meta")
        )


_FakeMoE.__name__ = "FusedMoE"


class _FakeMerged(torch.nn.Module):
    def __init__(self, sizes, in_features, bias=False):
        super().__init__()
        self.output_sizes = list(sizes)
        self.weight = torch.nn.Parameter(
            torch.empty(sum(sizes), in_features, device="meta")
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(sum(sizes), device="meta"))


_FakeMerged.__name__ = "MergedColumnParallelLinear"


class TestVllmTargetExpansion:
    def test_stacked_experts_expand_to_per_expert_names(self):
        # vLLM stores w13_weight/w2_weight but its loader consumes per-expert
        # gate/up/down, so the mapping target must be the latter.
        root = torch.nn.Module()
        root.experts = _FakeMoE(experts=4, inter=14, hidden=8)
        names = expand_vllm_params(root)
        assert names["experts.0.gate_proj.weight"] == (14, 8)
        assert names["experts.3.down_proj.weight"] == (8, 14)
        assert not any("w13_weight" in n for n in names)

    def test_fused_linear_expands_via_packed_modules_mapping(self):
        root = torch.nn.Module()
        root.mlp = torch.nn.Module()
        root.mlp.gate_up_proj = _FakeMerged((6, 6), 4)
        type(root).packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        try:
            names = expand_vllm_params(root)
        finally:
            del type(root).packed_modules_mapping
        assert names["mlp.gate_proj.weight"] == (6, 4)
        assert names["mlp.up_proj.weight"] == (6, 4)
        assert "mlp.gate_up_proj.weight" not in names

    def test_fused_bias_is_split_across_the_parts(self):
        # A fused linear concatenates its bias; without splitting it, every
        # attention_bias architecture leaves attn_q.bias unmatched.
        root = torch.nn.Module()
        root.mlp = torch.nn.Module()
        root.mlp.gate_up_proj = _FakeMerged((6, 6), 4, bias=True)
        type(root).packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        try:
            names = expand_vllm_params(root)
        finally:
            del type(root).packed_modules_mapping
        assert names["mlp.gate_proj.bias"] == (6,)
        assert names["mlp.up_proj.bias"] == (6,)

    def test_no_bias_emits_no_bias_names(self):
        root = torch.nn.Module()
        root.mlp = torch.nn.Module()
        root.mlp.gate_up_proj = _FakeMerged((6, 6), 4)
        type(root).packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        try:
            names = expand_vllm_params(root)
        finally:
            del type(root).packed_modules_mapping
        assert not any(n.endswith(".bias") for n in names)


# Real OLMoE-1B-7B-0924-Instruct names and shapes: GGUF header vs the params
# vLLM's loader accepts. q/k/v/o all share (2048, 2048), so only the names
# separate them; the experts are stacked on a leading dim of 64.
OLMOE_GGUF = {
    "blk.0.attn_k.weight": (2048, 2048),
    "blk.0.attn_k_norm.weight": (2048,),
    "blk.0.attn_norm.weight": (2048,),
    "blk.0.attn_output.weight": (2048, 2048),
    "blk.0.attn_q.weight": (2048, 2048),
    "blk.0.attn_q_norm.weight": (2048,),
    "blk.0.attn_v.weight": (2048, 2048),
    "blk.0.ffn_down_exps.weight": (64, 2048, 1024),
    "blk.0.ffn_gate_exps.weight": (64, 1024, 2048),
    "blk.0.ffn_gate_inp.weight": (64, 2048),
    "blk.0.ffn_norm.weight": (2048,),
    "blk.0.ffn_up_exps.weight": (64, 1024, 2048),
    "output.weight": (50304, 2048),
    "output_norm.weight": (2048,),
    "token_embd.weight": (50304, 2048),
}

OLMOE_VLLM = {
    "lm_head.weight": (50304, 2048),
    "model.embed_tokens.weight": (50304, 2048),
    "model.layers.0.input_layernorm.weight": (2048,),
    "model.layers.0.mlp.experts.0.down_proj.weight": (2048, 1024),
    "model.layers.0.mlp.experts.0.gate_proj.weight": (1024, 2048),
    "model.layers.0.mlp.experts.0.up_proj.weight": (1024, 2048),
    "model.layers.0.mlp.gate.weight": (64, 2048),
    "model.layers.0.post_attention_layernorm.weight": (2048,),
    "model.layers.0.self_attn.k_norm.weight": (2048,),
    "model.layers.0.self_attn.k_proj.weight": (2048, 2048),
    "model.layers.0.self_attn.o_proj.weight": (2048, 2048),
    "model.layers.0.self_attn.q_norm.weight": (2048,),
    "model.layers.0.self_attn.q_proj.weight": (2048, 2048),
    "model.layers.0.self_attn.v_proj.weight": (2048, 2048),
    "model.norm.weight": (2048,),
}


class TestAgainstHandWrittenAdapter:
    """Ground truth: the tool must derive the OLMoE mapper already in-tree."""

    def test_derived_rules_equal_build_olmoe_mapper(self):
        from vllm_gguf_plugin.weights_adapter.olmoe import build_olmoe_mapper

        result = reconcile(group(OLMOE_GGUF), group(OLMOE_VLLM))
        assert not result.unmatched_source
        assert not result.unmatched_target
        prefix, substr = build_mapper_rules(result.matched)
        expected = build_olmoe_mapper()
        assert prefix == expected.orig_to_new_prefix
        assert substr == expected.orig_to_new_substr

    def test_stacked_experts_pin_slot_zero_and_are_flagged(self):
        # split_stacked_experts() keys on ".experts.0.", so the rename must
        # name slot 0 rather than leave an unresolved index placeholder.
        from vllm_gguf_plugin.tools.gguf_map import format_adapter

        matched = reconcile(group(OLMOE_GGUF), group(OLMOE_VLLM)).matched
        text = format_adapter(matched)
        assert "{i}" not in text.split("MAPPER = ", 1)[1]
        assert "split_stacked_experts()" in text
