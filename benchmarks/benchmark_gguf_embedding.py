#!/usr/bin/env python3
"""Compare the two-stage and fused Q4_K embedding lookup paths."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from functools import partial
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from huggingface_hub import hf_hub_download
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization.vocal_embeds import apply_gguf_embedding


def load_q4_k_table(
    source_checkpoint: Path | None,
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    if source_checkpoint is None:
        path = hf_hub_download("Isotr0py/test-gguf-sample", "Quant_Q4_K_1024.gguf")
        reader = GGUFReader(path)
        tensor = next(t for t in reader.tensors if t.name.endswith("5120x1024"))
        source_label = "sample-derived"
    else:
        reader = GGUFReader(source_checkpoint)
        tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
        if tensor.tensor_type != GGMLQuantizationType.Q4_K:
            raise ValueError(
                f"Expected Q4_K token_embd.weight, got {tensor.tensor_type}"
            )
        source_label = f"checkpoint:{source_checkpoint}"

    dense = gguf.quants.dequantize(tensor.data, GGMLQuantizationType.Q4_K)
    hidden = dense.shape[1]
    vocab = dense.shape[0]
    return tensor.data, dense, vocab, hidden, source_label


def make_table(
    source: np.ndarray,
    dense_source: np.ndarray,
    vocab: int,
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_vocab, base_hidden = dense_source.shape
    if hidden % base_hidden:
        raise ValueError(
            f"hidden={hidden} must be a multiple of sample width {base_hidden}"
        )
    copies = hidden // base_hidden
    quant_rows = np.tile(source, (1, copies))
    dense_rows = np.tile(dense_source, (1, copies)).astype(np.float16)
    repeats = math.ceil(vocab / base_vocab)
    quant_table = np.tile(quant_rows, (repeats, 1))[:vocab].copy()
    dense_table = np.tile(dense_rows, (repeats, 1))[:vocab].copy()
    return (
        torch.as_tensor(quant_table, device=device),
        torch.as_tensor(dense_table, dtype=dtype, device=device),
    )


def make_ids(
    tokens: int,
    vocab: int,
    distribution: str,
    generator: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    if distribution == "uniform":
        values = generator.integers(0, vocab, size=tokens, dtype=np.int64)
    elif distribution == "repeated":
        values = np.full(tokens, vocab // 3, dtype=np.int64)
    elif distribution == "zipfian":
        values = np.minimum(generator.zipf(1.25, size=tokens) - 1, vocab - 1)
    else:
        raise ValueError(distribution)
    return torch.as_tensor(values, dtype=torch.long, device=device)


def two_stage(
    ids: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
    hidden: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
    assert hidden == weight.shape[1] // type_size * block_size
    flat = ids.flatten()
    quant = torch.index_select(weight, 0, flat)
    decoded = ops.ggml_dequantize(quant, weight_type, hidden, flat.numel(), dtype)
    return decoded.view(*ids.shape, hidden)


def two_stage_fake(
    ids: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
    hidden: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    del weight, weight_type
    return torch.empty((*ids.shape, hidden), device=ids.device, dtype=dtype)


direct_register_custom_op(
    op_name="_research_gguf_embedding_two_stage",
    op_func=two_stage,
    fake_impl=two_stage_fake,
)


def measure_events(fn, trials: int, graph_replays: int = 0) -> list[float]:
    values: list[float] = []
    if graph_replays:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_replays):
                fn()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            graph.replay()
            end.record()
        ends[-1].synchronize()
        values = [
            start.elapsed_time(end) * 1000 / graph_replays
            for start, end in zip(starts, ends, strict=True)
        ]
        return values

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    ends[-1].synchronize()
    return [
        start.elapsed_time(end) * 1000 for start, end in zip(starts, ends, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--vocab", type=int, nargs="+")
    parser.add_argument("--hidden", type=int, nargs="+")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 4, 16, 128, 512, 2048]
    )
    parser.add_argument(
        "--distributions", nargs="+", default=["uniform", "repeated", "zipfian"]
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16"]
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--graph-replays", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--experiment", default="E1")
    parser.add_argument(
        "--output", type=Path, default=Path("research/benchmark_results.csv")
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    raw, dense_source, source_vocab, source_hidden, source_label = load_q4_k_table(
        args.source_checkpoint
    )
    vocab_values = args.vocab or [source_vocab]
    hidden_values = args.hidden or [source_hidden]
    rng = np.random.default_rng(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "implementation",
        "quant_type",
        "vocab",
        "hidden_size",
        "tokens",
        "distribution",
        "dtype",
        "timing_mode",
        "median_us",
        "p95_us",
        "peak_allocated_bytes",
        "effective_GBps",
        "correctness",
        "notes",
    ]
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for vocab in vocab_values:
            for hidden in hidden_values:
                weight, dense_weight = make_table(
                    raw, dense_source, vocab, hidden, device, torch.float16
                )
                cases = [
                    (tokens, distribution)
                    for tokens in args.tokens
                    for distribution in args.distributions
                ]
                random.Random(args.seed + vocab + hidden).shuffle(cases)
                for tokens, distribution in cases:
                    ids = make_ids(tokens, vocab, distribution, rng, device)
                    reference = torch.embedding(dense_weight, ids)
                    for dtype_name in args.dtypes:
                        dtype = (
                            torch.float16 if dtype_name == "float16" else torch.bfloat16
                        )
                        dense_weight = dense_weight.to(dtype)
                        reference = torch.embedding(dense_weight, ids)
                        fns = {
                            "baseline_two_stage": partial(
                                torch.ops.vllm._research_gguf_embedding_two_stage,
                                ids,
                                weight,
                                int(GGMLQuantizationType.Q4_K),
                                hidden,
                                dtype,
                            ),
                            "fused_plugin": partial(
                                apply_gguf_embedding,
                                ids,
                                weight,
                                int(GGMLQuantizationType.Q4_K),
                                hidden,
                                dtype=dtype,
                            ),
                        }
                        implementations = list(fns.items())
                        random.Random(args.seed + tokens + vocab).shuffle(
                            implementations
                        )
                        for implementation, fn in implementations:
                            actual = fn()
                            torch.testing.assert_close(
                                actual, reference, atol=1e-2, rtol=4e-2
                            )
                            del actual
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            baseline_alloc = torch.cuda.memory_allocated()
                            sample_out = fn()
                            torch.cuda.synchronize()
                            peak = torch.cuda.max_memory_allocated() - baseline_alloc
                            del sample_out
                            torch.cuda.synchronize()
                            for mode, graph_replays in (
                                ("eager_cuda_events", 0),
                                ("cuda_graph_replay", args.graph_replays),
                            ):
                                samples = measure_events(fn, args.trials, graph_replays)
                                median = statistics.median(samples)
                                p95 = float(np.percentile(samples, 95))
                                row_bytes = hidden // 256 * 144
                                compressed_traffic = tokens * row_bytes
                                if implementation == "baseline_two_stage":
                                    compressed_traffic *= 2
                                total_traffic = (
                                    compressed_traffic
                                    + tokens
                                    * hidden
                                    * torch.empty((), dtype=dtype).element_size()
                                )
                                bandwidth = total_traffic / (median * 1e-6) / 1e9
                                writer.writerow(
                                    {
                                        "experiment": args.experiment,
                                        "implementation": implementation,
                                        "quant_type": "Q4_K",
                                        "vocab": vocab,
                                        "hidden_size": hidden,
                                        "tokens": tokens,
                                        "distribution": distribution,
                                        "dtype": dtype_name,
                                        "timing_mode": mode,
                                        "median_us": f"{median:.4f}",
                                        "p95_us": f"{p95:.4f}",
                                        "peak_allocated_bytes": peak,
                                        "effective_GBps": f"{bandwidth:.4f}",
                                        "correctness": "pass",
                                        "notes": f"{source_label}; seed={args.seed}",
                                    }
                                )
                                file.flush()
                            print(
                                f"{implementation} V={vocab} H={hidden} T={tokens} "
                                f"{distribution} {dtype_name} ok"
                            )
                del weight, dense_weight
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
