#!/usr/bin/env python3
"""Measure prompt-prefill plus decode on a local supported Q4_K GGUF model."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", default="openai-community/gpt2-large")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[1, 128, 512])
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("research/e2e_results.jsonl")
    )
    args = parser.parse_args()

    fused = os.environ.get("GGUF_PLUGIN_USE_FUSED_Q4_K_EMBEDDING", "1") == "1"
    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        tokenizer=args.tokenizer,
        quantization="gguf",
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.60,
    )
    load_seconds = time.perf_counter() - started
    params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True
    )

    def prompt(n: int) -> dict[str, list[int]]:
        return {"prompt_token_ids": [(i * 257 + 17) % 50257 for i in range(n)]}

    llm.generate([prompt(1)], params, use_tqdm=False)
    torch.cuda.synchronize()
    result: dict[str, object] = {
        "implementation": "fused" if fused else "two_stage",
        "model": str(args.model),
        "tokenizer": args.tokenizer,
        "vllm": __import__("vllm").__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "enforce_eager": True,
        "model_load_seconds": load_seconds,
        "warmup_runs": 1,
        "prompt_results": [],
    }
    for prompt_tokens in args.prompt_tokens:
        durations = []
        output_tokens = []
        output_token_ids = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate([prompt(prompt_tokens)], params, use_tqdm=False)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            output_tokens.append(len(outputs[0].outputs[0].token_ids))
            output_token_ids.append(outputs[0].outputs[0].token_ids)
        median = statistics.median(durations)
        p95 = float(np.percentile(durations, 95))
        result["prompt_results"].append(
            {
                "prompt_tokens": prompt_tokens,
                "output_tokens": statistics.median(output_tokens),
                "repeats": args.repeats,
                "output_token_ids": output_token_ids,
                "median_seconds": median,
                "p95_seconds": p95,
                "median_total_tokens_per_second": (
                    prompt_tokens + statistics.median(output_tokens)
                )
                / median,
                "median_output_tokens_per_second": statistics.median(output_tokens)
                / median,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as file:
        file.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
