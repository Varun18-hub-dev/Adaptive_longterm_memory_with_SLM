#!/usr/bin/env python3
"""
Command-line driver to execute Phase 4 research experiments:
Adaptive vs Flat vs Fixed-Allocation Memory Comparisons.

Usage:
    python run_experiments.py [--dataset {synthetic, longmemeval_dev, longmemeval_eval}]
                              [--budgets 50 100 150 200]
                              [--model {fake, qwen}]
                              [--model-path PATH]
                              [--ablation]
"""

import argparse
import os
import sys
import time

from benchmark_dataset import (
    create_or_load_longmemeval_benchmark,
    get_synthetic_benchmark_suite,
)
from evaluation_harness import ExperimentHarness
from memory_strategies import (
    AblatedMemoryStrategy,
    AdaptiveHierarchicalStrategy,
    FixedHierarchicalStrategy,
    FlatMemoryBaseline,
)
from model_adapter import (
    BaseLLMAdapter,
    FakeLLMAdapter,
    LlamaCppAdapter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Run Memory Budget Allocation Benchmark Comparisons"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synthetic", "longmemeval_dev", "longmemeval_eval"],
        default="synthetic",
        help="Dataset benchmark to evaluate (default: synthetic).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[50, 100, 150, 200],
        help="Storage budgets in tokens to evaluate (default: 50 100 150 200).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["fake", "qwen"],
        default="fake",
        help="Model adapter to use: fake (deterministic, fast) or qwen (local GGUF).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("QWEN_MODEL_PATH", r"C:\Users\Varun S\.ollama\models\blobs\sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6"),
        help="Path to local Qwen GGUF model file.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=256,
        help="Prompt-context budget for LLM generation (default: 256).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Include ablation strategies in the experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Directory to save structured evaluation results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of dataset examples to evaluate (default: evaluate full dataset split).",
    )
    return parser.parse_args()


def format_markdown_table(summary: dict, budgets: list) -> str:
    """Formats summary metrics into a clean Markdown comparison table."""
    headers = [
        "Strategy", "Budget", "Accuracy", "Token F1", "Recall@K",
        "MRR", "Stale Error", "Storage Used", "Latency (ms)"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    for strat_name, budget_dict in summary.items():
        for b in budgets:
            if b in budget_dict:
                m = budget_dict[b]
                row = [
                    f"`{strat_name}`",
                    str(b),
                    f"{m['accuracy'] * 100:.1f}%",
                    f"{m['token_f1']:.3f}",
                    f"{m['recall_at_k'] * 100:.1f}%",
                    f"{m['mrr']:.3f}",
                    f"{m['stale_memory_error_rate'] * 100:.1f}%",
                    f"{m['avg_storage_tokens']:.1f}",
                    f"{m['avg_latency_ms']:.1f}"
                ]
                lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    args = parse_args()

    # 1. Initialize Model Adapter
    if args.model == "fake":
        print("[*] Using FakeLLMAdapter (deterministic, fast automated evaluation).")
        llm_adapter: BaseLLMAdapter = FakeLLMAdapter()
    else:
        print(f"[*] Loading local Qwen GGUF model: {args.model_path}")
        llm_adapter = LlamaCppAdapter(model_path=args.model_path, n_ctx=2048, verbose=False)

    # 2. Load Dataset
    if args.dataset == "synthetic":
        dataset = get_synthetic_benchmark_suite()
        exp_name = "synthetic_comparison"
    elif args.dataset == "longmemeval_dev":
        dev_data, _ = create_or_load_longmemeval_benchmark()
        dataset = dev_data
        exp_name = "longmemeval_dev_comparison"
    else:
        _, eval_data = create_or_load_longmemeval_benchmark()
        dataset = eval_data
        exp_name = "longmemeval_eval_comparison"

    if args.limit is not None and args.limit > 0:
        dataset = dataset[:args.limit]
        exp_name = f"{exp_name}_n{len(dataset)}"

    print(f"[*] Loaded dataset '{args.dataset}' evaluating exactly {len(dataset)} examples.")

    # 3. Strategy Suite Setup
    strategy_constructors = {
        "flat_memory": lambda budget: FlatMemoryBaseline(total_budget=budget),
        "fixed_70_30": lambda budget: FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.70),
        "fixed_50_50": lambda budget: FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.50),
        "fixed_30_70": lambda budget: FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.30),
        "adaptive_hierarchical": lambda budget: AdaptiveHierarchicalStrategy(total_budget=budget),
    }

    if args.ablation:
        print("[*] Adding ablation strategies to suite...")
        strategy_constructors.update({
            "ablation_no_adaptive": lambda budget: AblatedMemoryStrategy(total_budget=budget, disable_adaptive=True),
            "ablation_no_utility": lambda budget: AblatedMemoryStrategy(total_budget=budget, disable_utility=True),
            "ablation_no_consolidation": lambda budget: AblatedMemoryStrategy(total_budget=budget, disable_consolidation=True),
            "ablation_no_temporal_validity": lambda budget: AblatedMemoryStrategy(total_budget=budget, disable_temporal_validity=True),
        })

    # 4. Execute Experiment
    harness = ExperimentHarness(
        llm_adapter=llm_adapter,
        max_context_tokens=args.max_context_tokens,
        output_dir=args.output_dir
    )

    t0 = time.time()
    payload = harness.run_benchmark_suite(
        strategy_constructors=strategy_constructors,
        dataset=dataset,
        budgets=args.budgets,
        experiment_name=exp_name
    )
    elapsed = time.time() - t0

    # 5. Display Markdown Results Table
    md_table = format_markdown_table(payload["summary"], args.budgets)
    print("\n" + "=" * 65)
    print("           RESEARCH EVALUATION RESULTS")
    print("=" * 65)
    print(f"Dataset:            {args.dataset} ({len(dataset)} examples)")
    print(f"Model Adapter:      {args.model}")
    print(f"Budgets:            {args.budgets}")
    print(f"Elapsed Time:       {elapsed:.2f}s\n")
    print(md_table)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
