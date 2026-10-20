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
    load_benchmark_dataset,
)
from candidate_generation import CandidateAdmissionGate
from evaluation_harness import ExperimentHarness
from memory_interface import (
    auto_discover_strategies,
    get_strategy,
    list_registered_strategies,
    load_strategy_module,
)
from memory_strategies import (
    AblatedMemoryStrategy,
    AdaptiveHierarchicalStrategy,
    FixedHierarchicalStrategy,
    FlatMemoryBaseline,
    MockMemoryStrategy,
)
from model_adapter import (
    BaseLLMAdapter,
    FakeLLMAdapter,
    LlamaCppAdapter,
    get_model_adapter,
    heuristic_bpe_token_count,
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
        default=os.environ.get("QWEN_MODEL_PATH"),
        help="Path to local Qwen GGUF model file (can also be set via QWEN_MODEL_PATH).",
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
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        help="Specific memory strategy to evaluate (default: 'all' runs comparative suite).",
    )
    parser.add_argument(
        "--strategy-module",
        type=str,
        default=None,
        help="Path to external Python file or module containing custom @register_strategy implementation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all registered memory strategies and exit.",
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

    # 0. Strategy Discovery: auto-scan workspace and load external strategy module if provided
    auto_discover_strategies()
    if args.strategy_module:
        load_strategy_module(args.strategy_module)
        print(f"[*] Loaded external strategy module: {args.strategy_module}")

    if args.list_strategies:
        print("[*] Available registered memory strategies:")
        for s in list_registered_strategies():
            print(f"  - {s}")
        print("  - all (evaluates all comparative strategies)")
        return

    # Seed for reproducibility
    import random
    random.seed(args.seed)

    # 1. Initialize Model Adapter
    llm_adapter = get_model_adapter(
        model_name=args.model,
        model_path=args.model_path,
        n_ctx=2048,
        verbose=False
    )
    print(f"[*] Initialized model adapter '{llm_adapter.name}' (native tokenizer: {llm_adapter.is_native_tokenizer})")

    # 2. Load Dataset (lazy split loading)
    dataset = load_benchmark_dataset(dataset_name=args.dataset)
    exp_name = f"{args.dataset}_eval"

    if args.limit is not None and args.limit > 0:
        dataset = dataset[:args.limit]
        exp_name = f"{exp_name}_n{len(dataset)}"

    print(f"[*] Loaded dataset '{args.dataset}' evaluating exactly {len(dataset)} examples.")

    # 3. Strategy Suite Setup (with uniform admission gating and tokenizer accounting)
    gate = CandidateAdmissionGate()
    tokenizer_fn = getattr(llm_adapter, "count_tokens", heuristic_bpe_token_count)

    core_strategies = {
        "flat_memory": lambda budget: FlatMemoryBaseline(total_budget=budget, admission_gate=gate, tokenizer_fn=tokenizer_fn),
        "fixed_70_30": lambda budget: FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.70, admission_gate=gate, tokenizer_fn=tokenizer_fn),
        "fixed_50_50": lambda budget: FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.50, admission_gate=gate, tokenizer_fn=tokenizer_fn),
        "fixed_30_70": lambda budget: FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.30, admission_gate=gate, tokenizer_fn=tokenizer_fn),
        "adaptive_hierarchical": lambda budget: AdaptiveHierarchicalStrategy(total_budget=budget, admission_gate=gate, tokenizer_fn=tokenizer_fn),
    }

    if args.strategy == "all":
        strategy_constructors = dict(core_strategies)
        if args.ablation:
            print("[*] Adding ablation strategies to suite...")
            strategy_constructors.update({
                "ablation_no_adaptive": lambda budget: AblatedMemoryStrategy(total_budget=budget, disable_adaptive=True, admission_gate=gate, tokenizer_fn=tokenizer_fn),
                "ablation_no_utility": lambda budget: AblatedMemoryStrategy(total_budget=budget, disable_utility=True, admission_gate=gate, tokenizer_fn=tokenizer_fn),
                "ablation_no_consolidation": lambda budget: AblatedMemoryStrategy(total_budget=budget, disable_consolidation=True, admission_gate=gate, tokenizer_fn=tokenizer_fn),
                "ablation_no_temporal_validity": lambda budget: AblatedMemoryStrategy(total_budget=budget, disable_temporal_validity=True, admission_gate=gate, tokenizer_fn=tokenizer_fn),
            })
    elif args.strategy in core_strategies:
        strategy_constructors = {args.strategy: core_strategies[args.strategy]}
    else:
        # Resolve custom registered strategy from registry
        strat_name = args.strategy
        strategy_constructors = {
            strat_name: lambda budget: get_strategy(strat_name, budget=budget, tokenizer_fn=tokenizer_fn, admission_gate=gate)
        }

    print(f"[*] Active strategies to evaluate: {list(strategy_constructors.keys())}")

    # 4. Execute Experiment
    harness = ExperimentHarness(
        llm_adapter=llm_adapter,
        max_context_tokens=args.max_context_tokens,
        output_dir=args.output_dir
    )

    metadata = {
        "seed": args.seed,
        "dataset": args.dataset,
        "model": llm_adapter.name,
        "is_native_tokenizer": llm_adapter.is_native_tokenizer,
        "max_context_tokens": args.max_context_tokens,
        "strategy": args.strategy,
        "strategy_module": args.strategy_module,
    }

    t0 = time.time()
    payload = harness.run_benchmark_suite(
        strategy_constructors=strategy_constructors,
        dataset=dataset,
        budgets=args.budgets,
        experiment_name=exp_name,
        metadata=metadata
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
