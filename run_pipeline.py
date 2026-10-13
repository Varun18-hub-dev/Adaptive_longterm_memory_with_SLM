#!/usr/bin/env python3
"""
Command-line interface to execute the Phase 1 Adaptive Memory Management pipeline.

Usage:
    python run_pipeline.py [--input INPUT_JSON] [--output OUTPUT_JSON] [--budget BUDGET]
"""

import argparse
import json
import os
import random
import sys
import time

from memory_core import (
    AllocationConfig,
    ScoringWeights,
    run_phase1_pipeline,
)


DEFAULT_SAMPLE_CONVERSATION = [
    {
        "role": "user",
        "content": "Hi, I am Alex. I work at deepmind on long-term memory architectures.",
        "timestamp": 1700000000.0,
    },
    {
        "role": "assistant",
        "content": "Hello Alex! Pleased to meet you. Long-term memory is an active and crucial area of research.",
        "timestamp": 1700000010.0,
    },
    {
        "role": "user",
        "content": "I always prefer concise Python examples using dataclasses. Please never use verbose boilerplate.",
        "timestamp": 1700000050.0,
    },
    {
        "role": "assistant",
        "content": "Noted. I will provide concise Python code with dataclasses and avoid unnecessary boilerplate.",
        "timestamp": 1700000060.0,
    },
    {
        "role": "user",
        "content": "Today I ran an experiment on CUDA and got an out-of-memory error when batch size was 32.",
        "timestamp": 1700000200.0,
    },
    {
        "role": "assistant",
        "content": "For batch size 32 OOM, try gradient accumulation or activation checkpointing.",
        "timestamp": 1700000220.0,
    },
    {
        "role": "user",
        "content": "I always prefer concise Python examples using dataclasses.",
        "timestamp": 1700000300.0,
    },
    {
        "role": "user",
        "content": "Can you help me debug this CUDA issue right now?",
        "timestamp": 1700000350.0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase 1: Adaptive Long-Term Memory Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input conversation JSON file. If omitted, a sample conversation is used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/phase1_memory_state.json",
        help="Path to save the resulting memory state JSON.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Total token budget for memory allocation (default: 100).",
    )
    parser.add_argument(
        "--min-episodic",
        type=float,
        default=0.20,
        help="Minimum episodic budget share (default: 0.20).",
    )
    parser.add_argument(
        "--max-episodic",
        type=float,
        default=0.80,
        help="Maximum episodic budget share (default: 0.80).",
    )
    parser.add_argument(
        "--weights",
        nargs=5,
        type=float,
        metavar=("W_R", "W_I", "W_F", "W_T", "W_C"),
        default=[0.25, 0.25, 0.20, 0.15, 0.15],
        help="Scoring weights (Recency, Importance, Frequency, Stability, Consistency). Must sum to 1.0.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="session_demo_01",
        help="Session identifier for provenance tracking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load conversation history
    if args.input and os.path.exists(args.input):
        print(f"[*] Loading conversation from {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            conversation_history = json.load(f)
    else:
        print("[*] Using default sample conversation history.")
        conversation_history = DEFAULT_SAMPLE_CONVERSATION

    # Initialize configs
    w_r, w_i, w_f, w_t, w_c = args.weights
    scoring_weights = ScoringWeights(w_r=w_r, w_i=w_i, w_f=w_f, w_t=w_t, w_c=w_c)
    allocation_config = AllocationConfig(
        total_budget=args.budget,
        min_episodic_share=args.min_episodic,
        max_episodic_share=args.max_episodic,
        min_semantic_share=1.0 - args.max_episodic,
        max_semantic_share=1.0 - args.min_episodic,
    )

    print(f"[*] Total Token Budget: {args.budget}")
    print(f"[*] Scoring Weights: {scoring_weights}")

    # Run pipeline
    results = run_phase1_pipeline(
        conversation_history=conversation_history,
        session_id=args.session_id,
        total_budget=args.budget,
        scoring_weights=scoring_weights,
        allocation_config=allocation_config,
        persistence_path=args.output,
        current_timestamp=time.time(),
    )

    summary = results["allocation_summary"]
    print("\n" + "=" * 50)
    print("           PHASE 1 EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Session ID:         {summary['session_id']}")
    print(f"Total Budget:       {summary['total_budget']} tokens")
    print(f"Allocated Episodic: {summary['allocated_budget']['episodic']} tokens")
    print(f"Allocated Semantic: {summary['allocated_budget']['semantic']} tokens")
    print(f"Used Episodic:      {summary['tokens_used']['episodic']} tokens")
    print(f"Used Semantic:      {summary['tokens_used']['semantic']} tokens")
    print(f"Total Used:         {summary['tokens_used']['total']} tokens")
    print(f"Candidates Total:   {summary['candidate_counts']['total']} (Episodic: {summary['candidate_counts']['episodic']}, Semantic: {summary['candidate_counts']['semantic']})")
    print(f"Selected Total:     {summary['selected_counts']['total']} (Episodic: {summary['selected_counts']['episodic']}, Semantic: {summary['selected_counts']['semantic']})")
    print(f"Persistence output: {os.path.abspath(args.output)}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
