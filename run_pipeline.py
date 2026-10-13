#!/usr/bin/env python3
"""
Command-line interface to execute the Adaptive Memory Management pipeline (Phases 1 & 2).

Usage:
    python run_pipeline.py [--phase {1,2}] [--input INPUT_JSON] [--output OUTPUT_JSON] [--budget BUDGET]
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
from memory_lifecycle import (
    AdaptiveMemoryManager,
    run_phase2_pipeline,
)


DEFAULT_SAMPLE_CONVERSATION_S1 = [
    {
        "role": "user",
        "content": "Hi, I am Alex. I work at OpenAI on long-term memory architectures.",
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

DEFAULT_SAMPLE_CONVERSATION_S2 = [
    {
        "role": "user",
        "content": "Hi, I moved to DeepMind recently. I work at DeepMind now.",
        "timestamp": 1700086400.0,
    },
    {
        "role": "assistant",
        "content": "Congratulations on moving to DeepMind!",
        "timestamp": 1700086410.0,
    },
    {
        "role": "user",
        "content": "I still prefer concise Python examples using dataclasses.",
        "timestamp": 1700086450.0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Adaptive Long-Term Memory Pipeline (Phase 1 & Phase 2)"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=2,
        help="Pipeline phase to execute: 1 (basic allocation) or 2 (lifecycle management & auditability). Default: 2.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input conversation JSON file. If omitted, sample multi-session conversation is used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the resulting memory state JSON. Defaults to outputs/phase{phase}_memory_state.json",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=120,
        help="Total token budget for memory allocation (default: 120).",
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
    random.seed(args.seed)

    output_path = args.output or f"outputs/phase{args.phase}_memory_state.json"

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

    print(f"[*] Executing Phase {args.phase} Pipeline")
    print(f"[*] Total Token Budget: {args.budget}")
    print(f"[*] Scoring Weights: {scoring_weights}")

    if args.phase == 1:
        if args.input and os.path.exists(args.input):
            with open(args.input, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = DEFAULT_SAMPLE_CONVERSATION_S1

        results = run_phase1_pipeline(
            conversation_history=history,
            session_id=args.session_id,
            total_budget=args.budget,
            scoring_weights=scoring_weights,
            allocation_config=allocation_config,
            persistence_path=output_path,
            current_timestamp=time.time(),
        )

        summary = results["allocation_summary"]
        print("\n" + "=" * 55)
        print("           PHASE 1 EXECUTION SUMMARY")
        print("=" * 55)
        print(f"Session ID:         {summary['session_id']}")
        print(f"Total Budget:       {summary['total_budget']} tokens")
        print(f"Allocated Episodic: {summary['allocated_budget']['episodic']} tokens")
        print(f"Allocated Semantic: {summary['allocated_budget']['semantic']} tokens")
        print(f"Used Episodic:      {summary['tokens_used']['episodic']} tokens")
        print(f"Used Semantic:      {summary['tokens_used']['semantic']} tokens")
        print(f"Total Used:         {summary['tokens_used']['total']} tokens")
        print(f"Persistence output: {os.path.abspath(output_path)}")
        print("=" * 55 + "\n")

    else:
        # Phase 2
        manager = AdaptiveMemoryManager(
            total_budget=args.budget,
            scoring_weights=scoring_weights,
            allocation_config=allocation_config,
            persistence_path=output_path,
        )

        if args.input and os.path.exists(args.input):
            with open(args.input, "r", encoding="utf-8") as f:
                sessions_data = json.load(f)
            if isinstance(sessions_data, list) and sessions_data and isinstance(sessions_data[0], list):
                for idx, s in enumerate(sessions_data):
                    manager.process_session(s, session_id=f"session_{idx+1}")
            else:
                manager.process_session(sessions_data, session_id=args.session_id)
        else:
            print("[*] Running multi-session evolution demonstration (Session 1 & Session 2)...")
            res_s1 = manager.process_session(
                DEFAULT_SAMPLE_CONVERSATION_S1, session_id="session_01", current_timestamp=1700000350.0
            )
            res_s2 = manager.process_session(
                DEFAULT_SAMPLE_CONVERSATION_S2, session_id="session_02", current_timestamp=1700086450.0
            )

        print("\n" + "=" * 55)
        print("           PHASE 2 LIFECYCLE EXECUTION SUMMARY")
        print("=" * 55)
        print(f"Total Budget Limit: {manager.total_budget} tokens")
        total_used_tokens = sum(m.token_cost for m in manager.episodic_store + manager.semantic_store)
        print(f"Active Stores Total Tokens: {total_used_tokens} tokens")
        print(f"Active Episodic Items:      {len(manager.episodic_store)}")
        print(f"Active Semantic Items:      {len(manager.semantic_store)}")
        print(f"Archived (Superseded/Exp):  {len(manager.archive_store)}")
        print(f"Total Audit Log Entries:    {len(manager.audit_log)}")

        print("\n--- Active Semantic Knowledge ---")
        for m in manager.semantic_store:
            preds = m.metadata.get("historical_predecessors", [])
            pred_note = f" (Superseded {len(preds)} older item)" if preds else ""
            print(f"  [{m.lifecycle_status.value.upper()}] {m.content}{pred_note} (U={m.utility_score:.3f}, cost={m.token_cost})")

        print("\n--- Archived / Superseded Items ---")
        for m in manager.archive_store:
            print(f"  [{m.lifecycle_status.value.upper()}] {m.content} (Superseded by: {m.metadata.get('superseded_by_id', 'N/A')})")

        print(f"\nPersisted State: {os.path.abspath(output_path)}")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
