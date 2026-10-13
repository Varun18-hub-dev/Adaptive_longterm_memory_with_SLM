#!/usr/bin/env python3
"""
Command-line interface to execute the Adaptive Memory Management pipeline (Phases 1, 2, & 3).

Usage:
    python run_pipeline.py [--phase {1,2,3}] [--query "Where do I work now?"] [--model-path PATH]
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
from memory_retrieval import (
    run_phase3_pipeline,
)
from model_adapter import (
    BaseLLMAdapter,
    FakeLLMAdapter,
    LlamaCppAdapter,
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
        description="Run Adaptive Long-Term Memory Pipeline (Phases 1, 2, & 3)"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Pipeline phase: 1 (basic allocation), 2 (lifecycle management), 3 (retrieval + LLM generation). Default: 3.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Where do I work now?",
        help="User query to test retrieval and response generation in Phase 3.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("QWEN_MODEL_PATH"),
        help="Path to local Qwen GGUF model file. If omitted, uses QWEN_MODEL_PATH env or falls back to FakeLLMAdapter.",
    )
    parser.add_argument(
        "--use-fake-llm",
        action="store_true",
        help="Force using the FakeLLMAdapter (useful for fast smoke-testing without model loading).",
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
        default=150,
        help="Total storage budget (tokens) for persistent memory stores (default: 150).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=256,
        help="Separate prompt-context budget (tokens) for assembled LLM prompt (default: 256).",
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
        "--use-hybrid-extractor",
        action="store_true",
        help="Use HybridCandidateExtractor (heuristic + Qwen SLM fallback) instead of pure heuristic extractor.",
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
    print(f"[*] Persistent Storage Budget: {args.budget} tokens")
    print(f"[*] Prompt-Context Budget:     {args.max_context_tokens} tokens")
    print(f"[*] Scoring Weights:           {scoring_weights}")

    # Configure candidate extractor
    candidate_extractor = None
    if args.use_hybrid_extractor:
        hybrid_adapter = None
        if args.model_path and os.path.exists(args.model_path):
            try:
                print(f"[*] Initializing local Qwen GGUF for HybridCandidateExtractor: {args.model_path}")
                hybrid_adapter = LlamaCppAdapter(model_path=args.model_path, n_ctx=2048, verbose=False)
            except Exception as exc:
                print(f"[!] Warning: Could not initialize LlamaCppAdapter for hybrid extractor ({exc}). Using documented fallback.")
        elif args.use_fake_llm:
            print("[*] Initializing FakeLLMAdapter for HybridCandidateExtractor.")
            hybrid_adapter = FakeLLMAdapter()
        else:
            print("[!] Documented fallback: No Qwen model available at model_path. "
                  "HybridCandidateExtractor configured with heuristic extraction fallback.")

        from candidate_generation import HybridCandidateExtractor
        candidate_extractor = HybridCandidateExtractor(
            llm_adapter=hybrid_adapter,
            trigger_slm_on_multi_clause=True
        )

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
            candidate_extractor=candidate_extractor,
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

    elif args.phase == 2:
        manager = AdaptiveMemoryManager(
            total_budget=args.budget,
            scoring_weights=scoring_weights,
            allocation_config=allocation_config,
            persistence_path=output_path,
            candidate_extractor=candidate_extractor,
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
            manager.process_session(
                DEFAULT_SAMPLE_CONVERSATION_S1, session_id="session_01", current_timestamp=1700000350.0
            )
            manager.process_session(
                DEFAULT_SAMPLE_CONVERSATION_S2, session_id="session_02", current_timestamp=1700086450.0
            )

        print("\n" + "=" * 55)
        print("           PHASE 2 LIFECYCLE EXECUTION SUMMARY")
        print("=" * 55)
        print(f"Storage Budget Limit:       {manager.total_budget} tokens")
        total_used_tokens = sum(m.token_cost for m in manager.episodic_store + manager.semantic_store)
        print(f"Active Stores Total Tokens: {total_used_tokens} tokens")
        print(f"Active Episodic Items:      {len(manager.episodic_store)}")
        print(f"Active Semantic Items:      {len(manager.semantic_store)}")
        print(f"Archived (Superseded/Exp):  {len(manager.archive_store)}")
        print(f"Total Audit Log Entries:    {len(manager.audit_log)}")
        print(f"Persisted State:            {os.path.abspath(output_path)}")
        print("=" * 55 + "\n")

    else:
        # Phase 3
        # Select adapter
        if args.use_fake_llm or not args.model_path:
            if not args.model_path and not args.use_fake_llm:
                print("[!] No --model-path or QWEN_MODEL_PATH provided. Falling back to FakeLLMAdapter.")
            else:
                print("[*] Using FakeLLMAdapter (mock inference).")
            llm_adapter: BaseLLMAdapter = FakeLLMAdapter()
        else:
            print(f"[*] Initializing local Qwen GGUF model via llama-cpp-python: {args.model_path}")
            llm_adapter = LlamaCppAdapter(model_path=args.model_path, n_ctx=2048, verbose=False)

        manager = AdaptiveMemoryManager(
            total_budget=args.budget,
            scoring_weights=scoring_weights,
            allocation_config=allocation_config,
            persistence_path=output_path,
            candidate_extractor=candidate_extractor,
        )

        print("[*] Intake multi-session conversations...")
        manager.process_session(
            DEFAULT_SAMPLE_CONVERSATION_S1, session_id="session_01", current_timestamp=1700000350.0
        )
        manager.process_session(
            DEFAULT_SAMPLE_CONVERSATION_S2, session_id="session_02", current_timestamp=1700086450.0
        )

        print(f"[*] Processing user query: '{args.query}'")
        res = run_phase3_pipeline(
            conversation_history=[],
            user_query=args.query,
            llm_adapter=llm_adapter,
            storage_budget=args.budget,
            max_context_tokens=args.max_context_tokens,
            existing_manager=manager,
            current_timestamp=1700086500.0,
            candidate_extractor=candidate_extractor,
        )

        print("\n" + "=" * 55)
        print("           PHASE 3 RETRIEVAL & GENERATION SUMMARY")
        print("=" * 55)
        print(f"User Query:               {res['user_query']}")
        print(f"Historical Query Detected: {res['is_historical_query']}")
        print(f"Retrieved Memories:       {res['retrieved_count']}")
        print(f"Context Tokens Used:      {res['context_result']['context_token_cost']} / {res['context_result']['max_context_tokens']}")
        print(f"Storage Tokens Used:      {res['storage_status']['active_episodic_tokens'] + res['storage_status']['active_semantic_tokens']} / {res['storage_status']['total_storage_budget']}")
        print("\n--- Assembled Memory Context ---")
        print(res["context_result"]["formatted_context"])
        print("\n--- Generated Qwen Response ---")
        print(res["response"])
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
