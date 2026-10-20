"""
Adaptive Long-Term Memory for Language Models
Phase 4: Reproducible Evaluation Harness & Research Comparison Engine

Measures:
1. Answer Quality: Exact Match, Substring Accuracy, Token F1.
2. Retrieval Quality: Recall@K, Precision@K, Mean Reciprocal Rank (MRR).
3. Temporal Accuracy & Stale-Memory Error Rate:
   - Current-fact accuracy vs historical-fact accuracy.
   - Frequency with which superseded facts are erroneously produced as current facts.
4. Storage & Context Efficiency:
   - Persistent memory budget utilization vs budget limit.
   - Prompt context token utilization vs context window.
5. Latency:
   - Retrieval latency, generation latency, end-to-end latency.
"""

from __future__ import annotations

import collections
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from benchmark_dataset import BenchmarkExample
from memory_core import MemoryItem
from memory_retrieval import assemble_memory_context
from memory_strategies import BaseMemoryStrategy
from model_adapter import BaseLLMAdapter, heuristic_bpe_token_count


# ============================================================================
# 1. Metric Calculations
# ============================================================================

def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation and redundant whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score between prediction and ground truth string."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def evaluate_answer_accuracy(
    prediction: str,
    expected_answer: str,
    acceptable_answers: List[str]
) -> Dict[str, float]:
    """
    Evaluates generated text answer against ground truth:
    - exact_match: normalized exact match
    - substring_accuracy: whether any acceptable answer is contained in prediction
    - token_f1: maximum token F1 across acceptable answers
    """
    pred_norm = normalize_text(prediction)
    exp_norm = normalize_text(expected_answer)

    exact_match = 1.0 if pred_norm == exp_norm else 0.0

    # Substring containment
    substring_match = 0.0
    for ans in [expected_answer] + acceptable_answers:
        ans_norm = normalize_text(ans)
        if ans_norm and ans_norm in pred_norm:
            substring_match = 1.0
            break

    # Best token F1
    best_f1 = 0.0
    for ans in [expected_answer] + acceptable_answers:
        score = compute_token_f1(prediction, ans)
        if score > best_f1:
            best_f1 = score

    return {
        "exact_match": exact_match,
        "substring_accuracy": substring_match,
        "token_f1": best_f1,
    }


def evaluate_retrieval_quality(
    retrieved_memories: List[MemoryItem],
    supporting_evidence: List[str]
) -> Dict[str, float]:
    """
    Computes Recall@K, Precision@K, and MRR against ground truth supporting evidence.
    """
    if not supporting_evidence:
        return {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0}

    if not retrieved_memories:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}

    # Normalize evidence strings
    evidence_norms = [normalize_text(ev) for ev in supporting_evidence]

    matches_found = 0
    first_hit_rank = 0

    for rank_idx, mem in enumerate(retrieved_memories, start=1):
        mem_norm = normalize_text(mem.content)
        is_hit = any(ev in mem_norm or mem_norm in ev for ev in evidence_norms)
        if is_hit:
            matches_found += 1
            if first_hit_rank == 0:
                first_hit_rank = rank_idx

    recall = min(1.0, matches_found / len(supporting_evidence))
    precision = matches_found / len(retrieved_memories)
    mrr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0

    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(mrr, 4),
    }


def check_stale_memory_error(
    prediction: str,
    retrieved_memories: List[MemoryItem],
    is_historical: bool
) -> float:
    """
    Detects if the model output erroneously asserted an older superseded value
    when answering a current state query (Stale Memory Error Rate).
    Returns 1.0 if stale error occurred, 0.0 otherwise.
    """
    if is_historical:
        return 0.0  # Historical queries are allowed to discuss past values

    pred_norm = normalize_text(prediction)

    for mem in retrieved_memories:
        predecessors = mem.metadata.get("historical_predecessors", [])
        for pred in predecessors:
            prev_val = normalize_text(pred.get("previous_value", ""))
            # If the output asserts the superseded value as fact without correction
            if prev_val and prev_val in pred_norm:
                current_val = normalize_text(mem.metadata.get("slot_value", ""))
                # If it mentions the old value but NOT the current value, it's a stale error
                if current_val and current_val not in pred_norm:
                    return 1.0

    return 0.0


# ============================================================================
# 2. Experiment Runner
# ============================================================================

@dataclass
class QuestionEvaluationResult:
    question_id: str
    category: str
    strategy_name: str
    budget: int
    user_query: str
    expected_answer: str
    generated_response: str
    retrieved_memory_ids: List[str]
    retrieved_contents: List[str]
    exact_match: float
    substring_accuracy: float
    token_f1: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    stale_memory_error: float
    storage_tokens_used: int
    context_tokens_used: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentHarness:
    """
    Executes fair, paired comparisons across memory strategies under identical inputs.
    """

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        max_context_tokens: int = 256,
        output_dir: str = "outputs/experiments"
    ):
        self.llm_adapter = llm_adapter
        self.max_context_tokens = max_context_tokens
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_single_comparison(
        self,
        strategy: BaseMemoryStrategy,
        example: BenchmarkExample,
        budget: int
    ) -> QuestionEvaluationResult:
        """
        Runs one strategy on one benchmark example under a fixed budget constraint.
        Strictly prevents data leakage: query and answer are never present during ingestion.
        Guarantees isolation: calls strategy.reset() before processing.
        """
        try:
            # 0. Enforce clean isolation
            if hasattr(strategy, "reset"):
                strategy.reset()

            # 1. Ingest multi-session conversation history
            for idx, session_turns in enumerate(example.sessions):
                session_id = f"{example.id}_sess_{idx+1}"
                session_timestamp = session_turns[-1].get("timestamp", 1000.0) if session_turns else 1000.0
                strategy.process_session(
                    conversation_history=session_turns,
                    session_id=session_id,
                    current_timestamp=session_timestamp
                )

            storage_tokens_used = strategy.get_storage_usage()

            # 2. Measure Retrieval Latency
            t0_retrieval = time.perf_counter()
            retrieved_raw = strategy.retrieve(query=example.query, top_k=5)
            retrieval_latency_ms = (time.perf_counter() - t0_retrieval) * 1000.0

            # Normalize retrieved items
            retrieved_memories = []
            retrieved_ids = []
            retrieved_contents = []

            for idx, item in enumerate(retrieved_raw):
                mem = item[0] if (isinstance(item, tuple) and len(item) == 2) else item
                retrieved_memories.append(mem)
                cid = getattr(mem, "id", None) or (mem.get("id") if isinstance(mem, dict) else f"mem_{idx+1}")
                ccontent = getattr(mem, "content", None) or (mem.get("content") if isinstance(mem, dict) else str(mem))
                retrieved_ids.append(str(cid))
                retrieved_contents.append(str(ccontent))

            # 3. Context Assembly
            context_res = assemble_memory_context(
                query=example.query,
                retrieved_memories=retrieved_raw,
                max_context_tokens=self.max_context_tokens,
                tokenizer_fn=getattr(self.llm_adapter, "count_tokens", heuristic_bpe_token_count)
            )

            messages = [
                {"role": "system", "content": f"{context_res.system_prompt}\n\nRELEVANT MEMORIES:\n{context_res.formatted_context}"},
                {"role": "user", "content": example.query}
            ]

            # 4. Measure Generation Latency
            t0_gen = time.perf_counter()
            response = self.llm_adapter.generate_chat(
                messages=messages,
                max_tokens=128,
                temperature=0.0  # Deterministic decoding for scientific evaluation
            )
            generation_latency_ms = (time.perf_counter() - t0_gen) * 1000.0
            total_latency_ms = retrieval_latency_ms + generation_latency_ms

            # 5. Evaluate Metrics
            ans_metrics = evaluate_answer_accuracy(
                prediction=response,
                expected_answer=example.expected_answer,
                acceptable_answers=example.acceptable_answers
            )
            ret_metrics = evaluate_retrieval_quality(
                retrieved_memories=retrieved_memories,
                supporting_evidence=example.supporting_evidence
            )
            stale_error = check_stale_memory_error(
                prediction=response,
                retrieved_memories=retrieved_memories,
                is_historical=example.is_historical
            )

            return QuestionEvaluationResult(
                question_id=example.id,
                category=example.category,
                strategy_name=strategy.name,
                budget=budget,
                user_query=example.query,
                expected_answer=example.expected_answer,
                generated_response=response,
                retrieved_memory_ids=retrieved_ids,
                retrieved_contents=retrieved_contents,
                exact_match=ans_metrics["exact_match"],
                substring_accuracy=ans_metrics["substring_accuracy"],
                token_f1=ans_metrics["token_f1"],
                recall_at_k=ret_metrics["recall_at_k"],
                precision_at_k=ret_metrics["precision_at_k"],
                mrr=ret_metrics["mrr"],
                stale_memory_error=stale_error,
                storage_tokens_used=storage_tokens_used,
                context_tokens_used=context_res.context_token_cost,
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                generation_latency_ms=round(generation_latency_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
                error=None
            )

        except Exception as exc:
            print(f"[!] Error evaluating question {example.id} with strategy {strategy.name}: {exc}")
            return QuestionEvaluationResult(
                question_id=example.id,
                category=example.category,
                strategy_name=strategy.name,
                budget=budget,
                user_query=example.query,
                expected_answer=example.expected_answer,
                generated_response=f"ERROR: {str(exc)}",
                retrieved_memory_ids=[],
                retrieved_contents=[],
                exact_match=0.0,
                substring_accuracy=0.0,
                token_f1=0.0,
                recall_at_k=0.0,
                precision_at_k=0.0,
                mrr=0.0,
                stale_memory_error=0.0,
                storage_tokens_used=0,
                context_tokens_used=0,
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=0.0,
                error=str(exc)
            )

    def run_benchmark_suite(
        self,
        strategy_constructors: Dict[str, Any],
        dataset: List[BenchmarkExample],
        budgets: List[int],
        experiment_name: str = "evaluation_run",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete paired comparison across all strategies and budgets.
        """
        all_results: List[QuestionEvaluationResult] = []

        print(f"[*] Starting Experiment: {experiment_name}")
        print(f"[*] Budgets to evaluate: {budgets}")
        print(f"[*] Total Questions: {len(dataset)}")
        print(f"[*] Strategies: {list(strategy_constructors.keys())}")

        for budget in budgets:
            print(f"\n---> Evaluating Budget: {budget} tokens <---", flush=True)
            for strat_key, strat_factory in strategy_constructors.items():
                print(f"  Running Strategy: {strat_key}...", flush=True)
                for ex_idx, example in enumerate(dataset, start=1):
                    # Fresh strategy instance per question to prevent cross-contamination
                    strategy_instance = strat_factory(budget=budget)
                    res = self.run_single_comparison(
                        strategy=strategy_instance,
                        example=example,
                        budget=budget
                    )
                    all_results.append(res)
                    print(f"    [{strat_key}] ({ex_idx}/{len(dataset)}) {example.id}: acc={res.substring_accuracy}, tokens={res.storage_tokens_used}", flush=True)

        # Aggregate metrics
        summary = self.aggregate_results(all_results, budgets)

        # Save to disk
        timestamp_str = int(time.time())
        output_file = os.path.join(self.output_dir, f"{experiment_name}_{timestamp_str}.json")
        payload = {
            "experiment_name": experiment_name,
            "timestamp": timestamp_str,
            "metadata": metadata or {},
            "budgets": budgets,
            "question_count": len(dataset),
            "summary": summary,
            "per_question_results": [r.to_dict() for r in all_results]
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"\n[+] Experiment results successfully saved to: {os.path.abspath(output_file)}")
        return payload

    def aggregate_results(
        self,
        results: List[QuestionEvaluationResult],
        budgets: List[int]
    ) -> Dict[str, Any]:
        """Calculates mean metrics per strategy and budget."""
        grouped: Dict[str, Dict[int, List[QuestionEvaluationResult]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

        for r in results:
            grouped[r.strategy_name][r.budget].append(r)

        summary: Dict[str, Any] = {}

        for strat_name, budget_dict in grouped.items():
            summary[strat_name] = {}
            for budget, items in budget_dict.items():
                n = len(items)
                failed_count = sum(1 for i in items if i.error is not None)
                succeeded_count = sum(1 for i in items if i.error is None)
                summary[strat_name][budget] = {
                    "count": n,
                    "succeeded_count": succeeded_count,
                    "failed_count": failed_count,
                    "accuracy": round(sum(i.substring_accuracy for i in items) / n, 4) if n else 0.0,
                    "exact_match": round(sum(i.exact_match for i in items) / n, 4) if n else 0.0,
                    "token_f1": round(sum(i.token_f1 for i in items) / n, 4) if n else 0.0,
                    "recall_at_k": round(sum(i.recall_at_k for i in items) / n, 4) if n else 0.0,
                    "precision_at_k": round(sum(i.precision_at_k for i in items) / n, 4) if n else 0.0,
                    "mrr": round(sum(i.mrr for i in items) / n, 4) if n else 0.0,
                    "stale_memory_error_rate": round(sum(i.stale_memory_error for i in items) / n, 4) if n else 0.0,
                    "avg_storage_tokens": round(sum(i.storage_tokens_used for i in items) / n, 1) if n else 0.0,
                    "avg_context_tokens": round(sum(i.context_tokens_used for i in items) / n, 1) if n else 0.0,
                    "avg_latency_ms": round(sum(i.total_latency_ms for i in items) / n, 1) if n else 0.0,
                }

                cat_breakdown: Dict[str, List[QuestionEvaluationResult]] = collections.defaultdict(list)
                for i in items:
                    cat_breakdown[i.category].append(i)
                summary[strat_name][budget]["by_category"] = {
                    cat: {
                        "count": len(c_items),
                        "accuracy": round(sum(ci.substring_accuracy for ci in c_items) / len(c_items), 4),
                        "token_f1": round(sum(ci.token_f1 for ci in c_items) / len(c_items), 4),
                    }
                    for cat, c_items in cat_breakdown.items()
                }

        return summary
