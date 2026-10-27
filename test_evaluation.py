"""
Unit Tests for Phase 4 Evaluation Metrics and Experiment Harness.
"""

import math
import tempfile
import unittest

from benchmark_dataset import BenchmarkExample, get_synthetic_benchmark_suite
from evaluation_harness import (
    ExperimentHarness,
    check_stale_memory_error,
    compute_token_f1,
    evaluate_answer_accuracy,
    evaluate_retrieval_quality,
)
from memory_core import LifecycleStatus, MemoryItem, MemoryType
from memory_strategies import AdaptiveHierarchicalStrategy, FlatMemoryBaseline
from model_adapter import FakeLLMAdapter


class TestEvaluationMetrics(unittest.TestCase):
    def test_answer_accuracy_metrics(self):
        # Exact match
        res_exact = evaluate_answer_accuracy("DeepMind", "DeepMind", ["deepmind"])
        self.assertEqual(res_exact["exact_match"], 1.0)
        self.assertEqual(res_exact["substring_accuracy"], 1.0)
        self.assertEqual(res_exact["token_f1"], 1.0)

        # Substring containment
        res_sub = evaluate_answer_accuracy("You currently work at DeepMind.", "DeepMind", ["deepmind"])
        self.assertEqual(res_sub["exact_match"], 0.0)
        self.assertEqual(res_sub["substring_accuracy"], 1.0)
        self.assertGreater(res_sub["token_f1"], 0.3)

        # Wrong answer
        res_wrong = evaluate_answer_accuracy("You work at Microsoft.", "DeepMind", ["deepmind"])
        self.assertEqual(res_wrong["substring_accuracy"], 0.0)
        self.assertEqual(res_wrong["exact_match"], 0.0)

    def test_retrieval_metrics(self):
        mem1 = MemoryItem(id="1", content="I work at DeepMind.", memory_type=MemoryType.SEMANTIC,
                          source_session_id="s", timestamp=0)
        mem2 = MemoryItem(id="2", content="I prefer PyTorch.", memory_type=MemoryType.SEMANTIC,
                          source_session_id="s", timestamp=0)

        metrics = evaluate_retrieval_quality([mem1, mem2], ["I work at DeepMind."])
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_retrieval_metrics_missing_evidence_returns_nan(self):
        mem1 = MemoryItem(id="1", content="I work at DeepMind.", memory_type=MemoryType.SEMANTIC,
                          source_session_id="s", timestamp=0)
        metrics = evaluate_retrieval_quality([mem1], [])
        self.assertTrue(math.isnan(metrics["recall_at_k"]))
        self.assertTrue(math.isnan(metrics["precision_at_k"]))
        self.assertTrue(math.isnan(metrics["mrr"]))

    def test_stale_memory_error_detection(self):
        mem = MemoryItem(
            id="1", content="I work at DeepMind now.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s", timestamp=2000.0,
            metadata={
                "slot_value": "deepmind",
                "historical_predecessors": [{"previous_value": "openai"}]
            }
        )

        # Output states OpenAI (the stale value) without mentioning DeepMind
        error_pred = check_stale_memory_error("You work at OpenAI.", [mem], is_historical=False)
        self.assertEqual(error_pred, 1.0)

        # Output states current value (DeepMind)
        correct_pred = check_stale_memory_error("You work at DeepMind.", [mem], is_historical=False)
        self.assertEqual(correct_pred, 0.0)

        # Historical query is allowed to mention past value
        hist_pred = check_stale_memory_error("You worked at OpenAI previously.", [mem], is_historical=True)
        self.assertEqual(hist_pred, 0.0)


class TestExperimentHarness(unittest.TestCase):
    def test_harness_run_small_suite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_llm = FakeLLMAdapter()
            harness = ExperimentHarness(llm_adapter=fake_llm, max_context_tokens=128, output_dir=tmpdir)

            strategies = {
                "flat": lambda budget: FlatMemoryBaseline(total_budget=budget),
                "adaptive": lambda budget: AdaptiveHierarchicalStrategy(total_budget=budget),
            }
            # Run first 2 examples
            dataset = get_synthetic_benchmark_suite()[:2]
            budgets = [50, 100]

            payload = harness.run_benchmark_suite(
                strategy_constructors=strategies,
                dataset=dataset,
                budgets=budgets,
                experiment_name="test_run"
            )

            self.assertIn("summary", payload)
            self.assertIn("per_question_results", payload)
            self.assertEqual(len(payload["per_question_results"]), 2 * 2 * 2)  # 2 budgets * 2 strats * 2 questions

    def test_aggregate_results_transparent_nan_handling(self):
        from evaluation_harness import QuestionEvaluationResult
        fake_llm = FakeLLMAdapter()
        harness = ExperimentHarness(llm_adapter=fake_llm)

        r1 = QuestionEvaluationResult(
            question_id="q1", category="c1", strategy_name="strat_test", budget=100,
            user_query="q1", expected_answer="a1", generated_response="a1",
            retrieved_memory_ids=["m1"], retrieved_contents=["c1"],
            exact_match=1.0, substring_accuracy=1.0, token_f1=1.0,
            recall_at_k=1.0, precision_at_k=1.0, mrr=1.0,
            stale_memory_error=0.0, storage_tokens_used=50, context_tokens_used=20,
            retrieval_latency_ms=1.0, generation_latency_ms=2.0, total_latency_ms=3.0
        )
        r2 = QuestionEvaluationResult(
            question_id="q2", category="c1", strategy_name="strat_test", budget=100,
            user_query="q2", expected_answer="a2", generated_response="a2",
            retrieved_memory_ids=[], retrieved_contents=[],
            exact_match=0.0, substring_accuracy=0.0, token_f1=0.0,
            recall_at_k=float("nan"), precision_at_k=float("nan"), mrr=float("nan"),
            stale_memory_error=0.0, storage_tokens_used=50, context_tokens_used=20,
            retrieval_latency_ms=1.0, generation_latency_ms=2.0, total_latency_ms=3.0
        )

        summary = harness.aggregate_results([r1, r2], budgets=[100])
        metrics = summary["strat_test"][100]

        # Total count is 2
        self.assertEqual(metrics["count"], 2)
        # Exactly 1 question had valid retrieval metrics
        self.assertEqual(metrics["retrieval_count"], 1)
        self.assertEqual(metrics["missing_evidence_count"], 1)
        # NaN is neither treated as 0.0 nor 1.0; recall average over valid questions is 1.0
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["precision_at_k"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
