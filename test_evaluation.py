"""
Unit Tests for Phase 4 Evaluation Metrics and Experiment Harness.
"""

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


if __name__ == "__main__":
    unittest.main()
