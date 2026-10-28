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

    def test_retrieval_metrics_empty_retrieved_memories_returns_zero(self):
        metrics = evaluate_retrieval_quality([], ["I work at DeepMind."])
        self.assertEqual(metrics["recall_at_k"], 0.0)
        self.assertEqual(metrics["precision_at_k"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_duplicate_propositions_count_as_single_evidence_coverage(self):
        # Two propositions from the same evidence fact (e.g. Fajitas in Q10)
        # Ground truth has 2 distinct facts: Fajitas (which contained two clauses/sentences) and Lentil soup
        ev_fajitas = (
            "I finished off the last of my chicken fajitas with onions and peppers. "
            "This is the third meal I got from my chicken fajitas."
        )
        ev_lentil = "I just made a big batch of lentil soup that lasted me for 5 lunches."

        prop_fajitas_1 = MemoryItem(
            id="fajita-1", content="This is the third meal I got from my chicken fajitas.",
            memory_type=MemoryType.SEMANTIC, source_session_id="s21", timestamp=100.0
        )
        prop_fajitas_2 = MemoryItem(
            id="fajita-2", content="I finished off the last of my chicken fajitas with onions and peppers.",
            memory_type=MemoryType.EPISODIC, source_session_id="s21", timestamp=100.0
        )
        distractor = MemoryItem(
            id="dist-1", content="I like walking in the park on weekends.",
            memory_type=MemoryType.EPISODIC, source_session_id="s1", timestamp=50.0
        )

        # Case 1: Both fajitas propositions retrieved, lentil soup not retrieved
        metrics = evaluate_retrieval_quality(
            retrieved_memories=[prop_fajitas_1, prop_fajitas_2, distractor],
            supporting_evidence=[ev_fajitas, ev_lentil]
        )
        # Recall must be 1 / 2 = 0.50 (only Fajitas covered, not 2/2)
        self.assertEqual(metrics["recall_at_k"], 0.50)
        # Precision@3 is 2 / 3 (2 relevant memories out of 3 retrieved)
        self.assertEqual(metrics["precision_at_k"], round(2 / 3, 4))
        # First hit rank is 1 -> MRR = 1.0
        self.assertEqual(metrics["mrr"], 1.0)

    def test_multiple_distinct_evidence_facts_coverage(self):
        ev1 = "I adopted a golden retriever named Max."
        ev2 = "I bought a blue bicycle."
        ev3 = "I moved to Seattle in June."

        mem1 = MemoryItem(id="m1", content="I adopted a golden retriever named Max.", memory_type=MemoryType.SEMANTIC, source_session_id="s1", timestamp=10.0)
        mem2 = MemoryItem(id="m2", content="I bought a blue bicycle.", memory_type=MemoryType.SEMANTIC, source_session_id="s2", timestamp=20.0)
        distractor = MemoryItem(id="d1", content="Weather is sunny today.", memory_type=MemoryType.EPISODIC, source_session_id="s3", timestamp=30.0)

        # 2 of 3 evidence facts covered
        metrics = evaluate_retrieval_quality(
            retrieved_memories=[distractor, mem1, mem2],
            supporting_evidence=[ev1, ev2, ev3]
        )
        self.assertEqual(metrics["recall_at_k"], round(2 / 3, 4))
        self.assertEqual(metrics["precision_at_k"], round(2 / 3, 4))
        self.assertEqual(metrics["mrr"], round(1.0 / 2, 4))  # First hit is rank 2

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

    def test_q7_multi_session_faith_evidence_count_and_retrieval_metrics(self):
        """
        Regression test for Audit A: Question 7 (5a7937c8) in longmemeval_dev has exactly
        3 distinct supporting evidence items (December 10 food drive, December 17 Bible study, December 24 mass).
        Retrieval of all 3 distinct items yields Recall@5 = 1.0 (3/3), Precision@5 = 0.60 (3/5), and MRR = 1.0.
        """
        from benchmark_dataset import load_benchmark_dataset
        dataset = load_benchmark_dataset("longmemeval_dev")
        q7 = next((ex for ex in dataset if ex.id == "5a7937c8"), None)
        self.assertIsNotNone(q7)
        self.assertEqual(len(q7.supporting_evidence), 3)

        # Simulated retrieved memories corresponding to the 3 faith activities and 2 distractors
        m1 = MemoryItem(id="m1", content="St. Mary's Church, which was on December 24th, with my family.", memory_type=MemoryType.EPISODIC, source_session_id="s23", timestamp=100.0)
        m2 = MemoryItem(id="m2", content="Day camps offer a structured environment with activities.", memory_type=MemoryType.EPISODIC, source_session_id="s10", timestamp=100.0)
        m3 = MemoryItem(id="m3", content="I actually just did a Bible study on this same topic at my church a few weeks ago, on December 17th.", memory_type=MemoryType.EPISODIC, source_session_id="s34", timestamp=100.0)
        m4 = MemoryItem(id="m4", content="Many churches offer candlelight services on Christmas Eve.", memory_type=MemoryType.EPISODIC, source_session_id="s23", timestamp=100.0)
        m5 = MemoryItem(id="m5", content="I actually helped out at the church's annual holiday food drive on December 10th.", memory_type=MemoryType.EPISODIC, source_session_id="s6", timestamp=100.0)

        ret = evaluate_retrieval_quality([m1, m2, m3, m4, m5], q7.supporting_evidence)
        self.assertEqual(ret["recall_at_k"], 1.0)
        self.assertEqual(ret["precision_at_k"], 0.6)
        self.assertEqual(ret["mrr"], 1.0)



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
