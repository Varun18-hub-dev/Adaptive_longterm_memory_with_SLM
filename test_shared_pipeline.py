"""
Unit and integration tests for the shared research pipeline:
- Layer 1: Dataset schema validation, ordering, and lazy loading
- Layer 1: Model adapter selection, validation, and error reporting
- Layer 2: Memory integration contract, registry, and reset isolation
- Layer 2: Context assembly with strategy-neutral evidence (RetrievedEvidence, dicts, strings)
- Layer 3: End-to-end benchmark execution with MockMemoryStrategy and error isolation
"""

import os
import unittest
from typing import Any, Dict, List

from benchmark_dataset import (
    BenchmarkExample,
    get_synthetic_benchmark_suite,
    load_benchmark_dataset,
    validate_benchmark_example,
)
from evaluation_harness import ExperimentHarness, QuestionEvaluationResult
from memory_interface import (
    BaseMemoryStrategy,
    MockMemoryStrategy,
    RetrievedEvidence,
    get_strategy,
    list_registered_strategies,
    register_strategy,
)
from memory_retrieval import assemble_memory_context
from model_adapter import (
    BaseLLMAdapter,
    FakeLLMAdapter,
    get_model_adapter,
    heuristic_bpe_token_count,
)


class TestDatasetValidationAndLoading(unittest.TestCase):
    def test_valid_synthetic_example_validation(self):
        suite = get_synthetic_benchmark_suite()
        self.assertGreaterEqual(len(suite), 8)
        for ex in suite:
            is_valid, errors = validate_benchmark_example(ex)
            self.assertTrue(is_valid, f"Synthetic example {ex.id} failed validation: {errors}")
            self.assertEqual(len(errors), 0)

    def test_malformed_example_detection(self):
        # Missing ID
        bad_id = BenchmarkExample(id="", category="cat", sessions=[[{"role": "user", "content": "hi"}]], query="q?", expected_answer="a", acceptable_answers=["a"], supporting_evidence=[])
        is_val, errs = validate_benchmark_example(bad_id)
        self.assertFalse(is_val)
        self.assertTrue(any("id" in e.lower() for e in errs))

        # Missing query
        bad_q = BenchmarkExample(id="ex1", category="cat", sessions=[[{"role": "user", "content": "hi"}]], query="", expected_answer="a", acceptable_answers=["a"], supporting_evidence=[])
        is_val, errs = validate_benchmark_example(bad_q)
        self.assertFalse(is_val)
        self.assertTrue(any("query" in e.lower() for e in errs))

        # Missing expected answer
        bad_ans = BenchmarkExample(id="ex1", category="cat", sessions=[[{"role": "user", "content": "hi"}]], query="q?", expected_answer="", acceptable_answers=[], supporting_evidence=[])
        is_val, errs = validate_benchmark_example(bad_ans)
        self.assertFalse(is_val)
        self.assertTrue(any("expected_answer" in e.lower() for e in errs))

        # Out-of-order turn timestamps within a session
        bad_order = BenchmarkExample(
            id="ex1", category="cat",
            sessions=[[
                {"role": "user", "content": "turn 1", "timestamp": 200.0},
                {"role": "assistant", "content": "turn 2", "timestamp": 100.0},  # Decreasing timestamp
            ]],
            query="q?", expected_answer="a", acceptable_answers=["a"], supporting_evidence=[]
        )
        is_val, errs = validate_benchmark_example(bad_order)
        self.assertFalse(is_val)
        self.assertTrue(any("out of order" in e.lower() for e in errs))

        # Invalid turn role
        bad_role = BenchmarkExample(
            id="ex_role", category="cat",
            sessions=[[{"role": "unknown_actor", "content": "hello", "timestamp": 100.0}]],
            query="q?", expected_answer="a", acceptable_answers=["a"], supporting_evidence=[]
        )
        is_val, errs = validate_benchmark_example(bad_role)
        self.assertFalse(is_val)
        self.assertTrue(any("invalid or missing role" in e.lower() for e in errs))

        # Data leakage: turn content identical to query
        leak_ex = BenchmarkExample(
            id="ex_leak", category="cat",
            sessions=[[{"role": "user", "content": "What is my password?", "timestamp": 100.0}]],
            query="What is my password?", expected_answer="secret", acceptable_answers=[], supporting_evidence=[]
        )
        is_val, errs = validate_benchmark_example(leak_ex)
        self.assertFalse(is_val)
        self.assertTrue(any("leaks benchmark evaluation query" in e.lower() for e in errs))

    def test_lazy_dataset_loading(self):
        # Synthetic loading
        synth = load_benchmark_dataset("synthetic")
        self.assertEqual(len(synth), 8)

        # Dev split loading
        dev = load_benchmark_dataset("longmemeval_dev")
        self.assertEqual(len(dev), 30)

        # Invalid dataset name
        with self.assertRaises(ValueError):
            load_benchmark_dataset("unknown_dataset_split")


class TestModelAdapterFactory(unittest.TestCase):
    def test_fake_adapter_initialization(self):
        adapter = get_model_adapter("fake")
        self.assertIsInstance(adapter, FakeLLMAdapter)
        self.assertFalse(adapter.is_native_tokenizer)
        tokens = adapter.tokenize("test prompt")
        self.assertGreater(len(tokens), 0)

    def test_missing_qwen_path_raises_actionable_error(self):
        # Missing file path
        with self.assertRaises(FileNotFoundError):
            get_model_adapter("qwen", model_path="C:/nonexistent/path/to/model.gguf")

        # Missing path argument entirely when env is unset
        old_env = os.environ.get("QWEN_MODEL_PATH")
        if "QWEN_MODEL_PATH" in os.environ:
            del os.environ["QWEN_MODEL_PATH"]
        try:
            with self.assertRaises(ValueError):
                get_model_adapter("qwen", model_path=None)
        finally:
            if old_env is not None:
                os.environ["QWEN_MODEL_PATH"] = old_env

    def test_unsupported_model_backend_raises_error(self):
        with self.assertRaises(ValueError):
            get_model_adapter("nonexistent_backend")


class TestMemoryStrategyContractAndRegistry(unittest.TestCase):
    def test_strategy_registration_and_retrieval(self):
        @register_strategy("test_custom_dummy")
        class DummyStrategy(BaseMemoryStrategy):
            def process_session(self, conversation_history, session_id, current_timestamp):
                return {"stored": len(conversation_history)}
            def retrieve(self, query, top_k=5):
                return [RetrievedEvidence(content="dummy evidence", score=1.0)]
            def get_persisted_memories(self):
                return [RetrievedEvidence(content="dummy evidence")]

        self.assertIn("test_custom_dummy", list_registered_strategies())
        strat = get_strategy("test_custom_dummy", budget=250)
        self.assertIsInstance(strat, DummyStrategy)
        self.assertEqual(strat.total_budget, 250)

    def test_unknown_strategy_raises_keyerror_with_options(self):
        with self.assertRaises(KeyError) as ctx:
            get_strategy("completely_unknown_strategy")
        self.assertIn("Available strategies", str(ctx.exception))

    def test_strategy_reset_isolation(self):
        strat = MockMemoryStrategy(total_budget=500)
        strat.process_session([{"role": "user", "content": "I live in Seattle.", "timestamp": 100.0}], "s1", 100.0)
        self.assertGreater(strat.get_storage_usage(), 0)

        # Reset clears memory
        strat.reset()
        self.assertEqual(strat.get_storage_usage(), 0)
        self.assertEqual(len(strat.get_persisted_memories()), 0)


class TestContextAssemblyStrategyNeutral(unittest.TestCase):
    def test_assemble_with_retrieved_evidence_objects(self):
        evidence = [
            RetrievedEvidence(content="User works at DeepMind.", score=0.9, lifecycle_status="current"),
            RetrievedEvidence(content="User previously worked at OpenAI.", score=0.8, lifecycle_status="superseded"),
        ]
        res = assemble_memory_context(
            query="Where does the user work?",
            retrieved_memories=evidence,
            max_context_tokens=256
        )
        self.assertIn("User works at DeepMind", res.formatted_context)
        self.assertIn("[CURRENT]", res.formatted_context)
        self.assertIn("[HISTORICAL / SUPERSEDED]", res.formatted_context)

    def test_assemble_with_plain_strings(self):
        string_evidence = [
            "User prefers Python 3.11 with dataclasses.",
            "User prefers dark mode editor theme."
        ]
        res = assemble_memory_context(
            query="What editor theme does the user prefer?",
            retrieved_memories=string_evidence,
            max_context_tokens=256
        )
        self.assertIn("User prefers dark mode editor theme", res.formatted_context)
        self.assertIn("[CURRENT]", res.formatted_context)

    def test_assemble_with_empty_evidence(self):
        res = assemble_memory_context(
            query="Any query",
            retrieved_memories=[],
            max_context_tokens=256
        )
        self.assertEqual(res.formatted_context, "No relevant memories found.")
        self.assertEqual(res.context_token_cost, 0)


class TestExperimentHarnessExecutionAndErrorIsolation(unittest.TestCase):
    def test_mock_memory_end_to_end_evaluation(self):
        llm = FakeLLMAdapter()
        harness = ExperimentHarness(llm_adapter=llm, max_context_tokens=256)
        strat = MockMemoryStrategy(total_budget=500)

        example = BenchmarkExample(
            id="synth_test",
            category="unit_test",
            sessions=[
                [{"role": "user", "content": "I work at DeepMind.", "timestamp": 1000.0}],
                [{"role": "user", "content": "I like PyTorch.", "timestamp": 2000.0}],
            ],
            query="Where do I work?",
            expected_answer="DeepMind",
            acceptable_answers=["DeepMind"],
            supporting_evidence=["I work at DeepMind."]
        )

        res = harness.run_single_comparison(strategy=strat, example=example, budget=500)
        self.assertIsInstance(res, QuestionEvaluationResult)
        self.assertIsNone(res.error)
        self.assertEqual(res.substring_accuracy, 1.0)
        self.assertEqual(res.recall_at_k, 1.0)
        self.assertLessEqual(res.storage_tokens_used, 500)

    def test_harness_catches_strategy_error_without_crashing(self):
        class BrokenStrategy(BaseMemoryStrategy):
            def process_session(self, conversation_history, session_id, current_timestamp):
                raise RuntimeError("Simulated crash in session ingestion")
            def retrieve(self, query, top_k=5):
                return []
            def get_persisted_memories(self):
                return []

        llm = FakeLLMAdapter()
        harness = ExperimentHarness(llm_adapter=llm)
        strat = BrokenStrategy(name="broken", total_budget=500)

        example = BenchmarkExample(
            id="fail_ex",
            category="failure_test",
            sessions=[[{"role": "user", "content": "crash test"}]],
            query="any query",
            expected_answer="ans",
            acceptable_answers=["ans"],
            supporting_evidence=[]
        )

        # Must not raise an exception; should capture error in result
        res = harness.run_single_comparison(strategy=strat, example=example, budget=500)
        self.assertIsNotNone(res.error)
        self.assertIn("Simulated crash", res.error)
        self.assertEqual(res.exact_match, 0.0)

    def test_state_isolation_across_questions_and_persistence_across_sessions(self):
        """
        Regression Test:
        1. Multi-session intake for Question 1 must accumulate across sessions.
        2. Clean reset between Question 1 and Question 2 must prevent Question 1 facts from leaking into Question 2.
        """
        llm = FakeLLMAdapter()
        harness = ExperimentHarness(llm_adapter=llm)
        strat = MockMemoryStrategy(total_budget=500)

        # Question 1 with 2 sessions
        q1 = BenchmarkExample(
            id="q1_multi_session",
            category="isolation_test",
            sessions=[
                [{"role": "user", "content": "My private secret key is OMEGA-99.", "timestamp": 100.0}],
                [{"role": "user", "content": "My favorite city is Zurich.", "timestamp": 200.0}],
            ],
            query="What is my private secret key and favorite city?",
            expected_answer="OMEGA-99 Zurich",
            acceptable_answers=["OMEGA-99", "Zurich"],
            supporting_evidence=["My private secret key is OMEGA-99.", "My favorite city is Zurich."]
        )

        res1 = harness.run_single_comparison(strategy=strat, example=q1, budget=500)
        self.assertIsNone(res1.error)
        # Verify both session turns are present in Question 1's persisted memories
        q1_contents = [m.content for m in strat.get_persisted_memories()]
        self.assertTrue(any("OMEGA-99" in c for c in q1_contents), "Session 1 memory missing from Q1")
        self.assertTrue(any("Zurich" in c for c in q1_contents), "Session 2 memory missing from Q1")

        # Question 2 (completely different question)
        q2 = BenchmarkExample(
            id="q2_isolated",
            category="isolation_test",
            sessions=[
                [{"role": "user", "content": "My pet is a green parrot named Kiwi.", "timestamp": 500.0}],
            ],
            query="What is my private secret key?",  # Query asking about Q1's secret key
            expected_answer="Unknown",
            acceptable_answers=[],
            supporting_evidence=[]
        )

        res2 = harness.run_single_comparison(strategy=strat, example=q2, budget=500)
        self.assertIsNone(res2.error)
        # Verify Question 2's persisted memories contain ONLY Question 2 content
        q2_contents = [m.content for m in strat.get_persisted_memories()]
        self.assertTrue(any("Kiwi" in c for c in q2_contents), "Q2 memory not stored")
        self.assertFalse(any("OMEGA-99" in c for c in q2_contents), "State leakage: Q1 secret key leaked into Q2!")
        self.assertFalse(any("Zurich" in c for c in q2_contents), "State leakage: Q1 city leaked into Q2!")
        # Verify retrieved items for Q2 do not contain Q1 facts
        self.assertFalse(any("OMEGA-99" in c for c in res2.retrieved_contents), "Q1 leaked into Q2 retrieval!")

    def test_aggregate_results_failure_metrics(self):
        """
        Verify that failures are explicitly counted, included in the denominator,
        and not counted as correct in summary accuracy.
        """
        harness = ExperimentHarness(llm_adapter=FakeLLMAdapter())

        success_res = QuestionEvaluationResult(
            question_id="q1", category="cat", strategy_name="mock", budget=100,
            user_query="q", expected_answer="a", generated_response="a",
            retrieved_memory_ids=[], retrieved_contents=[],
            exact_match=1.0, substring_accuracy=1.0, token_f1=1.0,
            recall_at_k=1.0, precision_at_k=1.0, mrr=1.0, stale_memory_error=0.0,
            storage_tokens_used=50, context_tokens_used=50,
            retrieval_latency_ms=1.0, generation_latency_ms=2.0, total_latency_ms=3.0,
            error=None
        )

        failed_res = QuestionEvaluationResult(
            question_id="q2", category="cat", strategy_name="mock", budget=100,
            user_query="q", expected_answer="a", generated_response="ERROR",
            retrieved_memory_ids=[], retrieved_contents=[],
            exact_match=0.0, substring_accuracy=0.0, token_f1=0.0,
            recall_at_k=0.0, precision_at_k=0.0, mrr=0.0, stale_memory_error=0.0,
            storage_tokens_used=0, context_tokens_used=0,
            retrieval_latency_ms=0.0, generation_latency_ms=0.0, total_latency_ms=0.0,
            error="CUDA out of memory"
        )

        summary = harness.aggregate_results([success_res, failed_res], budgets=[100])
        m = summary["mock"][100]

        self.assertEqual(m["count"], 2)
        self.assertEqual(m["succeeded_count"], 1)
        self.assertEqual(m["failed_count"], 1)
        # Denominator must be 2, so accuracy is 0.5 (1/2), NOT 1.0
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["exact_match"], 0.5)


if __name__ == "__main__":
    unittest.main()
