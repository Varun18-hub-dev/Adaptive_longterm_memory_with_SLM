"""
Comprehensive Unit and Integration Tests for Phase 3 (Hardened):
Query Retrieval, Relevance Filtering, Temporal Validity, Context Assembly, and Model Adapter.
"""

import unittest

from memory_core import (
    LifecycleStatus,
    MemoryItem,
    MemoryType,
    estimate_token_cost,
)
from memory_lifecycle import AdaptiveMemoryManager
from memory_retrieval import (
    PromptBudgetTracker,
    RetrievalWeights,
    assemble_memory_context,
    compute_query_relevance,
    extract_query_point_in_time,
    is_historical_query,
    is_memory_valid_at_timestamp,
    retrieve_memories,
    run_phase3_pipeline,
)
from model_adapter import FakeLLMAdapter


class TestModelAdapter(unittest.TestCase):
    def test_fake_llm_adapter_basic_and_canned(self):
        adapter = FakeLLMAdapter(canned_response="Mock answer.")
        res = adapter.generate("Hello?")
        self.assertEqual(res, "Mock answer.")

        # Test chat generation
        chat_res = adapter.generate_chat([{"role": "user", "content": "test"}])
        self.assertEqual(chat_res, "Mock answer.")
        self.assertEqual(len(adapter.call_history), 2)

    def test_fake_llm_adapter_token_counting(self):
        adapter = FakeLLMAdapter()
        tokens = adapter.tokenize("Where do I work now?")
        self.assertGreater(len(tokens), 0)
        self.assertEqual(adapter.count_tokens("Where do I work now?"), len(tokens))

        chat_tokens = adapter.count_chat_tokens([{"role": "user", "content": "hello"}])
        self.assertGreater(chat_tokens, adapter.count_tokens("hello"))

    def test_fake_llm_adapter_heuristic_answering(self):
        adapter = FakeLLMAdapter()
        ans_current = adapter.generate("Context: DeepMind\nWhere do I work?")
        self.assertIn("DeepMind", ans_current)

        ans_prev = adapter.generate("Context: OpenAI, DeepMind\nWhere did I work previously?")
        self.assertIn("OpenAI", ans_prev)


class TestQueryIntentAndRelevance(unittest.TestCase):
    def test_is_historical_query_detection(self):
        self.assertTrue(is_historical_query("Where did I work previously?"))
        self.assertTrue(is_historical_query("What was my old preference?"))
        self.assertTrue(is_historical_query("What company did I work at before?"))
        self.assertFalse(is_historical_query("Where do I work now?"))
        self.assertFalse(is_historical_query("What framework should I use today?"))

    def test_relevance_computation(self):
        mem_work = MemoryItem(
            id="1", content="I work at DeepMind.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s", timestamp=0, metadata={"slot_type": "employment"}
        )
        mem_code = MemoryItem(
            id="2", content="I always prefer PyTorch.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s", timestamp=0, metadata={"slot_type": "dl_framework"}
        )

        rel_work = compute_query_relevance("What company do I work for?", mem_work)
        rel_code = compute_query_relevance("What company do I work for?", mem_code)

        self.assertGreater(rel_work, rel_code)
        self.assertGreater(rel_work, 0.40)


class TestHardenedRetrievalAndDistractorExclusion(unittest.TestCase):
    def setUp(self):
        self.employer_mem = MemoryItem(
            id="cur-1", content="I work at DeepMind now.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s2", timestamp=2000.0, utility_score=0.85, recency=1.0,
            stability=0.8, lifecycle_status=LifecycleStatus.ACTIVE,
            metadata={"slot_type": "employment", "slot_value": "deepmind"}
        )
        self.cuda_debug_task = MemoryItem(
            id="task-1", content="Can you help me debug this CUDA issue right now?",
            memory_type=MemoryType.EPISODIC, source_session_id="s1", timestamp=1000.0,
            utility_score=0.40, recency=0.8, stability=0.2,
            lifecycle_status=LifecycleStatus.TEMPORARY
        )
        self.superseded_employer = MemoryItem(
            id="old-1", content="I work at OpenAI.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s1", timestamp=1000.0, utility_score=0.5, recency=0.4,
            stability=0.8, lifecycle_status=LifecycleStatus.SUPERSEDED,
            metadata={"slot_type": "employment", "slot_value": "openai", "superseded_at": 2000.0}
        )

    def test_distractor_task_excluded_under_competition(self):
        # Query asking for employer should retrieve employer and PRUNE the CUDA task
        query = "Where do I work now?"
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[self.cuda_debug_task],
            semantic_store=[self.employer_mem],
            archive_store=[self.superseded_employer],
            top_k=5
        )

        retrieved_ids = [m[0].id for m in retrieved]
        self.assertIn("cur-1", retrieved_ids, "Relevant employer memory must be retrieved")
        self.assertNotIn("task-1", retrieved_ids, "Irrelevant transient task must be excluded")
        self.assertNotIn("old-1", retrieved_ids, "Superseded memory must be excluded for current query")

    def test_point_in_time_query(self):
        # At timestamp 1500.0, OpenAI was valid, DeepMind was not yet created
        query = "Where did I work at timestamp 1500.0?"
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[],
            semantic_store=[self.employer_mem],
            archive_store=[self.superseded_employer],
            top_k=5
        )

        retrieved_ids = [m[0].id for m in retrieved]
        self.assertIn("old-1", retrieved_ids, "At t=1500, OpenAI was the active fact")
        self.assertNotIn("cur-1", retrieved_ids, "DeepMind (created at t=2000) was in the future at t=1500")


class TestContextAssemblyAndPromptBudgetAccounting(unittest.TestCase):
    def test_budget_tracker_accounting(self):
        tracker = PromptBudgetTracker(
            model_context_window=1000,
            generation_allowance=200,
            system_prompt_tokens=150,
            query_tokens=50,
            chat_template_overhead=20,
            max_memory_context_tokens=400
        )
        # Available = 1000 - (150 + 50 + 20 + 200) = 580
        # Capped by max_memory_context_tokens = 400
        allowed = tracker.calculate_allowed_memory_budget()
        self.assertEqual(allowed, 400)

        # When window is tight:
        tracker_tight = PromptBudgetTracker(
            model_context_window=500,
            generation_allowance=200,
            system_prompt_tokens=150,
            query_tokens=50,
            chat_template_overhead=20,
            max_memory_context_tokens=400
        )
        # Available = 500 - 420 = 80
        self.assertEqual(tracker_tight.calculate_allowed_memory_budget(), 80)

    def test_prompt_budget_strictly_enforced(self):
        items = [
            (MemoryItem(id=f"mem-{i}", content=f"User preference statement number {i} with extra text words.",
                        memory_type=MemoryType.SEMANTIC, source_session_id="s", timestamp=0,
                        token_cost=15, lifecycle_status=LifecycleStatus.ACTIVE), 0.9 - i * 0.05)
            for i in range(10)
        ]

        res = assemble_memory_context("query", items, max_context_tokens=75)
        self.assertLessEqual(res.context_token_cost, 75)
        self.assertTrue(len(res.selected_memories) < len(items))

    def test_duplicate_context_suppressed(self):
        mem1 = MemoryItem(id="1", content="I prefer PyTorch.", memory_type=MemoryType.SEMANTIC,
                          source_session_id="s", timestamp=0, lifecycle_status=LifecycleStatus.ACTIVE)
        mem2 = MemoryItem(id="2", content="I prefer PyTorch.", memory_type=MemoryType.SEMANTIC,
                          source_session_id="s", timestamp=0, lifecycle_status=LifecycleStatus.ACTIVE)

        res = assemble_memory_context("query", [(mem1, 0.9), (mem2, 0.85)], max_context_tokens=256)
        self.assertEqual(len(res.selected_memories), 1, "Duplicate memory content should only appear once in context")


class TestEndToEndPhase3Pipeline(unittest.TestCase):
    def test_full_pipeline_with_fake_adapter(self):
        conversation = [
            {"role": "user", "content": "I work at OpenAI. I love PyTorch.", "timestamp": 1000.0},
            {"role": "assistant", "content": "Noted.", "timestamp": 1010.0},
            {"role": "user", "content": "I moved to DeepMind. I work at DeepMind now.", "timestamp": 2000.0},
            {"role": "assistant", "content": "Congrats on DeepMind!", "timestamp": 2010.0},
            {"role": "user", "content": "Can you debug this CUDA issue right now?", "timestamp": 2020.0},
        ]

        fake_llm = FakeLLMAdapter()

        # Query 1: Current company (must retrieve DeepMind and exclude CUDA issue)
        result_current = run_phase3_pipeline(
            conversation_history=conversation,
            user_query="Where do I work now?",
            llm_adapter=fake_llm,
            storage_budget=200,
            max_context_tokens=256,
            current_timestamp=2050.0
        )

        self.assertFalse(result_current["is_historical_query"])
        self.assertIn("DeepMind", result_current["response"])
        self.assertIn("DeepMind", result_current["context_result"]["formatted_context"])
        self.assertNotIn("CUDA", result_current["context_result"]["formatted_context"])

        # Query 2: Historical company
        result_prev = run_phase3_pipeline(
            conversation_history=[],
            user_query="Where did I work previously before DeepMind?",
            llm_adapter=fake_llm,
            storage_budget=200,
            max_context_tokens=256,
            existing_manager=result_current["manager"],
            current_timestamp=2050.0
        )

        self.assertTrue(result_prev["is_historical_query"])
        self.assertIn("OpenAI", result_prev["response"])


if __name__ == "__main__":
    unittest.main()
