"""
Comprehensive Unit and Integration Tests for Phase 3:
Query Retrieval, Temporal Validity, Context Assembly, and Model Adapter.
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
    RetrievalWeights,
    assemble_memory_context,
    compute_query_relevance,
    is_historical_query,
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

    def test_fake_llm_adapter_heuristic_answering(self):
        adapter = FakeLLMAdapter()
        # Answers current company
        ans_current = adapter.generate("Context: DeepMind\nWhere do I work?")
        self.assertIn("DeepMind", ans_current)

        # Answers historical company
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


class TestRetrievalTemporalValidity(unittest.TestCase):
    def setUp(self):
        self.active_mem = MemoryItem(
            id="cur-1", content="I work at DeepMind now.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s2", timestamp=2000.0, utility_score=0.8, recency=1.0,
            stability=0.8, lifecycle_status=LifecycleStatus.ACTIVE,
            metadata={"slot_type": "employment", "slot_value": "deepmind"}
        )
        self.superseded_mem = MemoryItem(
            id="old-1", content="I work at OpenAI.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s1", timestamp=1000.0, utility_score=0.5, recency=0.4,
            stability=0.8, lifecycle_status=LifecycleStatus.SUPERSEDED,
            metadata={"slot_type": "employment", "slot_value": "openai", "superseded_by_id": "cur-1"}
        )
        self.expired_mem = MemoryItem(
            id="exp-1", content="Debugging line 23 error.", memory_type=MemoryType.EPISODIC,
            source_session_id="s1", timestamp=500.0, utility_score=0.2, recency=0.1,
            stability=0.2, lifecycle_status=LifecycleStatus.EXPIRED
        )

    def test_current_query_strictly_excludes_superseded_and_expired(self):
        query = "Where do I work now?"
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[],
            semantic_store=[self.active_mem, self.superseded_mem],
            archive_store=[self.expired_mem]
        )
        retrieved_ids = [m[0].id for m in retrieved]

        # Must include current fact
        self.assertIn("cur-1", retrieved_ids)
        # Must NEVER include superseded or expired fact for current query
        self.assertNotIn("old-1", retrieved_ids)
        self.assertNotIn("exp-1", retrieved_ids)

    def test_historical_query_retrieves_superseded_facts(self):
        query = "Where did I work previously before DeepMind?"
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[],
            semantic_store=[self.active_mem],
            archive_store=[self.superseded_mem, self.expired_mem]
        )
        retrieved_ids = [m[0].id for m in retrieved]

        # Historical query surfaces the superseded item
        self.assertIn("old-1", retrieved_ids)

    def test_empty_stores_retrieval(self):
        retrieved = retrieve_memories("Where do I live?", [], [], [])
        self.assertEqual(retrieved, [])


class TestContextAssemblyAndPromptBudget(unittest.TestCase):
    def test_prompt_budget_strictly_enforced(self):
        items = [
            (MemoryItem(id=f"mem-{i}", content=f"User preference statement number {i} with additional detail text.",
                        memory_type=MemoryType.SEMANTIC, source_session_id="s", timestamp=0,
                        token_cost=15, lifecycle_status=LifecycleStatus.ACTIVE), 0.9 - i * 0.05)
            for i in range(10)
        ]

        # Total tokens of all 10 items would exceed 150
        max_context = 90
        res = assemble_memory_context("query", items, max_context_tokens=max_context)

        # Verified strict enforcement of prompt context budget
        self.assertLessEqual(res.context_token_cost, max_context)
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
        ]

        fake_llm = FakeLLMAdapter()

        # Query 1: Current company
        result_current = run_phase3_pipeline(
            conversation_history=conversation,
            user_query="Where do I work now?",
            llm_adapter=fake_llm,
            storage_budget=200,
            max_context_tokens=256,
            current_timestamp=2020.0
        )

        self.assertFalse(result_current["is_historical_query"])
        self.assertIn("DeepMind", result_current["response"])
        self.assertIn("[CURRENT]", result_current["context_result"]["formatted_context"])
        self.assertNotIn("[HISTORICAL / SUPERSEDED]", result_current["context_result"]["formatted_context"])

        # Query 2: Historical company (reusing the same manager)
        result_prev = run_phase3_pipeline(
            conversation_history=[],
            user_query="Where did I work previously before my current job?",
            llm_adapter=fake_llm,
            storage_budget=200,
            max_context_tokens=256,
            existing_manager=result_current["manager"],
            current_timestamp=2020.0
        )

        self.assertTrue(result_prev["is_historical_query"])
        self.assertIn("OpenAI", result_prev["response"])


if __name__ == "__main__":
    unittest.main()
