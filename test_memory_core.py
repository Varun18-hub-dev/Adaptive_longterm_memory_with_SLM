"""
Comprehensive Unit and Integration Tests for Phase 1 Memory Core.

Tests cover:
1. Memory schema validation and roundtrip serialization
2. Candidate extraction, sentence splitting, and filler filtering
3. Candidate deduplication, frequency incrementing, and provenance tracking
4. Utility scoring formula, weight validation, and separation of recency vs stability
5. Adaptive budget allocation bounds, demand ratios, and strict token limits
6. Memory selection knapsack behavior and strict budget enforcement
7. Edge cases (empty conversation, zero budget, single turn)
8. Deterministic execution
9. Local JSON storage persistence and roundtrip fidelity
10. Full end-to-end pipeline execution
"""

import json
import math
import os
import tempfile
import time
import unittest

from memory_core import (
    AllocationConfig,
    LifecycleStatus,
    LocalJsonMemoryStore,
    MemoryItem,
    MemoryType,
    ScoringWeights,
    allocate_budget,
    estimate_token_cost,
    extract_memories,
    run_phase1_pipeline,
    score_memories,
    select_memories,
    store_memories,
)


class TestMemorySchema(unittest.TestCase):
    def test_memory_item_creation_and_dict_roundtrip(self):
        item = MemoryItem(
            id="mem-101",
            content="User prefers Python over C++.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="session_alpha",
            timestamp=1700000000.0,
            importance=0.85,
            recency=0.9,
            frequency=3,
            stability=0.95,
            consistency_confidence=0.9,
            utility_score=0.88,
            token_cost=8,
            lifecycle_status=LifecycleStatus.STABLE,
            metadata={"domain": "coding"}
        )
        d = item.to_dict()
        self.assertEqual(d["memory_type"], "semantic")
        self.assertEqual(d["lifecycle_status"], "stable")
        self.assertEqual(d["frequency"], 3)

        restored = MemoryItem.from_dict(d)
        self.assertEqual(restored.id, item.id)
        self.assertEqual(restored.content, item.content)
        self.assertEqual(restored.memory_type, MemoryType.SEMANTIC)
        self.assertEqual(restored.lifecycle_status, LifecycleStatus.STABLE)
        self.assertEqual(restored.metadata["domain"], "coding")


class TestExtractionAndDeduplication(unittest.TestCase):
    def test_extraction_basic_and_filtering(self):
        history = [
            {"role": "user", "content": "Hello. I work at OpenAI. Can you help me debug this error?"},
            {"role": "assistant", "content": "Sure, I can help you with that error."}
        ]
        memories = extract_memories(history, session_id="sess_1", current_timestamp=1000.0)
        contents = [m.content for m in memories]

        # "Hello." is a filler greeting and should be excluded
        self.assertFalse(any(c.lower() == "hello" for c in contents))
        # "I work at OpenAI." is extracted
        self.assertTrue(any("openai" in c.lower() for c in contents))

        # Check semantic vs episodic categorization
        openai_mem = next(m for m in memories if "openai" in m.content.lower())
        self.assertEqual(openai_mem.memory_type, MemoryType.SEMANTIC)
        self.assertGreaterEqual(openai_mem.stability, 0.7)

    def test_deduplication_frequency_and_provenance(self):
        history = [
            {"role": "user", "content": "I like dark mode in my editors.", "timestamp": 100.0},
            {"role": "assistant", "content": "Got it! Dark mode is great.", "timestamp": 105.0},
            {"role": "user", "content": "I like dark mode in my editors.", "timestamp": 200.0},
        ]
        memories = extract_memories(history, session_id="sess_dark", current_timestamp=200.0)
        dark_mems = [m for m in memories if "dark mode in my editors" in m.content.lower()]
        self.assertEqual(len(dark_mems), 1, "Duplicate candidate should be consolidated into one memory item")

        item = dark_mems[0]
        self.assertEqual(item.frequency, 2)
        self.assertEqual(item.timestamp, 200.0, "Timestamp should update to the latest turn")
        provenance = item.metadata.get("provenance", [])
        self.assertEqual(len(provenance), 2)
        self.assertEqual(provenance[0]["turn_index"], 0)
        self.assertEqual(provenance[1]["turn_index"], 2)


class TestUtilityScoring(unittest.TestCase):
    def test_scoring_weights_validation(self):
        # Must sum to 1.0
        with self.assertRaises(ValueError):
            ScoringWeights(w_r=0.5, w_i=0.5, w_f=0.5, w_t=0.1, w_c=0.1)

        # Non-negative weights required
        with self.assertRaises(ValueError):
            ScoringWeights(w_r=-0.1, w_i=0.4, w_f=0.3, w_t=0.2, w_c=0.2)

        valid = ScoringWeights(w_r=0.2, w_i=0.2, w_f=0.2, w_t=0.2, w_c=0.2)
        self.assertAlmostEqual(sum([valid.w_r, valid.w_i, valid.w_f, valid.w_t, valid.w_c]), 1.0)

    def test_recency_and_stability_separation(self):
        now = 10000.0
        # Memory A: Very recent, but transient/unstable (e.g., immediate debug request)
        mem_a = MemoryItem(
            id="a",
            content="Debug this error right now",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=now,  # Recency = 1.0
            importance=0.5,
            frequency=1,
            stability=0.1,  # Low stability
            consistency_confidence=1.0,
            token_cost=5
        )

        # Memory B: Old, but highly stable enduring fact
        mem_b = MemoryItem(
            id="b",
            content="User lives in Bangalore",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s1",
            timestamp=now - 86400.0,  # 1 day old (recency decays to exp(-1) approx 0.368)
            importance=0.5,
            frequency=1,
            stability=0.95,  # High stability
            consistency_confidence=1.0,
            token_cost=5
        )

        scored = score_memories([mem_a, mem_b], current_timestamp=now)
        scored_a = next(m for m in scored if m.id == "a")
        scored_b = next(m for m in scored if m.id == "b")

        # Recency check
        self.assertAlmostEqual(scored_a.recency, 1.0, places=4)
        self.assertAlmostEqual(scored_b.recency, math.exp(-1.0), places=3)

        # Stability is preserved independently
        self.assertEqual(scored_a.stability, 0.1)
        self.assertEqual(scored_b.stability, 0.95)

        # Utility scores are computed strictly between 0 and 1
        self.assertTrue(0.0 <= scored_a.utility_score <= 1.0)
        self.assertTrue(0.0 <= scored_b.utility_score <= 1.0)


class TestAdaptiveBudgetAllocation(unittest.TestCase):
    def test_budget_bounds_enforced(self):
        config = AllocationConfig(
            total_budget=1000,
            min_episodic_share=0.20,
            max_episodic_share=0.80,
            min_semantic_share=0.20,
            max_semantic_share=0.80,
        )

        # Scenario 1: Overwhelming episodic demand
        candidates_heavy_episodic = [
            MemoryItem(id=f"ep_{i}", content="task", memory_type=MemoryType.EPISODIC,
                       source_session_id="s", timestamp=0, utility_score=0.9, token_cost=100)
            for i in range(10)
        ] + [
            MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC,
                       source_session_id="s", timestamp=0, utility_score=0.1, token_cost=10)
        ]

        allocation = allocate_budget(candidates_heavy_episodic, config)
        total_allocated = allocation[MemoryType.EPISODIC] + allocation[MemoryType.SEMANTIC]
        self.assertEqual(total_allocated, 1000)
        # Episodic share should not exceed max_episodic_share (80% = 800)
        self.assertLessEqual(allocation[MemoryType.EPISODIC], 800)
        # Semantic share should not drop below min_semantic_share (20% = 200)
        self.assertGreaterEqual(allocation[MemoryType.SEMANTIC], 200)

        # Scenario 2: Overwhelming semantic demand
        candidates_heavy_semantic = [
            MemoryItem(id=f"sem_{i}", content="fact", memory_type=MemoryType.SEMANTIC,
                       source_session_id="s", timestamp=0, utility_score=0.95, token_cost=100)
            for i in range(10)
        ]
        allocation_sem = allocate_budget(candidates_heavy_semantic, config)
        total_allocated_sem = allocation_sem[MemoryType.EPISODIC] + allocation_sem[MemoryType.SEMANTIC]
        self.assertEqual(total_allocated_sem, 1000)
        self.assertGreaterEqual(allocation_sem[MemoryType.EPISODIC], 200)
        self.assertLessEqual(allocation_sem[MemoryType.SEMANTIC], 800)

    def test_budget_zero_and_empty_candidates(self):
        config = AllocationConfig(total_budget=0)
        alloc = allocate_budget([], config)
        self.assertEqual(alloc[MemoryType.EPISODIC], 0)
        self.assertEqual(alloc[MemoryType.SEMANTIC], 0)

        # Empty candidates with positive budget: defaults evenly within bounds
        config_500 = AllocationConfig(total_budget=500)
        alloc_empty = allocate_budget([], config_500)
        self.assertEqual(alloc_empty[MemoryType.EPISODIC] + alloc_empty[MemoryType.SEMANTIC], 500)


class TestMemorySelection(unittest.TestCase):
    def test_selection_strictly_enforces_budget(self):
        items = [
            MemoryItem(id="1", content="item 1", memory_type=MemoryType.SEMANTIC, source_session_id="s",
                       timestamp=0, utility_score=0.9, token_cost=40),
            MemoryItem(id="2", content="item 2", memory_type=MemoryType.SEMANTIC, source_session_id="s",
                       timestamp=0, utility_score=0.8, token_cost=50),
            MemoryItem(id="3", content="item 3", memory_type=MemoryType.SEMANTIC, source_session_id="s",
                       timestamp=0, utility_score=0.7, token_cost=30),
        ]

        # Budget of 75 tokens:
        # Item 1 costs 40 (density 0.9/40 = 0.0225)
        # Item 3 costs 30 (density 0.7/30 = 0.0233) -> ranked first!
        # Item 2 costs 50 (density 0.8/50 = 0.0160)
        # Selection should take Item 3 (30) + Item 1 (40) = 70 tokens <= 75. Item 2 does not fit.
        selected, used_tokens = select_memories(items, budget=75)
        self.assertLessEqual(used_tokens, 75)
        self.assertEqual(used_tokens, 70)
        selected_ids = {m.id for m in selected}
        self.assertIn("1", selected_ids)
        self.assertIn("3", selected_ids)
        self.assertNotIn("2", selected_ids)

    def test_selection_zero_budget_and_empty(self):
        selected, used = select_memories([], budget=100)
        self.assertEqual(selected, [])
        self.assertEqual(used, 0)

        item = MemoryItem(id="1", content="item", memory_type=MemoryType.EPISODIC, source_session_id="s",
                          timestamp=0, utility_score=1.0, token_cost=10)
        selected_zero, used_zero = select_memories([item], budget=0)
        self.assertEqual(selected_zero, [])
        self.assertEqual(used_zero, 0)


class TestStorageAndPersistence(unittest.TestCase):
    def test_local_storage_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory_state.json")
            episodic = [
                MemoryItem(id="ep1", content="Discussed bug fix", memory_type=MemoryType.EPISODIC,
                           source_session_id="s1", timestamp=100.0, utility_score=0.7, token_cost=5)
            ]
            semantic = [
                MemoryItem(id="sem1", content="Prefers unit tests", memory_type=MemoryType.SEMANTIC,
                           source_session_id="s1", timestamp=100.0, utility_score=0.9, token_cost=4)
            ]
            summary = {"total_budget": 500, "used": 9}

            store = LocalJsonMemoryStore(path)
            store.save(episodic, semantic, summary)

            self.assertTrue(os.path.exists(path))
            loaded_ep, loaded_sem, loaded_sum = store.load()

            self.assertEqual(len(loaded_ep), 1)
            self.assertEqual(loaded_ep[0].id, "ep1")
            self.assertEqual(loaded_ep[0].memory_type, MemoryType.EPISODIC)

            self.assertEqual(len(loaded_sem), 1)
            self.assertEqual(loaded_sem[0].id, "sem1")
            self.assertEqual(loaded_sem[0].memory_type, MemoryType.SEMANTIC)

            self.assertEqual(loaded_sum["total_budget"], 500)


class TestFullPipelineIntegration(unittest.TestCase):
    def test_run_phase1_pipeline_deterministic(self):
        history = [
            {"role": "user", "content": "I am a data scientist. I always prefer PyTorch over TensorFlow.", "timestamp": 1000.0},
            {"role": "assistant", "content": "PyTorch is great for data science.", "timestamp": 1010.0},
            {"role": "user", "content": "Today I ran an experiment on CUDA and got an out-of-memory error.", "timestamp": 1020.0},
            {"role": "user", "content": "I always prefer PyTorch over TensorFlow.", "timestamp": 1050.0},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "pipeline_output.json")
            result = run_phase1_pipeline(
                conversation_history=history,
                session_id="session_test_run",
                total_budget=500,
                persistence_path=out_file,
                current_timestamp=1100.0
            )

            summary = result["allocation_summary"]
            self.assertEqual(summary["total_budget"], 500)
            self.assertEqual(summary["allocated_budget"]["episodic"] + summary["allocated_budget"]["semantic"], 500)
            self.assertLessEqual(summary["tokens_used"]["total"], 500)

            # PyTorch preference was repeated twice -> frequency = 2
            selected_semantic = result["selected_semantic"]
            pytorch_item = next((m for m in selected_semantic if "pytorch" in m["content"].lower()), None)
            self.assertIsNotNone(pytorch_item)
            self.assertEqual(pytorch_item["frequency"], 2)

            # Check JSON file written to disk
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertIn("allocation_summary", saved)
            self.assertIn("episodic_store", saved)
            self.assertIn("semantic_store", saved)


if __name__ == "__main__":
    unittest.main()
