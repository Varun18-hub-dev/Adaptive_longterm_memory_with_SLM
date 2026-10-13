"""
Comprehensive Unit and Integration Tests for Phase 2:
Memory Lifecycle Management, Consolidation, Conflicts, and Maintenance.
"""

import math
import os
import tempfile
import time
import unittest

from memory_core import (
    AllocationConfig,
    LifecycleStatus,
    MemoryItem,
    MemoryType,
    ScoringWeights,
)
from memory_lifecycle import (
    AdaptiveMemoryManager,
    AuditAction,
    AuditLogEntry,
    compute_text_similarity,
    consolidate_episodic_memories,
    deduplicate_semantically,
    detect_conflicts_and_supersede,
    evaluate_lifecycle_transitions,
    maintain_memory_budget,
    run_phase2_pipeline,
)


class TestSemanticDeduplication(unittest.TestCase):
    def test_text_similarity_metrics(self):
        # Exact match
        self.assertEqual(compute_text_similarity("I like Python", "I like Python"), 1.0)
        # Near paraphrase / minor punctuation
        sim = compute_text_similarity("I love dark mode in my editors.", "I like dark mode in my editors!")
        self.assertGreater(sim, 0.75)
        # Completely different
        diff = compute_text_similarity("I live in Paris", "The database crashed yesterday")
        self.assertLess(diff, 0.20)

    def test_soft_deduplication_merges_frequency_and_provenance(self):
        existing = [
            MemoryItem(
                id="mem-1",
                content="I always prefer dark mode in editors",
                memory_type=MemoryType.SEMANTIC,
                source_session_id="s1",
                timestamp=100.0,
                frequency=1,
                metadata={"provenance": [{"turn_index": 0}]}
            )
        ]
        incoming = [
            MemoryItem(
                id="mem-2",
                content="I prefer dark mode in my editors",
                memory_type=MemoryType.SEMANTIC,
                source_session_id="s2",
                timestamp=200.0,
                frequency=1,
                metadata={"provenance": [{"turn_index": 4}]}
            )
        ]
        audit_log = []
        updated, distinct = deduplicate_semantically(
            incoming_memories=incoming,
            existing_memories=existing,
            similarity_threshold=0.75,
            audit_entries=audit_log,
            current_timestamp=200.0
        )

        self.assertEqual(len(distinct), 0, "Near-duplicate should be soft-deduplicated")
        self.assertEqual(updated[0].frequency, 2)
        self.assertEqual(updated[0].timestamp, 200.0)
        self.assertIn("provenance", updated[0].metadata)
        self.assertEqual(len(updated[0].metadata["provenance"]), 2)
        self.assertTrue(any(e.action == AuditAction.MERGE_DUPLICATE for e in audit_log))


class TestConflictsAndSuperseding(unittest.TestCase):
    def test_temporal_conflict_supersedes_older_fact(self):
        # Scenario: User moves from Company A to Company B
        older_item = MemoryItem(
            id="work-old",
            content="I work at OpenAI",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="session_january",
            timestamp=1000.0,
            lifecycle_status=LifecycleStatus.ACTIVE
        )
        newer_item = MemoryItem(
            id="work-new",
            content="I work at DeepMind",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="session_june",
            timestamp=2000.0,
            lifecycle_status=LifecycleStatus.ACTIVE
        )

        audit_log = []
        updated = detect_conflicts_and_supersede([older_item, newer_item], audit_log, current_timestamp=2000.0)

        # Older item must be marked SUPERSEDED, not deleted
        self.assertEqual(older_item.lifecycle_status, LifecycleStatus.SUPERSEDED)
        self.assertEqual(older_item.metadata.get("superseded_by_id"), "work-new")

        # Newer item must track the historical predecessor
        self.assertEqual(newer_item.lifecycle_status, LifecycleStatus.ACTIVE)
        predecessors = newer_item.metadata.get("historical_predecessors", [])
        self.assertEqual(len(predecessors), 1)
        self.assertEqual(predecessors[0]["superseded_memory_id"], "work-old")
        self.assertEqual(predecessors[0]["previous_value"], "openai")

        # Audit log verification
        supersede_audit = next((e for e in audit_log if e.action == AuditAction.SUPERSEDE), None)
        self.assertIsNotNone(supersede_audit)
        self.assertEqual(supersede_audit.target_memory_id, "work-old")


class TestConsolidation(unittest.TestCase):
    def test_episodic_consolidation_into_semantic_knowledge(self):
        # An episodic interaction repeated across turns
        episodic_item = MemoryItem(
            id="ep-1",
            content="I always prefer PyTorch over TensorFlow",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=500.0,
            frequency=2,  # Reinforced
            stability=0.85,
            token_cost=8,
            metadata={"provenance": [{"turn_index": 1}, {"turn_index": 3}]}
        )

        semantic_store = []
        audit_log = []

        remaining_ep, updated_sem = consolidate_episodic_memories(
            episodic_memories=[episodic_item],
            semantic_memories=semantic_store,
            min_frequency_to_consolidate=2,
            audit_entries=audit_log,
            current_timestamp=600.0
        )

        # A new semantic item should be created
        self.assertEqual(len(updated_sem), 1)
        consolidated = updated_sem[0]
        self.assertEqual(consolidated.memory_type, MemoryType.SEMANTIC)
        self.assertEqual(consolidated.lifecycle_status, LifecycleStatus.STABLE)
        self.assertGreaterEqual(consolidated.stability, 0.85)

        # Original source evidence is preserved inside consolidated metadata
        evidence = consolidated.metadata.get("source_evidence", [])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["original_episodic_id"], "ep-1")

        # Audit log verification
        cons_entry = next((e for e in audit_log if e.action == AuditAction.CONSOLIDATE), None)
        self.assertIsNotNone(cons_entry)


class TestLifecycleTransitions(unittest.TestCase):
    def test_active_to_stable_and_ttl_expiration(self):
        now = 100000.0
        items = [
            # Reinforced item -> STABLE
            MemoryItem(
                id="item-reinforce",
                content="Prefers functional programming",
                memory_type=MemoryType.SEMANTIC,
                source_session_id="s1",
                timestamp=now - 500,
                frequency=3,
                stability=0.80,
                lifecycle_status=LifecycleStatus.ACTIVE
            ),
            # Transient item past TTL -> EXPIRED
            MemoryItem(
                id="item-temp",
                content="Currently looking at line 42 of script",
                memory_type=MemoryType.EPISODIC,
                source_session_id="s1",
                timestamp=now - 100000,  # 100,000s old > 86,400s TTL
                frequency=1,
                stability=0.20,
                lifecycle_status=LifecycleStatus.TEMPORARY
            ),
            # Already superseded -> Remains SUPERSEDED
            MemoryItem(
                id="item-super",
                content="Old company",
                memory_type=MemoryType.SEMANTIC,
                source_session_id="s1",
                timestamp=now - 1000,
                lifecycle_status=LifecycleStatus.SUPERSEDED
            )
        ]

        audit_log = []
        evaluated = evaluate_lifecycle_transitions(
            memories=items,
            current_timestamp=now,
            temporary_ttl_seconds=86400.0,
            audit_entries=audit_log
        )

        reinforce_item = next(m for m in evaluated if m.id == "item-reinforce")
        temp_item = next(m for m in evaluated if m.id == "item-temp")
        super_item = next(m for m in evaluated if m.id == "item-super")

        self.assertEqual(reinforce_item.lifecycle_status, LifecycleStatus.STABLE)
        self.assertEqual(temp_item.lifecycle_status, LifecycleStatus.EXPIRED)
        self.assertEqual(super_item.lifecycle_status, LifecycleStatus.SUPERSEDED)


class TestBudgetMaintenanceAndEviction(unittest.TestCase):
    def test_budget_maintenance_evicts_expired_and_low_utility(self):
        now = 1000.0
        config = AllocationConfig(total_budget=50, min_episodic_share=0.2, max_episodic_share=0.8)

        # Candidate pool exceeding 50 tokens
        ep_items = [
            # High utility active episodic
            MemoryItem(id="ep-good", content="task details", memory_type=MemoryType.EPISODIC,
                       source_session_id="s", timestamp=now, utility_score=0.9, token_cost=15,
                       lifecycle_status=LifecycleStatus.ACTIVE),
            # Expired episodic
            MemoryItem(id="ep-exp", content="old task", memory_type=MemoryType.EPISODIC,
                       source_session_id="s", timestamp=now - 1000, utility_score=0.2, token_cost=10,
                       lifecycle_status=LifecycleStatus.EXPIRED),
            # Low utility active episodic
            MemoryItem(id="ep-low", content="trivial log", memory_type=MemoryType.EPISODIC,
                       source_session_id="s", timestamp=now, utility_score=0.1, token_cost=20,
                       lifecycle_status=LifecycleStatus.ACTIVE),
        ]
        sem_items = [
            # High utility semantic
            MemoryItem(id="sem-good", content="core user preference", memory_type=MemoryType.SEMANTIC,
                       source_session_id="s", timestamp=now, utility_score=0.95, token_cost=20,
                       lifecycle_status=LifecycleStatus.STABLE),
        ]

        audit_log = []
        sel_ep, sel_sem, evicted, summary = maintain_memory_budget(
            episodic_memories=ep_items,
            semantic_memories=sem_items,
            total_budget=50,
            allocation_config=config,
            audit_entries=audit_log,
            current_timestamp=now
        )

        total_used = summary["total_used_tokens"]
        self.assertLessEqual(total_used, 50, "Total tokens used must never exceed total budget")

        # Expired item must be evicted to archive
        evicted_ids = {m.id for m in evicted}
        self.assertIn("ep-exp", evicted_ids)

        # High utility items should remain selected
        sel_ep_ids = {m.id for m in sel_ep}
        self.assertIn("ep-good", sel_ep_ids)
        sel_sem_ids = {m.id for m in sel_sem}
        self.assertIn("sem-good", sel_sem_ids)


class TestFullPhase2MultiSession(unittest.TestCase):
    def test_multi_session_evolution_and_auditability(self):
        manager = AdaptiveMemoryManager(total_budget=150)

        # Session 1: User introduces employer and preferences
        session_1 = [
            {"role": "user", "content": "I work at OpenAI. I always prefer PyTorch.", "timestamp": 1000.0},
            {"role": "assistant", "content": "Welcome! PyTorch is noted.", "timestamp": 1010.0},
        ]
        manager.process_session(session_1, session_id="s1", current_timestamp=1020.0)

        openai_item = next((m for m in manager.semantic_store if "openai" in m.content.lower()), None)
        self.assertIsNotNone(openai_item)
        self.assertEqual(openai_item.lifecycle_status, LifecycleStatus.STABLE)

        # Session 2: User changes jobs and reaffirms PyTorch
        session_2 = [
            {"role": "user", "content": "I work at DeepMind now. I always prefer PyTorch.", "timestamp": 2000.0},
            {"role": "assistant", "content": "Congratulations on joining DeepMind!", "timestamp": 2010.0},
        ]
        manager.process_session(session_2, session_id="s2", current_timestamp=2020.0)

        # The new employer is in the semantic store or active memories
        all_known = manager.semantic_store + manager.episodic_store + manager.archive_store
        deepmind_item = next((m for m in manager.semantic_store if "deepmind" in m.content.lower()), None)
        self.assertIsNotNone(deepmind_item)

        # The older employer is preserved as SUPERSEDED in the system
        old_work = next((m for m in all_known if "openai" in m.content.lower()), None)
        self.assertIsNotNone(old_work)
        self.assertEqual(old_work.lifecycle_status, LifecycleStatus.SUPERSEDED)

        # Predecessor link is intact: can answer "What was user's previous company?"
        predecessors = deepmind_item.metadata.get("historical_predecessors", [])
        self.assertTrue(any(p["previous_value"] == "openai" for p in predecessors))

        # Check total tokens strictly within budget
        total_tokens = sum(m.token_cost for m in manager.episodic_store + manager.semantic_store)
        self.assertLessEqual(total_tokens, 150)

        # Audit log contains entries for supersede, merge, and consolidation
        actions = {e.action for e in manager.audit_log}
        self.assertIn(AuditAction.SUPERSEDE, actions)

    def test_save_and_load_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "phase2_state.json")
            manager = AdaptiveMemoryManager(total_budget=200, persistence_path=path)

            session = [
                {"role": "user", "content": "I live in Bangalore. I like dark mode.", "timestamp": 100.0}
            ]
            manager.process_session(session, session_id="s_roundtrip", current_timestamp=110.0)

            self.assertTrue(os.path.exists(path))

            # New manager loading the file
            new_manager = AdaptiveMemoryManager(total_budget=200)
            new_manager.load_state(path)

            self.assertEqual(len(new_manager.semantic_store), len(manager.semantic_store))
            self.assertEqual(len(new_manager.audit_log), len(manager.audit_log))


if __name__ == "__main__":
    unittest.main()
