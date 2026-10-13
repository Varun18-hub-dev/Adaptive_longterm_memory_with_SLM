"""
Integration tests for Phase 5.1: Candidate Pipeline Integration Hardening.

Covers:
1. Gate acceptance and rejection through process_session().
2. Duplicate and conflicting candidates (soft-deduplication and conflict superseding).
3. Historical memory and provenance preservation.
4. Default behavior when no gate is provided (backward compatibility).
5. Hybrid extractor activation and fallback.
6. Persistent-memory budget compliance under gate filtering.
"""

import unittest
from typing import Any, Dict, List

from candidate_generation import (
    AdmissionDecision,
    CandidateAdmissionGate,
    HybridCandidateExtractor,
)
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
)
from memory_retrieval import retrieve_memories
from model_adapter import BaseLLMAdapter, FakeLLMAdapter


class FaultyLLMAdapter(BaseLLMAdapter):
    """Adapter that raises an exception or outputs malformed content to test fallback resilience."""
    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7, **kwargs: Any) -> str:
        raise RuntimeError("Simulated SLM inference error: CUDA out of memory.")

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 128, temperature: float = 0.7, **kwargs: Any) -> str:
        raise RuntimeError("Simulated SLM inference error: connection reset.")

    def tokenize(self, text: str) -> List[int]:
        return [1] * len(text.split())


class TestCandidatePipelineHardening(unittest.TestCase):

    def test_gate_acceptance_and_rejection_through_process_session(self):
        """
        1. Test gate acceptance and rejection through process_session():
           Verifies that rejected candidates (filler, low confidence) cannot enter active stores,
           while valid candidates are admitted, and rejections are properly audited.
        """
        gate = CandidateAdmissionGate(
            min_importance_threshold=0.70,
            min_confidence_threshold=0.50
        )
        manager = AdaptiveMemoryManager(
            total_budget=500,
            admission_gate=gate
        )

        conversation = [
            # High-confidence personal fact -> Admitted (importance 0.85 >= 0.70, conf 1.0 >= 0.50)
            {"role": "user", "content": "I work at Google as a compiler engineer.", "timestamp": 100.0},
            # Low-importance transient task -> Rejected (importance 0.60 < 0.70)
            {"role": "user", "content": "Can you help me debug this CUDA issue today?", "timestamp": 110.0},
            # Low-confidence uncertain assertion -> Rejected (conf 0.45 < 0.50, REJECT_LOW_CONFIDENCE)
            {"role": "user", "content": "Maybe I want to learn Julia but I am totally unsure.", "timestamp": 120.0},
        ]

        result = manager.process_session(conversation, session_id="sess_gate_test", current_timestamp=130.0)

        active_memories = manager.episodic_store + manager.semantic_store
        active_contents = [m.content.lower() for m in active_memories]

        # 1. Valid fact admitted
        self.assertTrue(any("compiler engineer" in c or "google" in c for c in active_contents))

        # 2. Low importance and low confidence rejected from active stores
        self.assertFalse(any("cuda" in c for c in active_contents))
        self.assertFalse(any("julia" in c for c in active_contents))

        # 3. Rejections tracked in manager.rejected_candidates and return summary
        self.assertEqual(len(manager.rejected_candidates), 2)
        self.assertEqual(result["rejected_count"], 2)

        rejected_decisions = [dec for _, dec, _ in manager.rejected_candidates]
        self.assertIn(AdmissionDecision.REJECT_FILLER, rejected_decisions)
        self.assertIn(AdmissionDecision.REJECT_LOW_CONFIDENCE, rejected_decisions)

        # 4. Audit log records REJECT_CANDIDATE action
        reject_audit_entries = [e for e in manager.audit_log if e.action == AuditAction.REJECT_CANDIDATE]
        self.assertGreaterEqual(len(reject_audit_entries), 2)

    def test_duplicate_and_conflicting_candidates(self):
        """
        2. Test duplicate and conflicting candidates:
           Verifies soft-deduplication merges repeated observations,
           while evolving facts supersede older facts with historical predecessor links.
        """
        gate = CandidateAdmissionGate()
        manager = AdaptiveMemoryManager(
            total_budget=500,
            admission_gate=gate
        )

        # Session 1: Initial observation
        s1 = [{"role": "user", "content": "I work at OpenAI on research.", "timestamp": 1000.0}]
        manager.process_session(s1, session_id="s1", current_timestamp=1010.0)
        self.assertEqual(len(manager.episodic_store) + len(manager.semantic_store), 1)

        # Session 2: Duplicate observation (same fact reinforced)
        s2 = [{"role": "user", "content": "I work at OpenAI on research.", "timestamp": 2000.0}]
        manager.process_session(s2, session_id="s2", current_timestamp=2010.0)

        # Still only 1 distinct active fact, but reinforced frequency and provenance
        all_active = manager.episodic_store + manager.semantic_store
        self.assertEqual(len(all_active), 1)
        reinforced = all_active[0]
        self.assertEqual(reinforced.frequency, 2)
        self.assertGreaterEqual(len(reinforced.metadata.get("provenance", [])), 2)

        # Session 3: Conflict / Fact evolution
        s3 = [{"role": "user", "content": "I work at DeepMind now.", "timestamp": 3000.0}]
        manager.process_session(s3, session_id="s3", current_timestamp=3010.0)

        all_active_now = manager.episodic_store + manager.semantic_store
        deepmind_mem = next((m for m in all_active_now if "deepmind" in m.content.lower()), None)
        self.assertIsNotNone(deepmind_mem)

        # Predecessor chain recorded
        predecessors = deepmind_mem.metadata.get("historical_predecessors", [])
        self.assertGreaterEqual(len(predecessors), 1)
        self.assertTrue(any("openai" in p.get("original_content", "").lower() for p in predecessors))

        # Check supersede audit action exists
        supersede_logs = [e for e in manager.audit_log if e.action == AuditAction.SUPERSEDE]
        self.assertGreaterEqual(len(supersede_logs), 1)

    def test_historical_memory_and_provenance_preservation(self):
        """
        3. Test historical memory and provenance preservation:
           Verifies that past historical facts are tagged and remain accessible for queries.
        """
        gate = CandidateAdmissionGate()
        manager = AdaptiveMemoryManager(
            total_budget=500,
            admission_gate=gate
        )

        # Session 1: Explicit historical fact with temporal boundary
        s1 = [
            {"role": "user", "content": "From 2018 to 2022, I lived in Seattle.", "timestamp": 100.0}
        ]
        manager.process_session(s1, session_id="s_hist", current_timestamp=110.0)

        # Session 2: Current location
        s2 = [
            {"role": "user", "content": "I live in San Francisco now.", "timestamp": 200.0}
        ]
        manager.process_session(s2, session_id="s_curr", current_timestamp=210.0)

        all_memories = manager.episodic_store + manager.semantic_store
        seattle_mem = next((m for m in all_memories if "seattle" in m.content.lower()), None)
        sf_mem = next((m for m in all_memories if "san francisco" in m.content.lower()), None)

        self.assertIsNotNone(seattle_mem)
        self.assertIsNotNone(sf_mem)
        self.assertTrue(seattle_mem.metadata.get("is_historical_fact", False))

        # Historical query should retrieve Seattle
        hist_retrieved = retrieve_memories(
            query="Where did I live from 2018 to 2022?",
            episodic_store=manager.episodic_store,
            semantic_store=manager.semantic_store,
            archive_store=manager.archive_store,
            top_k=2
        )
        self.assertTrue(any("seattle" in m.content.lower() for m, _ in hist_retrieved))

        # Current query should prioritize San Francisco
        curr_retrieved = retrieve_memories(
            query="Where do I live now?",
            episodic_store=manager.episodic_store,
            semantic_store=manager.semantic_store,
            archive_store=manager.archive_store,
            top_k=2
        )
        self.assertTrue(any("san francisco" in m.content.lower() for m, _ in curr_retrieved))

    def test_default_behavior_when_no_gate_is_provided(self):
        """
        4. Test default behavior when no gate is provided:
           Verifies full backward compatibility when admission_gate is omitted or None.
        """
        manager = AdaptiveMemoryManager(total_budget=300)
        self.assertIsNone(manager.admission_gate)
        self.assertEqual(len(manager.rejected_candidates), 0)

        conversation = [
            {"role": "user", "content": "I work at DeepMind.", "timestamp": 100.0},
            {"role": "user", "content": "I prefer using PyTorch.", "timestamp": 110.0},
        ]
        res = manager.process_session(conversation, session_id="s_default", current_timestamp=120.0)

        self.assertEqual(res["rejected_count"], 0)
        self.assertEqual(len(manager.rejected_candidates), 0)
        self.assertGreaterEqual(len(manager.episodic_store) + len(manager.semantic_store), 2)

    def test_hybrid_extractor_activation_and_fallback(self):
        """
        5. Test hybrid extractor activation and fallback:
           Verifies that HybridCandidateExtractor switches between fast heuristics
           and model extraction, and falls back gracefully when the model adapter fails.
        """
        # A. Hybrid with FakeLLMAdapter
        fake_adapter = FakeLLMAdapter()
        hybrid_with_fake = HybridCandidateExtractor(
            llm_adapter=fake_adapter,
            trigger_slm_on_multi_clause=True
        )

        conv_complex = [
            {
                "role": "user",
                "content": (
                    "I work at DeepMind on reinforcement learning, live in London, "
                    "and prefer using Python with PyTorch for all neural network modeling experiments."
                ),
                "timestamp": 100.0
            }
        ]
        # Should execute without errors
        candidates = hybrid_with_fake.extract_candidates(conv_complex, session_id="s_hybrid")
        self.assertGreaterEqual(len(candidates), 1)

        # B. Fallback when model fails (FaultyLLMAdapter)
        faulty_adapter = FaultyLLMAdapter()
        hybrid_faulty = HybridCandidateExtractor(
            llm_adapter=faulty_adapter,
            trigger_slm_on_multi_clause=True
        )
        # Should gracefully catch exception and return heuristic candidates
        fallback_candidates = hybrid_faulty.extract_candidates(conv_complex, session_id="s_faulty")
        self.assertGreaterEqual(len(fallback_candidates), 1)
        self.assertTrue(any("deepmind" in c.content.lower() for c in fallback_candidates))

        # C. Fallback when no LLM adapter is provided
        hybrid_no_llm = HybridCandidateExtractor(llm_adapter=None, trigger_slm_on_multi_clause=True)
        no_llm_candidates = hybrid_no_llm.extract_candidates(conv_complex, session_id="s_no_llm")
        self.assertGreaterEqual(len(no_llm_candidates), 1)

    def test_persistent_memory_budget_compliance(self):
        """
        6. Test persistent-memory budget compliance:
           Verifies that the total token cost of active stores strictly respects the budget,
           evicting low-utility items to the archive store with proper audit logs.
        """
        gate = CandidateAdmissionGate()
        budget_limit = 40  # Tight budget
        manager = AdaptiveMemoryManager(
            total_budget=budget_limit,
            admission_gate=gate
        )

        # Feed multiple distinct sessions with substantial content
        for idx in range(4):
            conversation = [
                {
                    "role": "user",
                    "content": f"Task observation number {idx}: I ran an experiment on CUDA node {idx} with batch size {16 * (idx + 1)}.",
                    "timestamp": 1000.0 + idx * 100.0
                }
            ]
            manager.process_session(conversation, session_id=f"sess_budget_{idx}", current_timestamp=1000.0 + idx * 100.0 + 10.0)

        active_tokens = sum(m.token_cost for m in manager.episodic_store + manager.semantic_store)
        self.assertLessEqual(active_tokens, budget_limit)
        self.assertGreater(len(manager.archive_store), 0)

        # Verify eviction audit logs exist
        evict_logs = [e for e in manager.audit_log if e.action == AuditAction.EVICT]
        self.assertGreater(len(evict_logs), 0)


if __name__ == "__main__":
    unittest.main()
