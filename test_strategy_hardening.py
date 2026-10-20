"""
Unit and integration tests for hardened shared pipeline components:
- Identical candidate extraction across all 5 strategies
- Consolidation double-counting prevention
- FlatMemoryBaseline admission gate enforcement
- Assistant turn filtering policy
- Tokenizer consistency between storage accounting and prompt context
- Fixed and dynamic quota math under 500 tokens
- Persistent memory budget compliance under 500 tokens
"""

import unittest
from typing import Any, Dict, List

from candidate_generation import (
    AdmissionDecision,
    CandidateAdmissionGate,
    HeuristicCandidateExtractor,
)
from memory_core import (
    LifecycleStatus,
    MemoryItem,
    MemoryType,
    estimate_token_cost,
)
from memory_lifecycle import AdaptiveMemoryManager, consolidate_episodic_memories
from memory_strategies import (
    AblatedMemoryStrategy,
    AdaptiveHierarchicalStrategy,
    FixedHierarchicalStrategy,
    FlatMemoryBaseline,
)
from model_adapter import heuristic_bpe_token_count


class TestStrategyHardening(unittest.TestCase):
    def setUp(self):
        self.session_data: List[Dict[str, Any]] = [
            {"role": "user", "content": "My name is Alice and I work at DeepMind.", "timestamp": 1000.0},
            {"role": "assistant", "content": "Understood, dataclasses are great.", "timestamp": 1010.0},
            {"role": "user", "content": "I prefer using PyTorch for all my neural networks.", "timestamp": 1020.0},
            {"role": "assistant", "content": "Your dog's name is Luna.", "timestamp": 1030.0},
            {"role": "user", "content": "Can you debug this CUDA memory leak right now?", "timestamp": 1040.0},
        ]

    # 1. Assistant Turn Filtering Policy
    def test_assistant_turn_filtering_policy(self):
        extractor = HeuristicCandidateExtractor()
        candidates = extractor.extract_candidates(self.session_data, session_id="s_test", current_timestamp=1050.0)
        contents = [c.content for c in candidates]

        # Generic assistant advice/chit-chat must be filtered out
        self.assertFalse(
            any("dataclasses are great" in c.lower() for c in contents),
            "Generic conversational assistant advice must not become a memory candidate"
        )

        # Factual confirmation of user slot from assistant should be admitted
        self.assertTrue(
            any("luna" in c.lower() for c in contents),
            "Factual confirmation from assistant referencing user slots must be admitted"
        )

        # User facts must be admitted
        self.assertTrue(any("deepmind" in c.lower() for c in contents))
        self.assertTrue(any("pytorch" in c.lower() for c in contents))

    # 2. Identical Candidate Extraction Across All 5 Strategies
    def test_identical_candidate_extraction_across_strategies(self):
        gate = CandidateAdmissionGate()
        tokenizer = heuristic_bpe_token_count

        strategies = {
            "flat": FlatMemoryBaseline(total_budget=500, admission_gate=gate, tokenizer_fn=tokenizer),
            "fixed_70_30": FixedHierarchicalStrategy(total_budget=500, episodic_share=0.70, admission_gate=gate, tokenizer_fn=tokenizer),
            "fixed_50_50": FixedHierarchicalStrategy(total_budget=500, episodic_share=0.50, admission_gate=gate, tokenizer_fn=tokenizer),
            "fixed_30_70": FixedHierarchicalStrategy(total_budget=500, episodic_share=0.30, admission_gate=gate, tokenizer_fn=tokenizer),
            "adaptive": AdaptiveHierarchicalStrategy(total_budget=500, admission_gate=gate, tokenizer_fn=tokenizer),
        }

        # Candidate extraction from the shared extractor
        extractor = HeuristicCandidateExtractor(tokenizer_fn=tokenizer)
        expected_candidates = extractor.extract_candidates(self.session_data, "s1", 1050.0, tokenizer_fn=tokenizer)
        expected_keys = sorted(c.content for c in expected_candidates)

        for name, strat in strategies.items():
            if hasattr(strat, "candidate_extractor") and strat.candidate_extractor is not None:
                strat_cands = strat.candidate_extractor.extract_candidates(self.session_data, "s1", 1050.0, tokenizer_fn=tokenizer)
            else:
                extractor_default = HeuristicCandidateExtractor(tokenizer_fn=strat.tokenizer_fn)
                strat_cands = extractor_default.extract_candidates(self.session_data, "s1", 1050.0, tokenizer_fn=strat.tokenizer_fn)

            strat_keys = sorted(c.content for c in strat_cands)
            self.assertEqual(strat_keys, expected_keys, f"Strategy {name} extracted mismatched candidates")

    # 3. Flat Memory Admission Gate Enforcement
    def test_flat_memory_admission_gate_rejection(self):
        gate = CandidateAdmissionGate(min_importance_threshold=0.30, min_confidence_threshold=0.50)
        flat = FlatMemoryBaseline(total_budget=500, admission_gate=gate)

        filler_history = [
            {"role": "user", "content": "Hello! Good morning.", "timestamp": 100.0},
            {"role": "assistant", "content": "How can I help you today?", "timestamp": 105.0},
            {"role": "user", "content": "That's really cool. Thanks a lot!", "timestamp": 110.0},
        ]
        res = flat.process_session(filler_history, "s_filler", 120.0)
        # Low utility/filler items should be rejected or produce 0 stored items
        self.assertEqual(len(flat.memory_store), 0)
        self.assertEqual(flat.get_storage_usage(), 0)

    # 4. Consolidation Double-Counting Prevention
    def test_consolidation_double_counting_prevention(self):
        manager = AdaptiveMemoryManager(total_budget=500)

        # Ingest repeated episodic observation across multiple sessions
        sess1 = [{"role": "user", "content": "Today I debugged the CUDA memory issue on GPU node 1.", "timestamp": 1000.0}]
        sess2 = [{"role": "user", "content": "Today I debugged the CUDA memory issue on GPU node 1.", "timestamp": 2000.0}]

        manager.process_session(sess1, "s1", 1050.0)
        manager.process_session(sess2, "s2", 2050.0)

        # The observation should be consolidated into semantic store
        semantic_items = [m for m in manager.semantic_store if "cuda memory" in m.content.lower()]
        self.assertGreaterEqual(len(semantic_items), 1, "Should have consolidated reinforced episodic item into semantic store")

        # In active episodic store, the superseded episodic observation must NOT be present as active
        active_episodic = [m for m in manager.episodic_store if "cuda memory" in m.content.lower() and m.lifecycle_status == LifecycleStatus.ACTIVE]
        self.assertEqual(
            len(active_episodic), 0,
            "Consolidated episodic observation must be superseded and evicted from active episodic store to prevent double-counting"
        )

        # In archive store, the original observation is preserved for provenance
        archived_items = [m for m in manager.archive_store if "cuda memory" in m.content.lower()]
        self.assertGreaterEqual(len(archived_items), 1, "Consolidated observation must be preserved in archive")
        self.assertEqual(archived_items[0].lifecycle_status, LifecycleStatus.SUPERSEDED)

    # 5. Tokenizer Consistency Between Storage Accounting and Context Assembly
    def test_tokenizer_consistency(self):
        custom_bpe = heuristic_bpe_token_count
        strat = FlatMemoryBaseline(total_budget=500, tokenizer_fn=custom_bpe)

        history = [
            {"role": "user", "content": "I love developing scalable systems with PyTorch.", "timestamp": 1000.0}
        ]
        strat.process_session(history, "s1", 1010.0)

        for mem in strat.get_persisted_memories():
            expected_cost = custom_bpe(mem.content)
            self.assertEqual(
                mem.token_cost, expected_cost,
                f"Memory token_cost ({mem.token_cost}) must match custom tokenizer ({expected_cost})"
            )

    # 6. Fixed Quota Math under 500 tokens
    def test_fixed_quota_math_under_500_tokens(self):
        # 70:30
        f70 = FixedHierarchicalStrategy(total_budget=500, episodic_share=0.70)
        self.assertEqual(f70.episodic_budget, 350)
        self.assertEqual(f70.semantic_budget, 150)
        self.assertEqual(f70.episodic_budget + f70.semantic_budget, 500)

        # 50:50
        f50 = FixedHierarchicalStrategy(total_budget=500, episodic_share=0.50)
        self.assertEqual(f50.episodic_budget, 250)
        self.assertEqual(f50.semantic_budget, 250)
        self.assertEqual(f50.episodic_budget + f50.semantic_budget, 500)

        # 30:70
        f30 = FixedHierarchicalStrategy(total_budget=500, episodic_share=0.30)
        self.assertEqual(f30.episodic_budget, 150)
        self.assertEqual(f30.semantic_budget, 350)
        self.assertEqual(f30.episodic_budget + f30.semantic_budget, 500)

    # 7. Shared 500-Token Budget Enforcement Across All 5 Strategies
    def test_shared_500_token_budget_enforcement(self):
        budget = 500
        gate = CandidateAdmissionGate()
        tokenizer = heuristic_bpe_token_count

        strategies = [
            FlatMemoryBaseline(total_budget=budget, admission_gate=gate, tokenizer_fn=tokenizer),
            FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.70, admission_gate=gate, tokenizer_fn=tokenizer),
            FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.50, admission_gate=gate, tokenizer_fn=tokenizer),
            FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.30, admission_gate=gate, tokenizer_fn=tokenizer),
            AdaptiveHierarchicalStrategy(total_budget=budget, admission_gate=gate, tokenizer_fn=tokenizer),
            AblatedMemoryStrategy(total_budget=budget, disable_adaptive=True, admission_gate=gate, tokenizer_fn=tokenizer),
        ]

        # Multi-session load
        sessions = [
            [{"role": "user", "content": f"User fact {i}: I work at Organization_{i} and live in City_{i}."} for i in range(10)],
            [{"role": "user", "content": f"User preference {i}: I prefer Tool_{i} for Task_{i}."} for i in range(10)],
            [{"role": "user", "content": f"Episodic task {i}: Please debug issue_{i} in system_{i}."} for i in range(10)],
        ]

        for strat in strategies:
            for s_idx, sess in enumerate(sessions):
                strat.process_session(sess, f"session_{s_idx}", 1000.0 * (s_idx + 1))

            storage_used = strat.get_storage_usage()
            self.assertLessEqual(
                storage_used, budget,
                f"Strategy {strat.name} used {storage_used} tokens, exceeding budget {budget}"
            )

    # 8. Ablation Consolidation Toggle
    def test_ablation_consolidation_toggle(self):
        # With consolidation disabled
        strat_no_cons = AblatedMemoryStrategy(total_budget=500, disable_consolidation=True)
        # With standard adaptive
        strat_with_cons = AdaptiveHierarchicalStrategy(total_budget=500)

        sess1 = [{"role": "user", "content": "Today I debugged the CUDA memory issue on GPU node 1.", "timestamp": 1000.0}]
        sess2 = [{"role": "user", "content": "Today I debugged the CUDA memory issue on GPU node 1.", "timestamp": 2000.0}]

        strat_no_cons.process_session(sess1, "s1", 1010.0)
        strat_no_cons.process_session(sess2, "s2", 2010.0)

        strat_with_cons.process_session(sess1, "s1", 1010.0)
        strat_with_cons.process_session(sess2, "s2", 2010.0)

        # strat_with_cons should have consolidated CUDA memory into semantic store
        self.assertTrue(any("cuda memory" in m.content.lower() for m in strat_with_cons.manager.semantic_store))
        # strat_no_cons should have skipped consolidation into semantic store
        self.assertFalse(any("cuda memory" in m.content.lower() for m in strat_no_cons.manager.semantic_store))


if __name__ == "__main__":
    unittest.main()
