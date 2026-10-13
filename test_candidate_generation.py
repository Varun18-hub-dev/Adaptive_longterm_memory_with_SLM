"""
Unit and Integration Tests for Phase 5:
Modular Candidate Generation Pipeline & Hardened Admission Gate.
"""

import json
import unittest

from candidate_generation import (
    AdmissionDecision,
    CandidateAdmissionGate,
    HeuristicCandidateExtractor,
    HybridCandidateExtractor,
    SLMCandidateExtractor,
    decompose_multi_fact_sentence,
    detect_historical_intent,
    detect_negation,
    detect_uncertainty,
    is_filler_sentence,
)
from memory_core import LifecycleStatus, MemoryItem, MemoryType
from model_adapter import FakeLLMAdapter


class TestCandidateGenerationCases(unittest.TestCase):
    def setUp(self):
        self.extractor = HeuristicCandidateExtractor()
        self.gate = CandidateAdmissionGate()

    # 1. Stable personal facts
    def test_stable_personal_facts(self):
        history = [{"role": "user", "content": "I live in London and I am 32 years old.", "timestamp": 100.0}]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=100.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertTrue(any(c.memory_type == MemoryType.SEMANTIC for c in candidates))
        for c in candidates:
            self.assertGreaterEqual(c.stability, 0.70)
            self.assertEqual(c.lifecycle_status, LifecycleStatus.STABLE)

    # 2. User preferences
    def test_user_preferences(self):
        history = [{"role": "user", "content": "I prefer dark mode and I always use Python for scripting.", "timestamp": 100.0}]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=100.0)
        self.assertGreaterEqual(len(candidates), 1)
        pref_mem = next((c for c in candidates if "dark mode" in c.content.lower()), None)
        self.assertIsNotNone(pref_mem)
        self.assertEqual(pref_mem.memory_type, MemoryType.SEMANTIC)

    # 3. Temporary tasks
    def test_temporary_tasks(self):
        history = [{"role": "user", "content": "Please help me debug this CUDA out of memory error right now.", "timestamp": 100.0}]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=100.0)
        self.assertEqual(len(candidates), 1)
        task = candidates[0]
        self.assertEqual(task.memory_type, MemoryType.EPISODIC)
        self.assertLessEqual(task.stability, 0.40)
        self.assertEqual(task.lifecycle_status, LifecycleStatus.ACTIVE)

    # 4. Repeated facts
    def test_repeated_facts(self):
        history = [
            {"role": "user", "content": "I work at DeepMind.", "timestamp": 100.0},
            {"role": "assistant", "content": "Understood.", "timestamp": 110.0},
            {"role": "user", "content": "I work at DeepMind.", "timestamp": 200.0},
        ]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=200.0)
        self.assertEqual(len(candidates), 1, "Repeated exact fact should be deduplicated")
        self.assertEqual(candidates[0].frequency, 2)
        self.assertEqual(candidates[0].timestamp, 200.0)
        self.assertEqual(len(candidates[0].metadata["provenance"]), 2)

    # 5. Updated information
    def test_updated_information(self):
        history = [
            {"role": "user", "content": "I work at OpenAI.", "timestamp": 100.0},
            {"role": "user", "content": "I work at DeepMind now.", "timestamp": 200.0},
        ]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=200.0)
        self.assertEqual(len(candidates), 2)
        
        # Test admission gate resolves conflict and supersedes older fact
        res = self.gate.evaluate_candidates(candidates, existing_memories=[], current_timestamp=200.0)
        admitted = res.admitted_candidates
        self.assertEqual(len(admitted), 2)
        
        old_mem = next(c for c in admitted if "openai" in c.content.lower())
        new_mem = next(c for c in admitted if "deepmind" in c.content.lower())
        self.assertEqual(old_mem.lifecycle_status, LifecycleStatus.SUPERSEDED)
        self.assertEqual(new_mem.lifecycle_status, LifecycleStatus.ACTIVE)
        
        predecessors = new_mem.metadata.get("historical_predecessors", [])
        self.assertTrue(any(p["previous_value"] == "openai" for p in predecessors))

    # 6. Explicit contradictions
    def test_explicit_contradictions(self):
        history = [
            {"role": "user", "content": "I prefer PyTorch for deep learning.", "timestamp": 100.0},
            {"role": "user", "content": "I prefer JAX for deep learning.", "timestamp": 200.0},
        ]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=200.0)
        self.assertEqual(len(candidates), 2)

        res = self.gate.evaluate_candidates(candidates, existing_memories=[], current_timestamp=200.0)
        admitted = res.admitted_candidates
        old_fw = next(c for c in admitted if "pytorch" in c.content.lower())
        new_fw = next(c for c in admitted if "jax" in c.content.lower())
        self.assertEqual(old_fw.lifecycle_status, LifecycleStatus.SUPERSEDED)
        self.assertTrue(any(p["previous_value"] == "pytorch" for p in new_fw.metadata.get("historical_predecessors", [])))

    # 7. Historical facts distinguishable from current facts
    def test_historical_facts(self):
        history = [
            {"role": "user", "content": "I worked at Meta from 2020 to 2022.", "timestamp": 100.0},
            {"role": "user", "content": "I joined Anthropic in 2023.", "timestamp": 200.0},
        ]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=200.0)
        self.assertEqual(len(candidates), 2)
        meta_mem = next(c for c in candidates if "meta" in c.content.lower())
        self.assertTrue(meta_mem.metadata.get("is_historical_fact", False))

    # 8. Irrelevant conversation, filler, and distractors
    def test_filler_and_distractor_suppression(self):
        history = [
            {"role": "user", "content": "Hello! Good morning.", "timestamp": 100.0},
            {"role": "assistant", "content": "How can I help you today?", "timestamp": 105.0},
            {"role": "user", "content": "That's really cool. Thanks a lot!", "timestamp": 110.0},
            {"role": "assistant", "content": "You're welcome! Anytime.", "timestamp": 115.0},
            {"role": "user", "content": "Sounds awesome. No problem.", "timestamp": 120.0},
        ]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=125.0)
        self.assertEqual(len(candidates), 0, "All pure pleasantries and reaction fillers should be suppressed")

    # 9. Multiple facts in one sentence
    def test_multi_fact_clause_decomposition(self):
        history = [
            {"role": "user", "content": "I live in Seattle, work at Amazon, and prefer using vim.", "timestamp": 100.0}
        ]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=100.0)
        self.assertEqual(len(candidates), 3, "Multi-fact sentence should be decomposed into 3 atomic propositions")
        contents = [c.content for c in candidates]
        self.assertTrue(any("seattle" in c.lower() for c in contents))
        self.assertTrue(any("amazon" in c.lower() for c in contents))
        self.assertTrue(any("vim" in c.lower() for c in contents))

    # 10. Negation and uncertainty
    def test_negation_and_uncertainty(self):
        history = [
            {"role": "user", "content": "I am not sure if I like Rust yet, maybe I'll try it later.", "timestamp": 100.0},
            {"role": "user", "content": "I never drink coffee in the afternoon.", "timestamp": 200.0},
        ]
        candidates = self.extractor.extract_candidates(history, session_id="s1", current_timestamp=200.0)
        self.assertEqual(len(candidates), 2)

        unc_mem = next(c for c in candidates if "rust" in c.content.lower())
        self.assertTrue(unc_mem.metadata.get("uncertainty_detected", False))
        self.assertLessEqual(unc_mem.consistency_confidence, 0.50)
        self.assertNotEqual(unc_mem.lifecycle_status, LifecycleStatus.STABLE)

        neg_mem = next(c for c in candidates if "coffee" in c.content.lower())
        self.assertTrue(neg_mem.metadata.get("is_negation", False))


class TestSLMAndHybridExtraction(unittest.TestCase):
    def test_slm_candidate_extractor_with_fake_llm(self):
        mock_response = json.dumps([
            {
                "content": "User lives in Zurich",
                "memory_type": "semantic",
                "importance": 0.85,
                "temporal_stability": 0.90,
                "confidence": 1.0,
                "is_uncertain": False,
                "is_historical": False
            }
        ])
        adapter = FakeLLMAdapter(canned_response=f"```json\n{mock_response}\n```")
        slm_extractor = SLMCandidateExtractor(llm_adapter=adapter)

        history = [{"role": "user", "content": "I live in Zurich."}]
        candidates = slm_extractor.extract_candidates(history, session_id="s_slm")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].content, "User lives in Zurich")
        self.assertEqual(candidates[0].memory_type, MemoryType.SEMANTIC)
        self.assertEqual(candidates[0].importance, 0.85)

    def test_slm_candidate_extractor_fallback_on_invalid_json(self):
        # When model returns non-JSON text, gracefully falls back to heuristics
        adapter = FakeLLMAdapter(canned_response="Sorry, I cannot produce JSON right now.")
        slm_extractor = SLMCandidateExtractor(llm_adapter=adapter)

        history = [{"role": "user", "content": "I prefer writing code in Go."}]
        candidates = slm_extractor.extract_candidates(history, session_id="s_fallback")
        self.assertEqual(len(candidates), 1)
        self.assertIn("go", candidates[0].content.lower())
        self.assertEqual(candidates[0].memory_type, MemoryType.SEMANTIC)

    def test_hybrid_candidate_extractor(self):
        hybrid = HybridCandidateExtractor(trigger_slm_on_multi_clause=False)
        history = [
            {"role": "user", "content": "I work at DeepMind. Can you check this git log?"}
        ]
        candidates = hybrid.extract_candidates(history, session_id="s_hybrid")
        self.assertEqual(len(candidates), 2)
        contents = [c.content.lower() for c in candidates]
        self.assertTrue(any("deepmind" in c for c in contents))
        self.assertTrue(any("git log" in c for c in contents))


class TestCandidateAdmissionGate(unittest.TestCase):
    def setUp(self):
        self.gate = CandidateAdmissionGate(min_importance_threshold=0.30, min_confidence_threshold=0.40)

    def test_gate_rejects_filler_and_low_confidence(self):
        candidates = [
            MemoryItem(
                id="c1", content="That's really cool.", memory_type=MemoryType.SEMANTIC,
                source_session_id="s1", timestamp=100.0, importance=0.8, consistency_confidence=1.0
            ),
            MemoryItem(
                id="c2", content="Maybe I like Rust but I am totally unsure.", memory_type=MemoryType.SEMANTIC,
                source_session_id="s1", timestamp=100.0, importance=0.5, consistency_confidence=0.20
            ),
            MemoryItem(
                id="c3", content="I work at Google.", memory_type=MemoryType.SEMANTIC,
                source_session_id="s1", timestamp=100.0, importance=0.85, consistency_confidence=1.0
            )
        ]
        res = self.gate.evaluate_candidates(candidates, existing_memories=[])
        self.assertEqual(len(res.admitted_candidates), 1)
        self.assertEqual(res.admitted_candidates[0].content, "I work at Google.")
        
        rejections = res.rejected_candidates
        self.assertEqual(len(rejections), 2)
        self.assertEqual(rejections[0][1], AdmissionDecision.REJECT_FILLER)
        self.assertEqual(rejections[1][1], AdmissionDecision.REJECT_LOW_CONFIDENCE)


if __name__ == "__main__":
    unittest.main()
