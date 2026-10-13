"""
Unit and Integration Tests for Phase 6:
Memory Execution Trace Backend & Observability

Tests cover:
1. Trace IDs are unique and events are correctly ordered.
2. Actual candidate extraction results appear in the trace.
3. Scoring evidence matches the scores used by the pipeline.
4. Admission decisions match actual gate outcomes.
5. Deduplication and conflict events reference the correct memories.
6. Allocation and token-budget values match actual execution.
7. Retrieval evidence matches memories in the model context.
8. Hybrid SLM failures and heuristic fallbacks are recorded honestly.
9. Failed and skipped stages have correct statuses.
10. Tracing does not alter memory results (traced vs untraced equivalence).
11. Trace serialization round-trips correctly.
12. API requests, missing trace IDs, and invalid inputs handled correctly.
13. Trace storage does not expose another session's private information.
"""

import json
import os
import unittest
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from api_server import app
from candidate_generation import (
    AdmissionDecision,
    CandidateAdmissionGate,
    HybridCandidateExtractor,
)
from execution_trace import (
    EventStatus,
    ExecutionTrace,
    ExecutionTracer,
    OverallStatus,
    TraceEvent,
    TraceStage,
    TraceStore,
)
from memory_core import MemoryItem, MemoryType, ScoringWeights
from memory_lifecycle import AdaptiveMemoryManager
from model_adapter import BaseLLMAdapter, FakeLLMAdapter
from traced_pipeline import run_traced_pipeline


class BrokenLLMAdapter(BaseLLMAdapter):
    """Adapter designed to fail to verify trace error and fallback capture."""
    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7, **kwargs: Any) -> str:
        raise RuntimeError("Synthetic model generation failure.")

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 128, temperature: float = 0.7, **kwargs: Any) -> str:
        raise RuntimeError("Synthetic model generation failure.")

    def tokenize(self, text: str) -> List[int]:
        return [1] * len(text.split())


class TestMemoryExecutionTraceBackend(unittest.TestCase):
    def setUp(self):
        self.trace_store = TraceStore(":memory:")
        self.api_client = TestClient(app)

    # 1. Trace IDs are unique and events are correctly ordered
    def test_trace_id_uniqueness_and_event_ordering(self):
        tracer1 = ExecutionTracer(session_id="s1", prompt="Prompt 1")
        tracer2 = ExecutionTracer(session_id="s1", prompt="Prompt 2")
        self.assertNotEqual(tracer1.trace_id, tracer2.trace_id)

        # Run traced pipeline
        res = run_traced_pipeline(
            prompt="I work at DeepMind.",
            session_id="sess_order",
            trace_store=self.trace_store
        )
        trace: ExecutionTrace = res["trace"]

        self.assertGreaterEqual(len(trace.events), 10)
        # Check sequence numbers are strictly 1, 2, 3...
        seq_nums = [e.sequence_number for e in trace.events]
        self.assertEqual(seq_nums, list(range(1, len(trace.events) + 1)))

    # 2. Actual candidate extraction results appear in the trace
    def test_candidate_extraction_in_trace(self):
        prompt = "I live in Paris and work at HuggingFace."
        res = run_traced_pipeline(prompt=prompt, session_id="sess_extract", trace_store=self.trace_store)
        trace = res["trace"]

        extract_evt = next((e for e in trace.events if e.stage == TraceStage.CANDIDATE_EXTRACTION), None)
        self.assertIsNotNone(extract_evt)
        self.assertEqual(extract_evt.status, EventStatus.COMPLETED)
        self.assertGreaterEqual(extract_evt.outputs["candidate_count"], 1)

        candidates = extract_evt.outputs["candidates"]
        contents = [c["content"].lower() for c in candidates]
        self.assertTrue(any("paris" in c or "huggingface" in c for c in contents))

    # 3. Scoring evidence matches the scores used by the pipeline
    def test_scoring_evidence_matches_pipeline_scores(self):
        prompt = "I prefer using PyTorch for all deep learning models."
        res = run_traced_pipeline(prompt=prompt, session_id="sess_score", trace_store=self.trace_store)
        trace = res["trace"]
        manager = res["manager"]

        score_evt = next((e for e in trace.events if e.stage == TraceStage.UTILITY_SCORING), None)
        self.assertIsNotNone(score_evt)

        scored_cands = score_evt.outputs["scored_candidates"]
        self.assertGreaterEqual(len(scored_cands), 1)

        # Confirm scores in trace match the scores in the manager's store
        active_mems = {m.id: m for m in manager.episodic_store + manager.semantic_store}
        for sc in scored_cands:
            cand_id = sc["id"]
            if cand_id in active_mems:
                mem = active_mems[cand_id]
                self.assertAlmostEqual(sc["utility_score"], round(mem.utility_score, 4), places=3)
                self.assertAlmostEqual(sc["importance"], round(mem.importance, 4), places=3)
                self.assertAlmostEqual(sc["confidence"], round(mem.consistency_confidence, 4), places=3)

    # 4. Admission decisions match actual gate outcomes
    def test_admission_decisions_match_gate_outcomes(self):
        gate = CandidateAdmissionGate(min_importance_threshold=0.70, min_confidence_threshold=0.50)
        manager = AdaptiveMemoryManager(total_budget=300, admission_gate=gate)

        # Prompt with high-confidence fact
        res1 = run_traced_pipeline(
            prompt="I work at DeepMind as a staff research engineer.",
            session_id="sess_admit",
            manager=manager,
            trace_store=self.trace_store
        )
        gate_evt1 = next(e for e in res1["trace"].events if e.stage == TraceStage.ADMISSION_GATE)
        self.assertEqual(gate_evt1.outputs["admitted_count"], 1)
        self.assertEqual(gate_evt1.outputs["rejected_count"], 0)

        # Prompt with low-confidence uncertain assertion
        res2 = run_traced_pipeline(
            prompt="Maybe I want to learn Julia but I am totally unsure.",
            session_id="sess_admit",
            manager=manager,
            trace_store=self.trace_store
        )
        gate_evt2 = next(e for e in res2["trace"].events if e.stage == TraceStage.ADMISSION_GATE)
        self.assertEqual(gate_evt2.outputs["rejected_count"], 1)
        rejection = gate_evt2.outputs["rejections"][0]
        self.assertEqual(rejection["decision"], AdmissionDecision.REJECT_LOW_CONFIDENCE.value)

    # 5. Deduplication and conflict events reference the correct memories
    def test_deduplication_and_conflict_events(self):
        manager = AdaptiveMemoryManager(total_budget=500)

        # Turn 1: Initial fact
        run_traced_pipeline(prompt="I work at OpenAI.", session_id="sess_conflict", manager=manager, trace_store=self.trace_store)
        initial_mem_id = manager.semantic_store[0].id if manager.semantic_store else manager.episodic_store[0].id

        # Turn 2: Conflicting fact (evolution)
        res_evolve = run_traced_pipeline(
            prompt="I work at DeepMind now.",
            session_id="sess_conflict",
            manager=manager,
            trace_store=self.trace_store
        )
        conflict_evt = next(e for e in res_evolve["trace"].events if e.stage == TraceStage.CONFLICT_RESOLUTION)
        self.assertEqual(conflict_evt.decision, "conflicts_resolved")
        self.assertGreaterEqual(conflict_evt.outputs["conflicts_detected"], 1)
        self.assertIn(initial_mem_id, res_evolve["trace"].correlations["superseded_memory_ids"])

    # 6. Allocation and token-budget values match actual execution
    def test_allocation_and_budget_values_match_execution(self):
        budget_limit = 60
        manager = AdaptiveMemoryManager(total_budget=budget_limit)

        res = run_traced_pipeline(
            prompt="I live in London and work at DeepMind and code in Python.",
            session_id="sess_alloc",
            manager=manager,
            storage_budget=budget_limit,
            trace_store=self.trace_store
        )
        trace = res["trace"]
        alloc_evt = next(e for e in trace.events if e.stage == TraceStage.ADAPTIVE_ALLOCATION)
        self.assertEqual(alloc_evt.outputs["allocated_budget"]["total"], budget_limit)

        active_tokens = sum(m.token_cost for m in manager.episodic_store + manager.semantic_store)
        self.assertEqual(alloc_evt.outputs["tokens_used"]["total"], active_tokens)
        self.assertLessEqual(active_tokens, budget_limit)

    # 7. Retrieval evidence matches memories in the model context
    def test_retrieval_evidence_matches_context(self):
        manager = AdaptiveMemoryManager(total_budget=300)
        run_traced_pipeline(prompt="I work at DeepMind.", session_id="sess_ret", manager=manager, trace_store=self.trace_store)

        # Query
        res = run_traced_pipeline(
            prompt="Where do I work now?",
            session_id="sess_ret",
            manager=manager,
            trace_store=self.trace_store
        )
        trace = res["trace"]

        ret_evt = next(e for e in trace.events if e.stage == TraceStage.MEMORY_RETRIEVAL)
        ctx_evt = next(e for e in trace.events if e.stage == TraceStage.CONTEXT_ASSEMBLY)

        self.assertGreaterEqual(ret_evt.outputs["retrieved_count"], 1)
        selected_ids = ctx_evt.outputs["selected_memory_ids"]
        self.assertGreaterEqual(len(selected_ids), 1)

        # All selected IDs should be among retrieved IDs
        retrieved_ids = [m["id"] for m in ret_evt.outputs["retrieved_memories"]]
        for sid in selected_ids:
            self.assertIn(sid, retrieved_ids)

    # 8. Hybrid SLM failures and heuristic fallbacks are recorded honestly
    def test_hybrid_slm_failure_and_fallback_recorded_honestly(self):
        broken_llm = BrokenLLMAdapter()
        hybrid_extractor = HybridCandidateExtractor(llm_adapter=broken_llm, trigger_slm_on_multi_clause=True)

        prompt = (
            "I live in Tokyo, work at Sony on computer vision and neural modeling, "
            "and always prefer coding clean Python scripts using dataclasses without boilerplate."
        )
        res = run_traced_pipeline(
            prompt=prompt,
            session_id="sess_fallback",
            candidate_extractor=hybrid_extractor,
            trace_store=self.trace_store
        )
        trace = res["trace"]
        extract_evt = next(e for e in trace.events if e.stage == TraceStage.CANDIDATE_EXTRACTION)

        # Fallback must be recorded
        self.assertTrue(extract_evt.outputs["fallback_triggered"])
        self.assertGreaterEqual(extract_evt.outputs["candidate_count"], 1)

    # 9. Failed and skipped stages have correct statuses
    def test_failed_and_skipped_stages_statuses(self):
        # Skipped persistence test
        res = run_traced_pipeline(prompt="Hello there.", session_id="sess_skip", trace_store=self.trace_store)
        trace = res["trace"]
        persist_evt = next(e for e in trace.events if e.stage == TraceStage.PERSISTENCE)
        self.assertEqual(persist_evt.status, EventStatus.SKIPPED)

        # Failure handling test
        broken_llm = BrokenLLMAdapter()
        with self.assertRaises(Exception):
            run_traced_pipeline(
                prompt="Tell me about my memory.",
                session_id="sess_fail",
                llm_adapter=broken_llm,
                trace_store=self.trace_store
            )

    # 10. Tracing does not alter memory results (traced vs untraced equivalence)
    def test_tracing_does_not_alter_memory_results(self):
        prompt = "I live in Berlin and always drink espresso."
        hist = [{"role": "user", "content": prompt, "timestamp": 100.0}]

        # Run 1: Untraced direct execution
        mgr_untraced = AdaptiveMemoryManager(total_budget=200)
        mgr_untraced.process_session(hist, session_id="s_eq", current_timestamp=100.0)

        # Run 2: Traced execution
        mgr_traced = AdaptiveMemoryManager(total_budget=200)
        res_traced = run_traced_pipeline(
            prompt=prompt,
            session_id="s_eq",
            manager=mgr_traced,
            storage_budget=200,
            current_timestamp=100.0
        )

        # Compare memory stores
        untraced_items = [m.content for m in mgr_untraced.episodic_store + mgr_untraced.semantic_store]
        traced_items = [m.content for m in mgr_traced.episodic_store + mgr_traced.semantic_store]
        self.assertEqual(sorted(untraced_items), sorted(traced_items))

        # Compare token costs
        untraced_tokens = sum(m.token_cost for m in mgr_untraced.episodic_store + mgr_untraced.semantic_store)
        traced_tokens = sum(m.token_cost for m in mgr_traced.episodic_store + mgr_traced.semantic_store)
        self.assertEqual(untraced_tokens, traced_tokens)

    # 11. Trace serialization round-trips correctly
    def test_trace_serialization_roundtrip(self):
        res = run_traced_pipeline(prompt="I prefer light mode.", session_id="sess_ser", trace_store=self.trace_store)
        original_trace = res["trace"]

        d = original_trace.to_dict()
        json_str = json.dumps(d)
        reloaded_dict = json.loads(json_str)

        roundtrip_trace = ExecutionTrace.from_dict(reloaded_dict)
        self.assertEqual(original_trace.trace_id, roundtrip_trace.trace_id)
        self.assertEqual(original_trace.session_id, roundtrip_trace.session_id)
        self.assertEqual(len(original_trace.events), len(roundtrip_trace.events))
        self.assertEqual(original_trace.status, roundtrip_trace.status)

    # 12. API requests, missing trace IDs (404), and invalid inputs handled correctly
    def test_api_chat_and_trace_endpoints(self):
        # Empty prompt -> 400
        bad_resp = self.api_client.post("/api/chat", json={"prompt": "  "})
        self.assertEqual(bad_resp.status_code, 400)

        # Valid chat prompt -> 200
        chat_resp = self.api_client.post(
            "/api/chat",
            json={
                "prompt": "I work at DeepMind.",
                "session_id": "api_test_sess",
                "use_fake_llm": True
            }
        )
        self.assertEqual(chat_resp.status_code, 200)
        data = chat_resp.json()
        trace_id = data["trace_id"]
        self.assertTrue(trace_id.startswith("trc_"))

        # Retrieve trace -> 200
        t_resp = self.api_client.get(f"/api/traces/{trace_id}")
        self.assertEqual(t_resp.status_code, 200)
        self.assertEqual(t_resp.json()["trace_id"], trace_id)

        # Retrieve events -> 200
        evts_resp = self.api_client.get(f"/api/traces/{trace_id}/events")
        self.assertEqual(evts_resp.status_code, 200)
        self.assertGreaterEqual(evts_resp.json()["event_count"], 10)

        # Unknown trace ID -> 404
        missing_resp = self.api_client.get("/api/traces/trc_nonexistent_999")
        self.assertEqual(missing_resp.status_code, 404)

    # 13. Trace storage does not expose another session's private information
    def test_multi_session_isolation(self):
        # Session A
        self.api_client.post("/api/chat", json={"prompt": "I live in Berlin.", "session_id": "session_A", "use_fake_llm": True})
        # Session B
        self.api_client.post("/api/chat", json={"prompt": "I live in Tokyo.", "session_id": "session_B", "use_fake_llm": True})

        # Inspect Session A memories
        mem_a = self.api_client.get("/api/memories?session_id=session_A").json()
        all_content_a = [m["content"].lower() for m in mem_a["episodic_store"] + mem_a["semantic_store"]]
        self.assertTrue(any("berlin" in c for c in all_content_a))
        self.assertFalse(any("tokyo" in c for c in all_content_a))

        # Inspect Session B memories
        mem_b = self.api_client.get("/api/memories?session_id=session_B").json()
        all_content_b = [m["content"].lower() for m in mem_b["episodic_store"] + mem_b["semantic_store"]]
        self.assertTrue(any("tokyo" in c for c in all_content_b))
        self.assertFalse(any("berlin" in c for c in all_content_b))


if __name__ == "__main__":
    unittest.main()
