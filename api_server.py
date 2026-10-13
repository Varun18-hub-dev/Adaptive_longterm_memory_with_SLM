"""
Adaptive Long-Term Memory for Language Models
Phase 6: Lightweight FastAPI REST API for Memory Execution Traces & Memory Inspection

Endpoints:
- POST /api/chat — submit prompt, execute pipeline with trace, return response + trace ID
- GET /api/traces/{trace_id} — retrieve complete trace
- GET /api/traces/{trace_id}/events — retrieve ordered trace events
- GET /api/memories — inspect active episodic, semantic, and archive memories
- GET /api/memories/{memory_id} — inspect specific memory with audit provenance
- POST /api/sessions/reset — clear/reset session memories
- GET /api/health — server health and runtime status
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from candidate_generation import (
    CandidateAdmissionGate,
    HeuristicCandidateExtractor,
    HybridCandidateExtractor,
)
from execution_trace import ExecutionTrace, TraceStore
from memory_core import AllocationConfig, MemoryItem, ScoringWeights
from memory_lifecycle import AdaptiveMemoryManager
from model_adapter import BaseLLMAdapter, FakeLLMAdapter, LlamaCppAdapter
from traced_pipeline import run_traced_pipeline

# ============================================================================
# 1. Server Configuration & Session Registry
# ============================================================================

app = FastAPI(
    title="Adaptive Long-Term Memory Execution Trace API",
    version="1.0.0",
    description="Explainability and observability API for dynamic memory-budget allocation under evolving information.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread-safe session managers and trace store
_SESSION_MANAGERS: Dict[str, AdaptiveMemoryManager] = {}
_REGISTRY_LOCK = threading.Lock()
_TRACE_STORE: TraceStore = TraceStore("data/traces.db")
_SHARED_LLM_ADAPTER: Optional[BaseLLMAdapter] = None


def get_llm_adapter(use_fake_llm: bool = False, model_path: Optional[str] = None) -> BaseLLMAdapter:
    """Return model adapter with graceful fallback to FakeLLMAdapter."""
    global _SHARED_LLM_ADAPTER
    if use_fake_llm:
        return FakeLLMAdapter()

    path = model_path or os.environ.get("QWEN_MODEL_PATH")
    if path and os.path.exists(path):
        try:
            return LlamaCppAdapter(model_path=path, n_ctx=2048, verbose=False)
        except Exception:
            return FakeLLMAdapter()

    return FakeLLMAdapter()


def get_or_create_manager(
    session_id: str,
    storage_budget: int = 150,
    enable_admission_gate: bool = True
) -> AdaptiveMemoryManager:
    """Thread-safe retrieval or initialization of session memory manager."""
    with _REGISTRY_LOCK:
        if session_id not in _SESSION_MANAGERS:
            gate = CandidateAdmissionGate() if enable_admission_gate else None
            _SESSION_MANAGERS[session_id] = AdaptiveMemoryManager(
                total_budget=storage_budget,
                scoring_weights=ScoringWeights(),
                allocation_config=AllocationConfig(total_budget=storage_budget),
                admission_gate=gate,
            )
        return _SESSION_MANAGERS[session_id]


# ============================================================================
# 2. Pydantic Request & Response Models
# ============================================================================

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt or conversational turn.")
    session_id: str = Field(default="default_session", description="Session identifier for isolation.")
    storage_budget: int = Field(default=150, ge=20, le=5000, description="Persistent storage token budget.")
    max_context_tokens: int = Field(default=256, ge=32, le=2048, description="Prompt context token budget.")
    use_fake_llm: bool = Field(default=False, description="Force FakeLLMAdapter (mock inference).")
    use_hybrid_extractor: bool = Field(default=False, description="Use HybridCandidateExtractor.")
    enable_admission_gate: bool = Field(default=True, description="Enable quality & confidence admission gating.")


class ChatResponse(BaseModel):
    response: str
    trace_id: str
    session_id: str
    status: str
    duration_ms: float
    cited_memory_ids: List[str]


class ResetSessionRequest(BaseModel):
    session_id: str = Field(default="default_session", description="Session identifier to reset.")


# ============================================================================
# 3. API Endpoints
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(req: ChatRequest):
    """
    Submits a prompt, runs the instrumented memory pipeline, and returns the generated
    response alongside the unique execution trace ID.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    manager = get_or_create_manager(
        session_id=req.session_id,
        storage_budget=req.storage_budget,
        enable_admission_gate=req.enable_admission_gate
    )

    llm_adapter = get_llm_adapter(use_fake_llm=req.use_fake_llm)

    extractor = None
    if req.use_hybrid_extractor:
        extractor = HybridCandidateExtractor(llm_adapter=llm_adapter, trigger_slm_on_multi_clause=True)

    admission_gate = CandidateAdmissionGate() if req.enable_admission_gate else None

    try:
        pipeline_res = run_traced_pipeline(
            prompt=req.prompt,
            session_id=req.session_id,
            manager=manager,
            llm_adapter=llm_adapter,
            admission_gate=admission_gate,
            candidate_extractor=extractor,
            storage_budget=req.storage_budget,
            max_context_tokens=req.max_context_tokens,
            trace_store=_TRACE_STORE,
        )

        trace: ExecutionTrace = pipeline_res["trace"]
        cited_memories = trace.correlations.get("context_memory_ids", [])

        return ChatResponse(
            response=pipeline_res["response"],
            trace_id=trace.trace_id,
            session_id=trace.session_id,
            status=trace.status.value,
            duration_ms=round(trace.total_duration_ms or 0.0, 2),
            cited_memory_ids=cited_memories,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")


@app.get("/api/traces/{trace_id}")
def get_trace_endpoint(trace_id: str):
    """Retrieves full structured trace by trace_id."""
    trace = _TRACE_STORE.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return trace.to_dict()


@app.get("/api/traces/{trace_id}/events")
def get_trace_events_endpoint(trace_id: str):
    """Retrieves ordered stage events for a given trace_id."""
    events = _TRACE_STORE.get_trace_events(trace_id)
    if events is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return {
        "trace_id": trace_id,
        "event_count": len(events),
        "events": events,
    }


@app.get("/api/memories")
def get_memories_endpoint(session_id: str = Query(default="default_session")):
    """Inspects stored memories (episodic, semantic, archive) for a session."""
    with _REGISTRY_LOCK:
        manager = _SESSION_MANAGERS.get(session_id)

    if not manager:
        return {
            "session_id": session_id,
            "storage_budget": 150,
            "used_tokens": 0,
            "remaining_tokens": 150,
            "episodic_store": [],
            "semantic_store": [],
            "archive_store": [],
            "rejected_candidates": [],
        }

    total_used = sum(m.token_cost for m in manager.episodic_store + manager.semantic_store)
    return {
        "session_id": session_id,
        "storage_budget": manager.total_budget,
        "used_tokens": total_used,
        "remaining_tokens": max(0, manager.total_budget - total_used),
        "episodic_store": [m.to_dict() for m in manager.episodic_store],
        "semantic_store": [m.to_dict() for m in manager.semantic_store],
        "archive_store": [m.to_dict() for m in manager.archive_store],
        "rejected_candidates": [
            {
                "memory": c.to_dict(),
                "decision": getattr(d, "value", str(d)),
                "reason": r,
            }
            for c, d, r in manager.rejected_candidates
        ],
    }


@app.get("/api/memories/{memory_id}")
def get_memory_detail_endpoint(
    memory_id: str,
    session_id: str = Query(default="default_session")
):
    """Inspects a specific memory item and its audit log provenance within a session."""
    with _REGISTRY_LOCK:
        manager = _SESSION_MANAGERS.get(session_id)

    if not manager:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' has no memories.")

    # Search stores
    target_mem: Optional[MemoryItem] = None
    store_type = "unknown"

    for m in manager.episodic_store:
        if m.id == memory_id:
            target_mem = m
            store_type = "episodic"
            break

    if not target_mem:
        for m in manager.semantic_store:
            if m.id == memory_id:
                target_mem = m
                store_type = "semantic"
                break

    if not target_mem:
        for m in manager.archive_store:
            if m.id == memory_id:
                target_mem = m
                store_type = "archive"
                break

    if not target_mem:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found in session '{session_id}'.")

    # Find relevant audit history
    audit_history = [
        e.to_dict() for e in manager.audit_log
        if e.target_memory_id == memory_id or memory_id in e.related_memory_ids
    ]

    return {
        "session_id": session_id,
        "store_type": store_type,
        "memory": target_mem.to_dict(),
        "audit_history": audit_history,
    }


@app.post("/api/sessions/reset")
def reset_session_endpoint(req: ResetSessionRequest):
    """Resets memory state for a given session identifier."""
    with _REGISTRY_LOCK:
        if req.session_id in _SESSION_MANAGERS:
            del _SESSION_MANAGERS[req.session_id]
    return {
        "status": "success",
        "message": f"Session '{req.session_id}' reset successfully.",
    }


@app.get("/api/health")
def health_endpoint():
    """Health check and runtime capabilities."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions_count": len(_SESSION_MANAGERS),
        "qwen_model_path_configured": bool(os.environ.get("QWEN_MODEL_PATH")),
    }
