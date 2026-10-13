"""
Adaptive Long-Term Memory for Language Models
Phase 6: Structured Memory Execution Trace Backend & Observability

Provides typed data structures, stage enumerations, execution tracer,
and SQLite persistence for full pipeline observability.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# 1. Pipeline Stage & Status Enums
# ============================================================================

class TraceStage(str, Enum):
    PROMPT_RECEIVED = "prompt_received"
    CANDIDATE_EXTRACTION = "candidate_extraction"
    CANDIDATE_NORMALIZATION = "candidate_normalization"
    UTILITY_SCORING = "utility_scoring"
    ADMISSION_GATE = "admission_gate"
    DEDUPLICATION = "deduplication"
    CONFLICT_RESOLUTION = "conflict_resolution"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    LIFECYCLE_TRANSITIONS = "lifecycle_transitions"
    ADAPTIVE_ALLOCATION = "adaptive_allocation"
    PERSISTENCE = "persistence"
    MEMORY_RETRIEVAL = "memory_retrieval"
    RERANKING = "reranking"
    CONTEXT_ASSEMBLY = "context_assembly"
    SLM_GENERATION = "slm_generation"
    RESPONSE_COMPLETED = "response_completed"


class EventStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class OverallStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# 2. Trace Event & Execution Trace Data Structures
# ============================================================================

@dataclass
class TraceEvent:
    event_id: str
    trace_id: str
    sequence_number: int
    stage: TraceStage
    status: EventStatus
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    summary: str = ""
    candidate_ids: List[str] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    decision: Optional[str] = None
    explanation: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "sequence_number": self.sequence_number,
            "stage": self.stage.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms is not None else None,
            "summary": self.summary,
            "candidate_ids": self.candidate_ids,
            "memory_ids": self.memory_ids,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "decision": self.decision,
            "explanation": self.explanation,
            "error_details": self.error_details,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceEvent:
        return cls(
            event_id=data["event_id"],
            trace_id=data["trace_id"],
            sequence_number=data["sequence_number"],
            stage=TraceStage(data["stage"]),
            status=EventStatus(data["status"]),
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            duration_ms=data.get("duration_ms"),
            summary=data.get("summary", ""),
            candidate_ids=data.get("candidate_ids", []),
            memory_ids=data.get("memory_ids", []),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            decision=data.get("decision"),
            explanation=data.get("explanation"),
            error_details=data.get("error_details"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionTrace:
    trace_id: str
    session_id: str
    prompt: str
    status: OverallStatus
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    events: List[TraceEvent] = field(default_factory=list)
    response: Optional[str] = None
    error: Optional[str] = None
    correlations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": round(self.total_duration_ms, 2) if self.total_duration_ms is not None else None,
            "events": [e.to_dict() for e in self.events],
            "response": self.response,
            "error": self.error,
            "correlations": self.correlations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionTrace:
        events = [TraceEvent.from_dict(e) for e in data.get("events", [])]
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            trace_id=data["trace_id"],
            session_id=data.get("session_id", "default"),
            prompt=data.get("prompt", ""),
            status=OverallStatus(data.get("status", OverallStatus.COMPLETED.value)),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            total_duration_ms=data.get("total_duration_ms"),
            events=events,
            response=data.get("response"),
            error=data.get("error"),
            correlations=data.get("correlations", {}),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# 3. Execution Tracer
# ============================================================================

class ExecutionTracer:
    """
    Thread-safe execution tracer managing stage lifecycles and structured event capture.
    """

    def __init__(
        self,
        trace_id: Optional[str] = None,
        session_id: str = "default_session",
        prompt: str = ""
    ):
        self.trace_id = trace_id or f"trc_{uuid.uuid4().hex[:12]}"
        self.session_id = session_id
        self.prompt = prompt
        self.start_time = time.time()
        self._perf_start = time.perf_counter()

        self._lock = threading.Lock()
        self._seq = 0
        self._active_events: Dict[str, TraceEvent] = {}
        self._events: List[TraceEvent] = []
        self._correlations: Dict[str, Any] = {
            "candidate_ids": [],
            "admitted_memory_ids": [],
            "rejected_candidate_ids": [],
            "retrieved_memory_ids": [],
            "context_memory_ids": [],
            "superseded_memory_ids": [],
        }

    def start_stage(
        self,
        stage: TraceStage,
        inputs: Optional[Dict[str, Any]] = None,
        summary: str = ""
    ) -> str:
        """Start a new pipeline stage and return its unique event_id."""
        with self._lock:
            self._seq += 1
            event_id = f"evt_{self._seq:02d}_{stage.value}"
            event = TraceEvent(
                event_id=event_id,
                trace_id=self.trace_id,
                sequence_number=self._seq,
                stage=stage,
                status=EventStatus.RUNNING,
                start_time=time.time(),
                summary=summary,
                inputs=inputs or {},
                metadata={"perf_start": time.perf_counter()},
            )
            self._active_events[event_id] = event
            self._events.append(event)
            return event_id

    def complete_stage(
        self,
        event_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        decision: Optional[str] = None,
        explanation: Optional[str] = None,
        candidate_ids: Optional[List[str]] = None,
        memory_ids: Optional[List[str]] = None,
        summary: Optional[str] = None
    ) -> None:
        """Mark an active stage as completed with structured evidence."""
        with self._lock:
            event = self._active_events.pop(event_id, None)
            if not event:
                return

            event.end_time = time.time()
            perf_start = event.metadata.pop("perf_start", None)
            if perf_start is not None:
                event.duration_ms = (time.perf_counter() - perf_start) * 1000.0
            else:
                event.duration_ms = (event.end_time - event.start_time) * 1000.0

            event.status = EventStatus.COMPLETED
            if outputs is not None:
                event.outputs = outputs
            if decision is not None:
                event.decision = decision
            if explanation is not None:
                event.explanation = explanation
            if candidate_ids is not None:
                event.candidate_ids = candidate_ids
                self._correlations["candidate_ids"].extend(candidate_ids)
            if memory_ids is not None:
                event.memory_ids = memory_ids
            if summary is not None:
                event.summary = summary

    def skip_stage(self, stage: TraceStage, reason: str = "") -> str:
        """Record an explicit skipped stage."""
        with self._lock:
            self._seq += 1
            event_id = f"evt_{self._seq:02d}_{stage.value}"
            now = time.time()
            event = TraceEvent(
                event_id=event_id,
                trace_id=self.trace_id,
                sequence_number=self._seq,
                stage=stage,
                status=EventStatus.SKIPPED,
                start_time=now,
                end_time=now,
                duration_ms=0.0,
                summary=f"Skipped: {reason}" if reason else "Skipped",
                decision="skipped",
                explanation=reason or "Stage not required for current configuration",
            )
            self._events.append(event)
            return event_id

    def fail_stage(
        self,
        event_id: str,
        error: Any,
        safe_summary: str = "",
        recovered: bool = False
    ) -> None:
        """Mark an active stage as failed with safe diagnostic details."""
        with self._lock:
            event = self._active_events.pop(event_id, None)
            if not event:
                return

            event.end_time = time.time()
            perf_start = event.metadata.pop("perf_start", None)
            if perf_start is not None:
                event.duration_ms = (time.perf_counter() - perf_start) * 1000.0

            event.status = EventStatus.FAILED
            event.decision = "recovered_via_fallback" if recovered else "stage_failed"
            event.error_details = {
                "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
                "message": str(error),
                "safe_summary": safe_summary or "An error occurred during stage execution.",
                "recovered": recovered,
            }
            if safe_summary:
                event.explanation = safe_summary

    def add_correlation(self, key: str, value: Any) -> None:
        """Register correlation entity (e.g. cited memories, superseded IDs)."""
        with self._lock:
            if key in self._correlations and isinstance(self._correlations[key], list):
                if isinstance(value, list):
                    self._correlations[key].extend(value)
                else:
                    self._correlations[key].append(value)
            else:
                self._correlations[key] = value

    def finalize(
        self,
        response: Optional[str] = None,
        error: Optional[str] = None
    ) -> ExecutionTrace:
        """Finalize and return the complete immutable ExecutionTrace."""
        with self._lock:
            end_time = time.time()
            total_duration_ms = (time.perf_counter() - self._perf_start) * 1000.0

            overall_status = OverallStatus.FAILED if error else OverallStatus.COMPLETED

            # Close any orphaned active events
            for event in self._active_events.values():
                event.end_time = end_time
                event.status = EventStatus.FAILED if error else EventStatus.COMPLETED

            return ExecutionTrace(
                trace_id=self.trace_id,
                session_id=self.session_id,
                prompt=self.prompt,
                status=overall_status,
                start_time=self.start_time,
                end_time=end_time,
                total_duration_ms=total_duration_ms,
                events=list(self._events),
                response=response,
                error=error,
                correlations=dict(self._correlations),
            )


# ============================================================================
# 4. SQLite Trace Store
# ============================================================================

class TraceStore:
    """
    Persistent SQLite storage for execution traces with session isolation.
    """

    def __init__(self, db_path: str = "data/traces.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._is_memory = (db_path == ":memory:")
        if not self._is_memory:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    prompt TEXT,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    duration_ms REAL,
                    response TEXT,
                    trace_json TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON traces(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON traces(start_time)")
            self._conn.commit()

    def save_trace(self, trace: ExecutionTrace) -> None:
        """Persist an ExecutionTrace."""
        with self._lock:
            cursor = self._conn.cursor()
            trace_json = json.dumps(trace.to_dict(), ensure_ascii=False)
            cursor.execute("""
                INSERT OR REPLACE INTO traces (
                    trace_id, session_id, prompt, status, start_time, duration_ms, response, trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.trace_id,
                trace.session_id,
                trace.prompt,
                trace.status.value,
                trace.start_time,
                trace.total_duration_ms,
                trace.response,
                trace_json,
            ))
            self._conn.commit()

    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Retrieve full trace by trace_id."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT trace_json FROM traces WHERE trace_id = ?", (trace_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row["trace_json"])
            return ExecutionTrace.from_dict(data)

    def get_trace_events(self, trace_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve ordered events for a trace."""
        trace = self.get_trace(trace_id)
        if not trace:
            return None
        return [e.to_dict() for e in trace.events]

    def list_traces(
        self,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List recent traces, optionally filtered by session_id."""
        with self._lock:
            cursor = self._conn.cursor()
            if session_id:
                cursor.execute("""
                    SELECT trace_id, session_id, prompt, status, start_time, duration_ms, response
                    FROM traces WHERE session_id = ?
                    ORDER BY start_time DESC LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT trace_id, session_id, prompt, status, start_time, duration_ms, response
                    FROM traces
                    ORDER BY start_time DESC LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace by trace_id."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all stored traces."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM traces")
            self._conn.commit()

    def close(self) -> None:
        """Close connection cleanly."""
        with self._lock:
            if self._conn:
                self._conn.close()
