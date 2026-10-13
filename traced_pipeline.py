"""
Adaptive Long-Term Memory for Language Models
Phase 6: Instrumented Pipeline Orchestration with High-Fidelity Tracing

Orchestrates all 16 memory pipeline stages, collecting structured operational evidence
at every step without mutating core algorithm behaviors.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from candidate_generation import (
    AdmissionDecision,
    AdmissionResult,
    BaseCandidateExtractor,
    CandidateAdmissionGate,
    HeuristicCandidateExtractor,
    HybridCandidateExtractor,
    decompose_multi_fact_sentence,
    is_filler_sentence,
)
from execution_trace import (
    EventStatus,
    ExecutionTrace,
    ExecutionTracer,
    TraceStage,
    TraceStore,
)
from memory_core import (
    AllocationConfig,
    LifecycleStatus,
    MemoryItem,
    MemoryType,
    ScoringWeights,
    estimate_token_cost,
    score_memories,
)
from memory_lifecycle import (
    AdaptiveMemoryManager,
    AuditAction,
    AuditLogEntry,
    consolidate_episodic_memories,
    deduplicate_semantically,
    detect_conflicts_and_supersede,
    evaluate_lifecycle_transitions,
    maintain_memory_budget,
)
from memory_retrieval import (
    PromptBudgetTracker,
    assemble_memory_context,
    compute_query_relevance,
    extract_query_point_in_time,
    is_historical_query,
    is_memory_valid_at_timestamp,
    retrieve_memories,
)
from model_adapter import BaseLLMAdapter, FakeLLMAdapter, heuristic_bpe_token_count

logger = logging.getLogger(__name__)


def run_traced_pipeline(
    prompt: str,
    session_id: str = "default_session",
    manager: Optional[AdaptiveMemoryManager] = None,
    llm_adapter: Optional[BaseLLMAdapter] = None,
    admission_gate: Optional[CandidateAdmissionGate] = None,
    candidate_extractor: Optional[BaseCandidateExtractor] = None,
    storage_budget: int = 150,
    max_context_tokens: int = 256,
    scoring_weights: Optional[ScoringWeights] = None,
    allocation_config: Optional[AllocationConfig] = None,
    current_timestamp: Optional[float] = None,
    trace_store: Optional[TraceStore] = None,
    tracer: Optional[ExecutionTracer] = None,
) -> Dict[str, Any]:
    """
    Executes the full long-term memory pipeline with fine-grained execution tracing across all 16 stages.
    Returns:
        {
            "response": str,
            "trace": ExecutionTrace,
            "manager": AdaptiveMemoryManager,
            "correlations": Dict[str, Any]
        }
    """
    if current_timestamp is None:
        current_timestamp = time.time()

    # Initialize model adapter if omitted
    if llm_adapter is None:
        llm_adapter = FakeLLMAdapter()

    # Initialize memory manager if omitted
    if manager is None:
        manager = AdaptiveMemoryManager(
            total_budget=storage_budget,
            scoring_weights=scoring_weights or ScoringWeights(),
            allocation_config=allocation_config or AllocationConfig(total_budget=storage_budget),
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor,
        )
    else:
        if admission_gate is not None:
            manager.admission_gate = admission_gate
        if candidate_extractor is not None:
            manager.candidate_extractor = candidate_extractor

    # Initialize execution tracer
    if tracer is None:
        tracer = ExecutionTracer(session_id=session_id, prompt=prompt)

    try:
        # ====================================================================
        # Stage 1: Prompt Received
        # ====================================================================
        evt_prompt = tracer.start_stage(
            TraceStage.PROMPT_RECEIVED,
            inputs={
                "prompt": prompt,
                "session_id": session_id,
                "storage_budget": manager.total_budget,
                "max_context_tokens": max_context_tokens,
                "timestamp": current_timestamp,
            },
            summary=f"User prompt received in session '{session_id}'"
        )
        tracer.complete_stage(
            evt_prompt,
            outputs={
                "prompt_char_length": len(prompt),
                "is_question": "?" in prompt,
                "received_at": current_timestamp,
            },
            decision="accepted_for_processing",
            explanation="Prompt ingested successfully for extraction, memory update, and response retrieval."
        )

        # ====================================================================
        # Stage 2: Candidate Extraction
        # ====================================================================
        extractor_inst = manager.candidate_extractor or candidate_extractor or HeuristicCandidateExtractor()
        extractor_name = type(extractor_inst).__name__

        evt_extract = tracer.start_stage(
            TraceStage.CANDIDATE_EXTRACTION,
            inputs={"extractor": extractor_name, "turn_count": 1},
            summary=f"Extracting candidate memories using {extractor_name}"
        )

        conversation_history = [{"role": "user", "content": prompt, "timestamp": current_timestamp}]
        extracted_candidates: List[MemoryItem] = []
        extraction_fallback_occurred = False

        try:
            extracted_candidates = extractor_inst.extract_candidates(
                conversation_history=conversation_history,
                session_id=session_id,
                current_timestamp=current_timestamp,
            )
        except Exception as exc:
            logger.warning("Primary extractor failed (%s). Falling back to heuristic extractor.", exc)
            extraction_fallback_occurred = True
            heuristic_fallback = HeuristicCandidateExtractor()
            extracted_candidates = heuristic_fallback.extract_candidates(
                conversation_history=conversation_history,
                session_id=session_id,
                current_timestamp=current_timestamp,
            )

        fallback_flag = (
            extraction_fallback_occurred
            or getattr(extractor_inst, "last_fallback", False)
            or (hasattr(extractor_inst, "slm") and bool(extractor_inst.slm) and getattr(extractor_inst.slm, "last_fallback", False))
        )

        cand_ids = [c.id for c in extracted_candidates]
        tracer.complete_stage(
            evt_extract,
            outputs={
                "candidate_count": len(extracted_candidates),
                "extractor_used": extractor_name,
                "fallback_triggered": fallback_flag,
                "candidates": [
                    {
                        "id": c.id,
                        "content": c.content,
                        "memory_type": c.memory_type.value,
                        "token_cost": c.token_cost,
                        "provenance": c.metadata.get("provenance", []),
                    }
                    for c in extracted_candidates
                ],
            },
            decision="candidates_extracted" if extracted_candidates else "no_candidates_found",
            explanation=(
                f"Extracted {len(extracted_candidates)} atomic candidate(s) via {extractor_name}"
                + (" (with fallback)" if extraction_fallback_occurred else "")
            ),
            candidate_ids=cand_ids
        )

        # ====================================================================
        # Stage 3: Candidate Normalization & Decomposition
        # ====================================================================
        evt_norm = tracer.start_stage(
            TraceStage.CANDIDATE_NORMALIZATION,
            inputs={"raw_candidate_count": len(extracted_candidates)},
            summary="Normalizing candidate propositions and analyzing linguistic properties"
        )
        norm_details = []
        for c in extracted_candidates:
            norm_details.append({
                "id": c.id,
                "content": c.content,
                "is_uncertain": c.metadata.get("uncertainty_detected", False),
                "is_negation": c.metadata.get("is_negation", False),
                "is_historical": c.metadata.get("is_historical_fact", False),
                "slot_type": c.metadata.get("slot_type"),
            })
        tracer.complete_stage(
            evt_norm,
            outputs={"normalized_candidates": norm_details},
            decision="normalized",
            explanation="Linguistic anchors (uncertainty, negation, historical validity) tagged in metadata."
        )

        # ====================================================================
        # Stage 4: Utility Scoring
        # ====================================================================
        evt_score = tracer.start_stage(
            TraceStage.UTILITY_SCORING,
            inputs={"scoring_weights": asdict(manager.scoring_weights)},
            summary="Computing multi-factor utility score U(m) for candidates"
        )
        scored_candidates = score_memories(
            memories=extracted_candidates,
            weights=manager.scoring_weights,
            current_timestamp=current_timestamp
        )
        score_evidence = []
        for c in scored_candidates:
            score_evidence.append({
                "id": c.id,
                "content": c.content,
                "utility_score": round(c.utility_score, 4),
                "importance": round(c.importance, 4),
                "recency": round(c.recency, 4),
                "frequency": c.frequency,
                "stability": round(c.stability, 4),
                "confidence": round(c.consistency_confidence, 4),
                "token_cost": c.token_cost,
                "utility_density": round(c.utility_score / max(1, c.token_cost), 4),
            })
        tracer.complete_stage(
            evt_score,
            outputs={"scored_candidates": score_evidence},
            decision="scored",
            explanation="Utility scores computed using U(m) = w_r*R + w_i*I + w_f*F + w_t*T + w_c*C."
        )

        # ====================================================================
        # Stage 5: Admission Gate
        # ====================================================================
        evt_gate = tracer.start_stage(
            TraceStage.ADMISSION_GATE,
            inputs={
                "gate_configured": manager.admission_gate is not None,
                "min_importance": getattr(manager.admission_gate, "min_importance_threshold", 0.20),
                "min_confidence": getattr(manager.admission_gate, "min_confidence_threshold", 0.30),
            },
            summary="Evaluating candidate quality and confidence admission thresholds"
        )

        admitted_candidates: List[MemoryItem] = []
        rejected_candidates: List[Tuple[MemoryItem, Any, str]] = []

        if manager.admission_gate is not None and hasattr(manager.admission_gate, "filter_quality"):
            admitted_candidates, rejected_candidates = manager.admission_gate.filter_quality(
                scored_candidates,
                audit_entries=manager.audit_log,
                current_timestamp=current_timestamp
            )
            manager.rejected_candidates.extend(rejected_candidates)
        else:
            # Default backward compatibility: admit all scored candidates
            admitted_candidates = scored_candidates

        rejection_details = [
            {
                "id": c.id,
                "content": c.content,
                "decision": getattr(d, "value", str(d)),
                "reason": r,
            }
            for c, d, r in rejected_candidates
        ]
        tracer.add_correlation("admitted_memory_ids", [c.id for c in admitted_candidates])
        tracer.add_correlation("rejected_candidate_ids", [c.id for c, _, _ in rejected_candidates])

        tracer.complete_stage(
            evt_gate,
            outputs={
                "admitted_count": len(admitted_candidates),
                "rejected_count": len(rejected_candidates),
                "rejections": rejection_details,
            },
            decision="filtered" if manager.admission_gate else "bypassed_default",
            explanation=(
                f"{len(admitted_candidates)} candidate(s) admitted, {len(rejected_candidates)} rejected."
                if manager.admission_gate
                else "Admission gate omitted; all scored candidates admitted by default."
            ),
            candidate_ids=[c.id for c in admitted_candidates]
        )

        # Partition admitted candidates by memory type
        new_ep = [m for m in admitted_candidates if m.memory_type == MemoryType.EPISODIC]
        new_sem = [m for m in admitted_candidates if m.memory_type == MemoryType.SEMANTIC]

        # ====================================================================
        # Stage 6: Deduplication
        # ====================================================================
        evt_dedup = tracer.start_stage(
            TraceStage.DEDUPLICATION,
            inputs={
                "incoming_episodic": len(new_ep),
                "incoming_semantic": len(new_sem),
                "existing_episodic": len(manager.episodic_store),
                "existing_semantic": len(manager.semantic_store),
            },
            summary="Checking semantic similarity and merging redundant observations"
        )
        prev_audit_len = len(manager.audit_log)
        manager.episodic_store, distinct_ep = deduplicate_semantically(
            incoming_memories=new_ep,
            existing_memories=manager.episodic_store,
            audit_entries=manager.audit_log,
            current_timestamp=current_timestamp
        )
        manager.semantic_store, distinct_sem = deduplicate_semantically(
            incoming_memories=new_sem,
            existing_memories=manager.semantic_store,
            audit_entries=manager.audit_log,
            current_timestamp=current_timestamp
        )
        manager.episodic_store.extend(distinct_ep)
        manager.semantic_store.extend(distinct_sem)

        dedup_audits = [
            e.to_dict() for e in manager.audit_log[prev_audit_len:]
            if e.action == AuditAction.MERGE_DUPLICATE
        ]
        tracer.complete_stage(
            evt_dedup,
            outputs={
                "distinct_episodic_added": len(distinct_ep),
                "distinct_semantic_added": len(distinct_sem),
                "merged_observations_count": len(dedup_audits),
                "merge_events": dedup_audits,
            },
            decision="deduplicated",
            explanation=f"Added {len(distinct_ep) + len(distinct_sem)} distinct items; soft-merged {len(dedup_audits)} duplicates."
        )

        # ====================================================================
        # Stage 7: Conflict Resolution
        # ====================================================================
        evt_conflict = tracer.start_stage(
            TraceStage.CONFLICT_RESOLUTION,
            inputs={"active_memories_checked": len(manager.episodic_store) + len(manager.semantic_store)},
            summary="Detecting temporal contradictions and linking superseded facts"
        )
        prev_audit_len = len(manager.audit_log)
        all_active = manager.episodic_store + manager.semantic_store
        detect_conflicts_and_supersede(
            all_memories=all_active,
            audit_entries=manager.audit_log,
            current_timestamp=current_timestamp
        )
        supersede_audits = [
            e.to_dict() for e in manager.audit_log[prev_audit_len:]
            if e.action == AuditAction.SUPERSEDE
        ]
        tracer.add_correlation("superseded_memory_ids", [e["target_memory_id"] for e in supersede_audits])
        tracer.complete_stage(
            evt_conflict,
            outputs={
                "conflicts_detected": len(supersede_audits),
                "supersede_events": supersede_audits,
            },
            decision="conflicts_resolved" if supersede_audits else "no_conflicts",
            explanation=(
                f"Resolved {len(supersede_audits)} evolving slot conflict(s) with historical link."
                if supersede_audits else "No contradicting or evolving facts detected."
            )
        )

        # ====================================================================
        # Stage 8: Memory Consolidation
        # ====================================================================
        evt_consol = tracer.start_stage(
            TraceStage.MEMORY_CONSOLIDATION,
            inputs={"episodic_count": len(manager.episodic_store)},
            summary="Consolidating recurring episodic memories into durable semantic knowledge"
        )
        prev_audit_len = len(manager.audit_log)
        manager.episodic_store, manager.semantic_store = consolidate_episodic_memories(
            episodic_memories=manager.episodic_store,
            semantic_memories=manager.semantic_store,
            audit_entries=manager.audit_log,
            current_timestamp=current_timestamp
        )
        consol_audits = [
            e.to_dict() for e in manager.audit_log[prev_audit_len:]
            if e.action == AuditAction.CONSOLIDATE
        ]
        tracer.complete_stage(
            evt_consol,
            outputs={
                "consolidations_count": len(consol_audits),
                "consolidation_events": consol_audits,
                "current_episodic_count": len(manager.episodic_store),
                "current_semantic_count": len(manager.semantic_store),
            },
            decision="consolidated" if consol_audits else "no_consolidation_needed",
            explanation=(
                f"Promoted {len(consol_audits)} recurring episodic observation(s) to semantic knowledge."
                if consol_audits else "No episodic memories met consolidation threshold."
            )
        )

        # Re-score stores
        manager.episodic_store = score_memories(manager.episodic_store, manager.scoring_weights, current_timestamp)
        manager.semantic_store = score_memories(manager.semantic_store, manager.scoring_weights, current_timestamp)

        # ====================================================================
        # Stage 9: Lifecycle Transitions
        # ====================================================================
        evt_trans = tracer.start_stage(
            TraceStage.LIFECYCLE_TRANSITIONS,
            summary="Evaluating lifecycle status transitions (TTL expiration, stabilization)"
        )
        prev_audit_len = len(manager.audit_log)
        manager.episodic_store = evaluate_lifecycle_transitions(
            manager.episodic_store, current_timestamp=current_timestamp, audit_entries=manager.audit_log
        )
        manager.semantic_store = evaluate_lifecycle_transitions(
            manager.semantic_store, current_timestamp=current_timestamp, audit_entries=manager.audit_log
        )
        trans_audits = [
            e.to_dict() for e in manager.audit_log[prev_audit_len:]
            if e.action == AuditAction.TRANSITION_STATUS
        ]
        tracer.complete_stage(
            evt_trans,
            outputs={"transitions_count": len(trans_audits), "transition_events": trans_audits},
            decision="transitions_evaluated",
            explanation=f"Evaluated lifecycle statuses; {len(trans_audits)} status update(s) recorded."
        )

        # ====================================================================
        # Stage 10: Adaptive Allocation
        # ====================================================================
        evt_alloc = tracer.start_stage(
            TraceStage.ADAPTIVE_ALLOCATION,
            inputs={
                "total_budget": manager.total_budget,
                "min_episodic_share": manager.allocation_config.min_episodic_share,
                "max_episodic_share": manager.allocation_config.max_episodic_share,
            },
            summary="Rebalancing persistent-memory budgets and evicting low-utility overflow"
        )
        (
            manager.episodic_store,
            manager.semantic_store,
            new_evicted,
            maint_summary
        ) = maintain_memory_budget(
            episodic_memories=manager.episodic_store,
            semantic_memories=manager.semantic_store,
            total_budget=manager.total_budget,
            allocation_config=manager.allocation_config,
            audit_entries=manager.audit_log,
            current_timestamp=current_timestamp
        )
        manager.archive_store.extend(new_evicted)

        tracer.complete_stage(
            evt_alloc,
            outputs={
                "allocated_budget": {
                    "episodic": maint_summary["episodic_budget"],
                    "semantic": maint_summary["semantic_budget"],
                    "total": maint_summary["total_budget"],
                },
                "tokens_used": {
                    "episodic": maint_summary["used_tokens_episodic"],
                    "semantic": maint_summary["used_tokens_semantic"],
                    "total": maint_summary["total_used_tokens"],
                },
                "remaining_capacity": maint_summary["total_budget"] - maint_summary["total_used_tokens"],
                "evicted_count": maint_summary["evicted_count"],
                "active_items_count": maint_summary["active_count"],
            },
            decision="budget_maintained",
            explanation=(
                f"Dynamic budget allocation enforced. Episodic: {maint_summary['used_tokens_episodic']}/{maint_summary['episodic_budget']}, "
                f"Semantic: {maint_summary['used_tokens_semantic']}/{maint_summary['semantic_budget']}. "
                f"Evicted {maint_summary['evicted_count']} item(s)."
            )
        )

        # ====================================================================
        # Stage 11: Persistence
        # ====================================================================
        if manager.persistence_path:
            evt_persist = tracer.start_stage(
                TraceStage.PERSISTENCE,
                inputs={"persistence_path": manager.persistence_path},
                summary=f"Persisting memory state to {manager.persistence_path}"
            )
            manager.save_state(manager.persistence_path, maint_summary)
            tracer.complete_stage(
                evt_persist,
                outputs={"saved_path": os.path.abspath(manager.persistence_path)},
                decision="persisted_to_disk",
                explanation="Active episodic/semantic stores, archives, and audit logs saved."
            )
        else:
            tracer.skip_stage(TraceStage.PERSISTENCE, reason="In-memory store; no persistence_path configured.")

        # ====================================================================
        # Stage 12: Memory Retrieval
        # ====================================================================
        evt_ret = tracer.start_stage(
            TraceStage.MEMORY_RETRIEVAL,
            inputs={"query": prompt, "top_k": 5},
            summary="Retrieving relevant memory items across stores"
        )
        is_hist = is_historical_query(prompt)
        point_in_time = extract_query_point_in_time(prompt)

        retrieved_raw = retrieve_memories(
            query=prompt,
            episodic_store=manager.episodic_store,
            semantic_store=manager.semantic_store,
            archive_store=manager.archive_store,
            top_k=5
        )
        tracer.add_correlation("retrieved_memory_ids", [m.id for m, _ in retrieved_raw])

        tracer.complete_stage(
            evt_ret,
            outputs={
                "is_historical_query": is_hist,
                "point_in_time": point_in_time,
                "retrieved_count": len(retrieved_raw),
                "retrieved_memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "type": m.memory_type.value,
                        "status": m.lifecycle_status.value,
                        "relevance_score": round(score, 4),
                        "token_cost": m.token_cost,
                    }
                    for m, score in retrieved_raw
                ],
            },
            decision="memories_retrieved" if retrieved_raw else "no_relevant_memories",
            explanation=f"Retrieved {len(retrieved_raw)} candidate memory assertion(s) for query.",
            memory_ids=[m.id for m, _ in retrieved_raw]
        )

        # ====================================================================
        # Stage 13: Reranking
        # ====================================================================
        evt_rerank = tracer.start_stage(
            TraceStage.RERANKING,
            inputs={"retrieved_count": len(retrieved_raw), "margin_pruning_ratio": 0.70},
            summary="Applying multi-factor ranking and margin pruning"
        )
        ranked_evidence = [
            {
                "id": m.id,
                "content": m.content,
                "final_rank_score": score,
                "utility_score": round(m.utility_score, 4),
            }
            for m, score in retrieved_raw
        ]
        tracer.complete_stage(
            evt_rerank,
            outputs={"ranked_memories": ranked_evidence},
            decision="reranked",
            explanation="Ranked memories based on relevance, utility, recency, and stability."
        )

        # ====================================================================
        # Stage 14: Context Assembly
        # ====================================================================
        evt_ctx = tracer.start_stage(
            TraceStage.CONTEXT_ASSEMBLY,
            inputs={"max_context_tokens": max_context_tokens},
            summary="Assembling retrieved memories into bounded prompt context"
        )
        tokenizer_fn = getattr(llm_adapter, "count_tokens", heuristic_bpe_token_count)
        context_result = assemble_memory_context(
            query=prompt,
            retrieved_memories=retrieved_raw,
            max_context_tokens=max_context_tokens,
            tokenizer_fn=tokenizer_fn,
        )
        selected_mem_ids = [m.id for m in context_result.selected_memories]
        tracer.add_correlation("context_memory_ids", selected_mem_ids)

        tracer.complete_stage(
            evt_ctx,
            outputs={
                "context_token_cost": context_result.context_token_cost,
                "context_budget": context_result.context_budget,
                "selected_memories_count": len(context_result.selected_memories),
                "selected_memory_ids": selected_mem_ids,
                "formatted_context_preview": context_result.formatted_context[:200] + ("..." if len(context_result.formatted_context) > 200 else ""),
            },
            decision="context_assembled",
            explanation=f"Assembled {len(selected_mem_ids)} memory item(s) using {context_result.context_token_cost}/{context_result.context_budget} context tokens.",
            memory_ids=selected_mem_ids
        )

        # ====================================================================
        # Stage 15: SLM Generation
        # ====================================================================
        evt_gen = tracer.start_stage(
            TraceStage.SLM_GENERATION,
            inputs={
                "model_adapter": type(llm_adapter).__name__,
                "system_prompt": context_result.system_prompt,
                "prompt": prompt,
            },
            summary="Invoking language model adapter for response synthesis"
        )

        messages = [
            {
                "role": "system",
                "content": f"{context_result.system_prompt}\n\nRELEVANT MEMORIES:\n{context_result.formatted_context}",
            },
            {"role": "user", "content": prompt},
        ]

        t0 = time.perf_counter()
        response_text = llm_adapter.generate_chat(
            messages=messages,
            max_tokens=256,
            temperature=0.7
        )
        gen_duration_ms = (time.perf_counter() - t0) * 1000.0
        response_tokens = tokenizer_fn(response_text)

        tracer.complete_stage(
            evt_gen,
            outputs={
                "response_text": response_text,
                "duration_ms": round(gen_duration_ms, 2),
                "response_token_cost": response_tokens,
            },
            decision="response_generated",
            explanation=f"Generated response ({response_tokens} tokens) in {gen_duration_ms:.1f}ms."
        )

        # ====================================================================
        # Stage 16: Response Completed
        # ====================================================================
        evt_comp = tracer.start_stage(
            TraceStage.RESPONSE_COMPLETED,
            inputs={"response_length": len(response_text)},
            summary="Finalizing pipeline execution and packaging trace results"
        )
        tracer.complete_stage(
            evt_comp,
            outputs={
                "final_response": response_text,
                "cited_memory_ids": selected_mem_ids,
            },
            decision="pipeline_succeeded",
            explanation="Pipeline completed all stages successfully."
        )

        # Finalize trace
        trace = tracer.finalize(response=response_text)
        if trace_store is not None:
            trace_store.save_trace(trace)

        return {
            "response": response_text,
            "trace": trace,
            "manager": manager,
            "correlations": trace.correlations,
        }

    except Exception as exc:
        logger.exception("Pipeline failure during execution: %s", exc)
        trace = tracer.finalize(error=str(exc))
        if trace_store is not None:
            trace_store.save_trace(trace)
        raise
