"""
Adaptive Long-Term Memory for Language Models
Phase 2: Lifecycle Management, Semantic Maintenance & Auditability

Pipeline:
Candidate Extraction
  -> Semantic Deduplication & Alias Linking
  -> Conflict & Update Detection (Superseding Stale Facts)
  -> Episodic-to-Semantic Consolidation (Preserving Historical Evidence)
  -> Lifecycle State Transitions (ACTIVE, STABLE, TEMPORARY, SUPERSEDED, EXPIRED)
  -> Budget Maintenance & Eviction
  -> Structured Auditing & State Persistence
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    score_memories,
    select_memories,
)


# ============================================================================
# 1. Audit Logging Schema
# ============================================================================

class AuditAction(str, Enum):
    MERGE_DUPLICATE = "merge_duplicate"
    CONSOLIDATE = "consolidate"
    SUPERSEDE = "supersede"
    TRANSITION_STATUS = "transition_status"
    EVICT = "evict"
    MAINTAIN_BUDGET = "maintain_budget"
    REJECT_CANDIDATE = "reject_candidate"


@dataclass
class AuditLogEntry:
    id: str
    timestamp: float
    action: AuditAction
    target_memory_id: str
    related_memory_ids: List[str]
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditLogEntry:
        data_copy = dict(data)
        data_copy["action"] = AuditAction(data_copy["action"])
        return cls(**data_copy)


# ============================================================================
# 2. Semantic Similarity & Soft Deduplication
# ============================================================================

def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Computes a hybrid similarity metric combining:
    1. Token-level Jaccard similarity (word set overlap)
    2. Character-level Gestalt pattern matching (difflib SequenceMatcher)

    Returns a score in [0.0, 1.0].
    """
    t1_clean = re.sub(r"[^\w\s]", "", text1.lower()).strip()
    t2_clean = re.sub(r"[^\w\s]", "", text2.lower()).strip()

    if t1_clean == t2_clean:
        return 1.0
    if not t1_clean or not t2_clean:
        return 0.0

    # Token Jaccard
    tokens1 = set(t1_clean.split())
    tokens2 = set(t2_clean.split())
    union = tokens1.union(tokens2)
    jaccard = len(tokens1.intersection(tokens2)) / len(union) if union else 0.0

    # Sequence matcher ratio
    seq_ratio = difflib.SequenceMatcher(None, t1_clean, t2_clean).ratio()

    # Weighted combination
    similarity = 0.5 * jaccard + 0.5 * seq_ratio
    return round(similarity, 4)


def deduplicate_semantically(
    incoming_memories: List[MemoryItem],
    existing_memories: List[MemoryItem],
    similarity_threshold: float = 0.80,
    audit_entries: Optional[List[AuditLogEntry]] = None,
    current_timestamp: Optional[float] = None
) -> Tuple[List[MemoryItem], List[MemoryItem]]:
    """
    Soft deduplication: merges incoming memories that are semantically identical
    or near-paraphrases of existing memories.

    Preserves frequency, increases confidence, links variations in metadata,
    and updates provenance without duplicating storage footprint.

    Returns:
        Tuple of (updated_existing_memories, distinct_new_memories)
    """
    if current_timestamp is None:
        current_timestamp = time.time()
    if audit_entries is None:
        audit_entries = []

    distinct_new: List[MemoryItem] = []

    for incoming in incoming_memories:
        matched_existing: Optional[MemoryItem] = None
        best_sim = 0.0

        for existing in existing_memories:
            # Check similarity if same memory type or same source subject
            sim = compute_text_similarity(incoming.content, existing.content)
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                matched_existing = existing

        if matched_existing is not None:
            # Merge into matched_existing
            old_freq = matched_existing.frequency
            matched_existing.frequency += incoming.frequency
            matched_existing.timestamp = max(matched_existing.timestamp, incoming.timestamp)
            matched_existing.consistency_confidence = min(1.0, matched_existing.consistency_confidence + 0.05)

            # Preserve alias/paraphrase
            aliases = matched_existing.metadata.setdefault("aliases", [])
            if incoming.content not in aliases and incoming.content != matched_existing.content:
                aliases.append(incoming.content)

            # Preserve provenance
            existing_prov = matched_existing.metadata.setdefault("provenance", [])
            incoming_prov = incoming.metadata.get("provenance", [])
            existing_prov.extend(incoming_prov)

            # If a temporary memory is reinforced, reactivate it
            if matched_existing.lifecycle_status == LifecycleStatus.TEMPORARY:
                matched_existing.lifecycle_status = LifecycleStatus.ACTIVE

            # Audit log
            audit_entries.append(AuditLogEntry(
                id=str(uuid.uuid4()),
                timestamp=current_timestamp,
                action=AuditAction.MERGE_DUPLICATE,
                target_memory_id=matched_existing.id,
                related_memory_ids=[incoming.id],
                reason=f"Merged semantically similar candidate (similarity: {best_sim:.2f}) into existing memory.",
                metadata={"similarity": best_sim, "old_frequency": old_freq, "new_frequency": matched_existing.frequency}
            ))
        else:
            distinct_new.append(incoming)

    return existing_memories, distinct_new


# ============================================================================
# 3. Conflict Detection and Temporal Superseding
# ============================================================================

# Structured slot regexes to recognize entity-attribute slots for conflict checking
SLOT_PATTERNS = [
    # Employer / Workplace
    (r"\b(?:work at|joined|employed at|employer is)\s+([A-Za-z0-9_\s]+)", "employment"),
    # Residence / Location
    (r"\b(?:live in|moved to|reside in|located in)\s+([A-Za-z0-9_\s]+)", "location"),
    # Preferences: Theme / Editor / Mode
    (r"\b(?:prefer|like|use)\s+(dark|light)\s+mode\b", "ui_theme"),
    # Programming language preference
    (r"\b(?:prefer|like)\s+(python|c\+\+|rust|java|go|javascript)\b", "favorite_language"),
    # Deep learning framework
    (r"\b(?:prefer|like)\s+(pytorch|tensorflow|jax)\b", "dl_framework"),
]


def extract_fact_slot(text: str) -> Optional[Tuple[str, str]]:
    """
    Extracts (slot_type, value) from text if it asserts an identifiable entity-attribute fact.
    Example: "I work at OpenAI" -> ("employment", "openai")
    """
    text_clean = text.lower().strip()
    for pattern, slot_type in SLOT_PATTERNS:
        match = re.search(pattern, text_clean)
        if match:
            raw_value = match.group(1).strip()
            # Clean punctuation and trailing adverbs/fillers (now, currently, today, etc.)
            val = re.sub(r"[^\w\s]", "", raw_value).strip()
            val = re.sub(r"\b(now|currently|today|recently|right now|yesterday)\b", "", val).strip()
            if val:
                return slot_type, val
    return None


def detect_conflicts_and_supersede(
    all_memories: List[MemoryItem],
    audit_entries: Optional[List[AuditLogEntry]] = None,
    current_timestamp: Optional[float] = None
) -> List[MemoryItem]:
    """
    Identifies evolving information where a newer fact updates an older fact.
    Marks the older fact as SUPERSEDED rather than treating every change as an error.

    Crucial Research Guarantee:
    Preserves both the original historical observation and the superseded link,
    ensuring historical evidence is maintained for temporal queries.
    """
    if current_timestamp is None:
        current_timestamp = time.time()
    if audit_entries is None:
        audit_entries = []

    # Map slot_type -> List[MemoryItem]
    slot_buckets: Dict[str, List[MemoryItem]] = {}

    for mem in all_memories:
        slot = extract_fact_slot(mem.content)
        if slot:
            slot_type, val = slot
            mem.metadata["slot_type"] = slot_type
            mem.metadata["slot_value"] = val
            slot_buckets.setdefault(slot_type, []).append(mem)

    for slot_type, items in slot_buckets.items():
        if len(items) <= 1:
            continue

        # Sort items chronologically
        items_sorted = sorted(items, key=lambda m: m.timestamp)

        # The newest assertion represents the current ground truth
        newest = items_sorted[-1]

        for older in items_sorted[:-1]:
            # If the values differ, the newer observation supersedes the older
            if older.metadata.get("slot_value") != newest.metadata.get("slot_value"):
                if older.lifecycle_status != LifecycleStatus.SUPERSEDED:
                    older.lifecycle_status = LifecycleStatus.SUPERSEDED
                    older.metadata["superseded_by_id"] = newest.id
                    older.metadata["superseded_at"] = current_timestamp

                    # Link previous history in newest memory
                    history_list = newest.metadata.setdefault("historical_predecessors", [])
                    history_list.append({
                        "superseded_memory_id": older.id,
                        "previous_value": older.metadata.get("slot_value"),
                        "timestamp": older.timestamp,
                        "original_content": older.content
                    })

                    audit_entries.append(AuditLogEntry(
                        id=str(uuid.uuid4()),
                        timestamp=current_timestamp,
                        action=AuditAction.SUPERSEDE,
                        target_memory_id=older.id,
                        related_memory_ids=[newest.id],
                        reason=(
                            f"Slot '{slot_type}' updated from '{older.metadata.get('slot_value')}' "
                            f"to '{newest.metadata.get('slot_value')}'. Older memory marked SUPERSEDED."
                        ),
                        metadata={
                            "slot_type": slot_type,
                            "old_value": older.metadata.get("slot_value"),
                            "new_value": newest.metadata.get("slot_value"),
                            "old_timestamp": older.timestamp,
                            "new_timestamp": newest.timestamp
                        }
                    ))

    return all_memories


# ============================================================================
# 4. Episodic-to-Semantic Consolidation
# ============================================================================

def consolidate_episodic_memories(
    episodic_memories: List[MemoryItem],
    semantic_memories: List[MemoryItem],
    min_frequency_to_consolidate: int = 2,
    audit_entries: Optional[List[AuditLogEntry]] = None,
    current_timestamp: Optional[float] = None
) -> Tuple[List[MemoryItem], List[MemoryItem]]:
    """
    Consolidates repeated or reinforced episodic observations into compact, durable semantic knowledge.

    Research Principle:
    Retains original observation provenance within metadata['source_evidence'] so historical queries
    can be resolved even after consolidation.
    """
    if current_timestamp is None:
        current_timestamp = time.time()
    if audit_entries is None:
        audit_entries = []

    remaining_episodic: List[MemoryItem] = []
    new_semantic: List[MemoryItem] = list(semantic_memories)

    for ep_mem in episodic_memories:
        # Check if memory has been reinforced frequently enough or represents an enduring preference
        is_candidate_for_consolidation = (
            ep_mem.frequency >= min_frequency_to_consolidate or
            ep_mem.stability >= 0.70 or
            ep_mem.metadata.get("slot_type") is not None
        )

        if is_candidate_for_consolidation and ep_mem.lifecycle_status in (LifecycleStatus.ACTIVE, LifecycleStatus.TEMPORARY):
            # Check if an existing semantic memory already covers this
            existing_sem = next(
                (sm for sm in new_semantic if compute_text_similarity(sm.content, ep_mem.content) >= 0.75),
                None
            )

            if existing_sem:
                # Merge into existing semantic memory
                existing_sem.frequency += ep_mem.frequency
                existing_sem.stability = min(1.0, existing_sem.stability + 0.1)
                existing_sem.importance = max(existing_sem.importance, ep_mem.importance)
                evidence = existing_sem.metadata.setdefault("source_evidence", [])
                evidence.append({
                    "original_episodic_id": ep_mem.id,
                    "content": ep_mem.content,
                    "timestamp": ep_mem.timestamp,
                    "provenance": ep_mem.metadata.get("provenance", [])
                })

                audit_entries.append(AuditLogEntry(
                    id=str(uuid.uuid4()),
                    timestamp=current_timestamp,
                    action=AuditAction.CONSOLIDATE,
                    target_memory_id=existing_sem.id,
                    related_memory_ids=[ep_mem.id],
                    reason=f"Consolidated repeated episodic observation '{ep_mem.content}' into existing semantic memory.",
                    metadata={"reinforced_frequency": existing_sem.frequency}
                ))
            else:
                # Promote to new compact semantic memory
                consolidated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"semantic_cons:{ep_mem.id}"))
                consolidated_item = MemoryItem(
                    id=consolidated_id,
                    content=ep_mem.content,
                    memory_type=MemoryType.SEMANTIC,
                    source_session_id=ep_mem.source_session_id,
                    timestamp=ep_mem.timestamp,
                    importance=min(1.0, ep_mem.importance + 0.1),
                    recency=ep_mem.recency,
                    frequency=ep_mem.frequency,
                    stability=max(0.80, ep_mem.stability),
                    consistency_confidence=ep_mem.consistency_confidence,
                    utility_score=ep_mem.utility_score,
                    token_cost=ep_mem.token_cost,
                    lifecycle_status=LifecycleStatus.STABLE,
                    metadata={
                        "consolidated_from_id": ep_mem.id,
                        "source_evidence": [{
                            "original_episodic_id": ep_mem.id,
                            "content": ep_mem.content,
                            "timestamp": ep_mem.timestamp,
                            "provenance": ep_mem.metadata.get("provenance", [])
                        }]
                    }
                )
                new_semantic.append(consolidated_item)

                audit_entries.append(AuditLogEntry(
                    id=str(uuid.uuid4()),
                    timestamp=current_timestamp,
                    action=AuditAction.CONSOLIDATE,
                    target_memory_id=consolidated_item.id,
                    related_memory_ids=[ep_mem.id],
                    reason=f"Promoted reinforced episodic observation into durable semantic knowledge.",
                    metadata={"consolidated_id": consolidated_id}
                ))

            # Mark episodic observation as consolidated and superseded by semantic representation
            target_sem_id = existing_sem.id if existing_sem else consolidated_id
            ep_mem.metadata["consolidated"] = True
            ep_mem.lifecycle_status = LifecycleStatus.SUPERSEDED
            ep_mem.metadata["superseded_by_id"] = target_sem_id
            ep_mem.metadata["superseded_at"] = current_timestamp
            remaining_episodic.append(ep_mem)
        else:
            remaining_episodic.append(ep_mem)

    return remaining_episodic, new_semantic


# ============================================================================
# 5. Lifecycle Transitions (ACTIVE, STABLE, TEMPORARY, SUPERSEDED, EXPIRED)
# ============================================================================

def evaluate_lifecycle_transitions(
    memories: List[MemoryItem],
    current_timestamp: Optional[float] = None,
    temporary_ttl_seconds: float = 3600.0 * 24.0,  # 24 hours TTL for temporary context
    recency_expiration_threshold: float = 0.05,
    audit_entries: Optional[List[AuditLogEntry]] = None
) -> List[MemoryItem]:
    """
    Applies justified lifecycle state transitions across all stored memories:
    - ACTIVE -> STABLE: Reinforced (frequency >= 3, stability >= 0.75)
    - ACTIVE -> TEMPORARY: Low stability transient requests (stability <= 0.3)
    - TEMPORARY -> EXPIRED: Age exceeds temporary TTL
    - ACTIVE -> EXPIRED: Decayed recency below expiration threshold with low frequency
    """
    if current_timestamp is None:
        current_timestamp = time.time()
    if audit_entries is None:
        audit_entries = []

    for mem in memories:
        old_status = mem.lifecycle_status
        new_status = old_status

        # If already superseded, preserve superseded status
        if old_status == LifecycleStatus.SUPERSEDED:
            continue

        age = current_timestamp - mem.timestamp

        # Check temporary items for TTL expiry
        if old_status == LifecycleStatus.TEMPORARY:
            if age > temporary_ttl_seconds:
                new_status = LifecycleStatus.EXPIRED

        elif old_status == LifecycleStatus.ACTIVE:
            # Check for stabilization
            if mem.frequency >= 3 and mem.stability >= 0.70:
                new_status = LifecycleStatus.STABLE
            # Check for transient classification
            elif mem.stability <= 0.30 and mem.frequency == 1:
                new_status = LifecycleStatus.TEMPORARY
            # Check for expiration of low-utility unreinforced episodic items
            elif mem.memory_type == MemoryType.EPISODIC and mem.recency < recency_expiration_threshold and mem.frequency == 1:
                new_status = LifecycleStatus.EXPIRED

        if new_status != old_status:
            mem.lifecycle_status = new_status
            audit_entries.append(AuditLogEntry(
                id=str(uuid.uuid4()),
                timestamp=current_timestamp,
                action=AuditAction.TRANSITION_STATUS,
                target_memory_id=mem.id,
                related_memory_ids=[],
                reason=f"Lifecycle transition: {old_status.value} -> {new_status.value}.",
                metadata={"previous_status": old_status.value, "new_status": new_status.value, "age": age}
            ))

    return memories


# ============================================================================
# 6. Budget Maintenance & Eviction
# ============================================================================

def maintain_memory_budget(
    episodic_memories: List[MemoryItem],
    semantic_memories: List[MemoryItem],
    total_budget: int,
    allocation_config: AllocationConfig,
    audit_entries: Optional[List[AuditLogEntry]] = None,
    current_timestamp: Optional[float] = None
) -> Tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem], Dict[str, Any]]:
    """
    Enforces total storage budget under evolving arrivals:
    1. Evaluates allocation share limits between episodic and semantic stores.
    2. Separates active persistent memories from evicted/superseded/expired archives.
    3. If active items exceed allocated capacity, evicts lowest utility-density items first.
    4. Guarantees: total_active_tokens <= total_budget.

    Returns:
        Tuple of (selected_episodic, selected_semantic, evicted_archive, maintenance_summary)
    """
    if current_timestamp is None:
        current_timestamp = time.time()
    if audit_entries is None:
        audit_entries = []

    # Dynamic budget allocation based on active demand
    all_candidates = episodic_memories + semantic_memories
    budget_split = allocate_budget(all_candidates, allocation_config)
    episodic_budget = budget_split[MemoryType.EPISODIC]
    semantic_budget = budget_split[MemoryType.SEMANTIC]

    evicted_archive: List[MemoryItem] = []

    def _filter_and_evict_store(
        items: List[MemoryItem],
        store_budget: int,
        store_type: MemoryType
    ) -> Tuple[List[MemoryItem], int]:
        # Filter out expired or superseded items first
        active_items: List[MemoryItem] = []
        for item in items:
            if item.lifecycle_status in (LifecycleStatus.EXPIRED, LifecycleStatus.SUPERSEDED):
                evicted_archive.append(item)
                audit_entries.append(AuditLogEntry(
                    id=str(uuid.uuid4()),
                    timestamp=current_timestamp,
                    action=AuditAction.EVICT,
                    target_memory_id=item.id,
                    related_memory_ids=[],
                    reason=f"Evicted {item.lifecycle_status.value} memory from active budget pool to archive.",
                    metadata={"status": item.lifecycle_status.value, "token_cost": item.token_cost}
                ))
            else:
                active_items.append(item)

        # Select items that strictly fit store_budget
        selected, used_tokens = select_memories(active_items, store_budget)
        selected_ids = {m.id for m in selected}

        # Any active items not selected due to budget are evicted
        for item in active_items:
            if item.id not in selected_ids:
                evicted_archive.append(item)
                audit_entries.append(AuditLogEntry(
                    id=str(uuid.uuid4()),
                    timestamp=current_timestamp,
                    action=AuditAction.EVICT,
                    target_memory_id=item.id,
                    related_memory_ids=[],
                    reason=f"Evicted active memory due to {store_type.value} token budget cap ({store_budget} tokens).",
                    metadata={"utility_score": item.utility_score, "token_cost": item.token_cost}
                ))

        return selected, used_tokens

    selected_episodic, used_ep = _filter_and_evict_store(episodic_memories, episodic_budget, MemoryType.EPISODIC)
    selected_semantic, used_sem = _filter_and_evict_store(semantic_memories, semantic_budget, MemoryType.SEMANTIC)

    summary = {
        "total_budget": total_budget,
        "episodic_budget": episodic_budget,
        "semantic_budget": semantic_budget,
        "used_tokens_episodic": used_ep,
        "used_tokens_semantic": used_sem,
        "total_used_tokens": used_ep + used_sem,
        "evicted_count": len(evicted_archive),
        "active_count": len(selected_episodic) + len(selected_semantic),
    }

    return selected_episodic, selected_semantic, evicted_archive, summary


# ============================================================================
# 7. Complete Phase 2 Memory Manager
# ============================================================================

class AdaptiveMemoryManager:
    """
    Stateful memory manager handling the full Phase 2 lifecycle:
    - Multi-session interaction intake
    - Semantic deduplication
    - Conflict superseding & temporal provenance preservation
    - Consolidation into durable semantic knowledge
    - Status evaluation & budget maintenance
    - Full audit log persistence
    """
    def __init__(
        self,
        total_budget: int = 1000,
        scoring_weights: Optional[ScoringWeights] = None,
        allocation_config: Optional[AllocationConfig] = None,
        persistence_path: Optional[str] = None,
        admission_gate: Optional[Any] = None,
        candidate_extractor: Optional[Any] = None,
        tokenizer_fn: Optional[Any] = None,
        enable_consolidation: bool = True
    ):
        self.total_budget = total_budget
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.allocation_config = allocation_config or AllocationConfig(total_budget=total_budget)
        self.persistence_path = persistence_path
        self.admission_gate = admission_gate
        self.candidate_extractor = candidate_extractor
        self.tokenizer_fn = tokenizer_fn
        self.enable_consolidation = enable_consolidation

        self.episodic_store: List[MemoryItem] = []
        self.semantic_store: List[MemoryItem] = []
        self.archive_store: List[MemoryItem] = []
        self.audit_log: List[AuditLogEntry] = []
        self.rejected_candidates: List[Tuple[MemoryItem, Any, str]] = []

    def process_session(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        current_timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Process a conversation session through the Phase 2 pipeline."""
        if current_timestamp is None:
            current_timestamp = time.time()

        # 1. Candidate Extraction
        if self.candidate_extractor is not None:
            new_candidates = self.candidate_extractor.extract_candidates(
                conversation_history=conversation_history,
                session_id=session_id,
                current_timestamp=current_timestamp,
                tokenizer_fn=self.tokenizer_fn
            )
        else:
            new_candidates = extract_memories(
                conversation_history=conversation_history,
                session_id=session_id,
                current_timestamp=current_timestamp,
                tokenizer_fn=self.tokenizer_fn
            )

        # 2. Score candidates
        scored_new = score_memories(
            memories=new_candidates,
            weights=self.scoring_weights,
            current_timestamp=current_timestamp
        )

        # 2b. Candidate Admission Gating (if admission_gate is provided)
        session_rejected: List[Tuple[MemoryItem, Any, str]] = []
        if self.admission_gate is not None:
            if hasattr(self.admission_gate, "filter_quality"):
                scored_new, session_rejected = self.admission_gate.filter_quality(
                    scored_new,
                    audit_entries=self.audit_log,
                    current_timestamp=current_timestamp
                )
            elif hasattr(self.admission_gate, "evaluate_candidates"):
                eval_res = self.admission_gate.evaluate_candidates(
                    scored_new,
                    existing_memories=self.episodic_store + self.semantic_store,
                    current_timestamp=current_timestamp
                )
                scored_new = eval_res.admitted_candidates
                session_rejected = eval_res.rejected_candidates
                if eval_res.audit_entries:
                    self.audit_log.extend(eval_res.audit_entries)
            self.rejected_candidates.extend(session_rejected)

        # Partition new candidates
        new_ep = [m for m in scored_new if m.memory_type == MemoryType.EPISODIC]
        new_sem = [m for m in scored_new if m.memory_type == MemoryType.SEMANTIC]

        # 3. Semantic Deduplication
        self.episodic_store, distinct_ep = deduplicate_semantically(
            incoming_memories=new_ep,
            existing_memories=self.episodic_store,
            audit_entries=self.audit_log,
            current_timestamp=current_timestamp
        )
        self.semantic_store, distinct_sem = deduplicate_semantically(
            incoming_memories=new_sem,
            existing_memories=self.semantic_store,
            audit_entries=self.audit_log,
            current_timestamp=current_timestamp
        )

        self.episodic_store.extend(distinct_ep)
        self.semantic_store.extend(distinct_sem)

        # 4. Conflict detection & temporal superseding
        all_active = self.episodic_store + self.semantic_store
        detect_conflicts_and_supersede(
            all_memories=all_active,
            audit_entries=self.audit_log,
            current_timestamp=current_timestamp
        )

        # 5. Consolidation of repeated episodic observations into semantic store
        if self.enable_consolidation:
            self.episodic_store, self.semantic_store = consolidate_episodic_memories(
                episodic_memories=self.episodic_store,
                semantic_memories=self.semantic_store,
                audit_entries=self.audit_log,
                current_timestamp=current_timestamp
            )

        # 6. Re-score all memories with updated recency and frequencies
        self.episodic_store = score_memories(self.episodic_store, self.scoring_weights, current_timestamp)
        self.semantic_store = score_memories(self.semantic_store, self.scoring_weights, current_timestamp)

        # 7. Evaluate lifecycle status transitions
        self.episodic_store = evaluate_lifecycle_transitions(
            self.episodic_store, current_timestamp=current_timestamp, audit_entries=self.audit_log
        )
        self.semantic_store = evaluate_lifecycle_transitions(
            self.semantic_store, current_timestamp=current_timestamp, audit_entries=self.audit_log
        )

        # 8. Maintain budget & evict
        (
            self.episodic_store,
            self.semantic_store,
            new_evicted,
            maintenance_summary
        ) = maintain_memory_budget(
            episodic_memories=self.episodic_store,
            semantic_memories=self.semantic_store,
            total_budget=self.total_budget,
            allocation_config=self.allocation_config,
            audit_entries=self.audit_log,
            current_timestamp=current_timestamp
        )
        self.archive_store.extend(new_evicted)

        # 9. Persist if configured
        if self.persistence_path:
            self.save_state(self.persistence_path, maintenance_summary)

        return {
            "session_id": session_id,
            "maintenance_summary": maintenance_summary,
            "episodic_count": len(self.episodic_store),
            "semantic_count": len(self.semantic_store),
            "archive_count": len(self.archive_store),
            "audit_log_count": len(self.audit_log),
            "rejected_count": len(session_rejected),
            "rejected_candidates": session_rejected,
        }

    def save_state(self, filepath: str, summary: Optional[Dict[str, Any]] = None) -> None:
        """Persist memory state including active stores, archive, and audit log."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "version": "phase_2.0",
            "saved_at": time.time(),
            "total_budget": self.total_budget,
            "summary": summary or {},
            "episodic_store": [m.to_dict() for m in self.episodic_store],
            "semantic_store": [m.to_dict() for m in self.semantic_store],
            "archive_store": [m.to_dict() for m in self.archive_store],
            "audit_log": [e.to_dict() for e in self.audit_log],
            "rejected_candidates": [
                (c.to_dict(), getattr(d, "value", str(d)), r) for c, d, r in self.rejected_candidates
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_state(self, filepath: str) -> None:
        """Load persisted Phase 2 state."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.total_budget = data.get("total_budget", self.total_budget)
        self.episodic_store = [MemoryItem.from_dict(d) for d in data.get("episodic_store", [])]
        self.semantic_store = [MemoryItem.from_dict(d) for d in data.get("semantic_store", [])]
        self.archive_store = [MemoryItem.from_dict(d) for d in data.get("archive_store", [])]
        self.audit_log = [AuditLogEntry.from_dict(d) for d in data.get("audit_log", [])]


def run_phase2_pipeline(
    conversation_history: List[Dict[str, Any]],
    session_id: str = "session_0",
    total_budget: int = 1000,
    scoring_weights: Optional[ScoringWeights] = None,
    allocation_config: Optional[AllocationConfig] = None,
    persistence_path: Optional[str] = None,
    current_timestamp: Optional[float] = None,
    admission_gate: Optional[Any] = None,
    candidate_extractor: Optional[Any] = None
) -> Dict[str, Any]:
    """Clean procedural runner for Phase 2 pipeline."""
    manager = AdaptiveMemoryManager(
        total_budget=total_budget,
        scoring_weights=scoring_weights,
        allocation_config=allocation_config,
        persistence_path=persistence_path,
        admission_gate=admission_gate,
        candidate_extractor=candidate_extractor
    )
    res = manager.process_session(
        conversation_history=conversation_history,
        session_id=session_id,
        current_timestamp=current_timestamp
    )
    return {
        "manager": manager,
        "session_result": res,
        "selected_episodic": [m.to_dict() for m in manager.episodic_store],
        "selected_semantic": [m.to_dict() for m in manager.semantic_store],
        "archive_store": [m.to_dict() for m in manager.archive_store],
        "audit_log": [e.to_dict() for e in manager.audit_log],
    }
