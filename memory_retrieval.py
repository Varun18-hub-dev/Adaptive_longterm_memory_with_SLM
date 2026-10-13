"""
Adaptive Long-Term Memory for Language Models
Phase 3: Query Processing, Budgeted Retrieval & Context Assembly

Pipeline:
User Query
  -> Query Intent & Temporal Validity Analysis
  -> Dual-Store Retrieval (Episodic + Semantic + Archive for Historical)
  -> Multi-Factor Relevance & Temporal Validity Ranking
  -> Prompt-Context Assembly (Strict Context Budget & Deduplication)
  -> LLM Generation via Model Adapter
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from memory_core import (
    AllocationConfig,
    LifecycleStatus,
    MemoryItem,
    MemoryType,
    ScoringWeights,
    estimate_token_cost,
)
from memory_lifecycle import (
    AdaptiveMemoryManager,
    compute_text_similarity,
    extract_fact_slot,
)
from model_adapter import BaseLLMAdapter, FakeLLMAdapter


# ============================================================================
# 1. Retrieval Configuration & Intent Analysis
# ============================================================================

@dataclass
class RetrievalWeights:
    """Configurable weights for multi-factor memory retrieval ranking."""
    w_rel: float = 0.50    # Relevance to user query
    w_util: float = 0.25   # Intrinsic memory utility score U(m)
    w_rec: float = 0.15    # Recency / temporal freshness R(m)
    w_stab: float = 0.10   # Temporal stability / durability T(m)

    def __post_init__(self):
        total = self.w_rel + self.w_util + self.w_rec + self.w_stab
        if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"Retrieval weights must sum to 1.0; got {total:.4f}")


def is_historical_query(query: str) -> bool:
    """
    Detects whether the query specifically inquires about historical, past, or superseded facts
    (e.g., 'What was my previous job?', 'Where did I work before?', 'What was my earlier preference?').
    """
    q_lower = query.lower()
    historical_keywords = [
        r"\bprevious\b", r"\bpreviously\b", r"\bbefore\b", r"\bearlier\b",
        r"\bpast\b", r"\bformer\b", r"\bformerly\b", r"\bprior\b",
        r"\bused to\b", r"\bold\b", r"\binitial\b", r"\bfirst\b",
        r"\blast time\b", r"\byesterday\b"
    ]
    return any(re.search(pat, q_lower) for pat in historical_keywords)


def compute_query_relevance(query: str, memory: MemoryItem) -> float:
    """
    Calculates query-memory relevance combining:
    1. Lexical/semantic text similarity
    2. Fact slot matching (e.g. 'work'/'employer' aligning with 'employment' slot)
    3. Content keyword overlap
    """
    base_sim = compute_text_similarity(query, memory.content)

    q_lower = query.lower()
    mem_slot = memory.metadata.get("slot_type")

    slot_boost = 0.0
    if mem_slot:
        slot_keywords = {
            "employment": ["work", "job", "company", "employer", "employed", "office"],
            "location": ["live", "located", "city", "reside", "where"],
            "ui_theme": ["theme", "mode", "dark", "light", "ui", "editor"],
            "favorite_language": ["language", "code", "programming", "python", "c++", "rust", "java"],
            "dl_framework": ["framework", "pytorch", "tensorflow", "jax", "deep learning", "cuda"],
        }
        keywords = slot_keywords.get(mem_slot, [])
        if any(kw in q_lower for kw in keywords):
            slot_boost = 0.35

    # Keyword intersection
    q_words = set(re.findall(r"\w+", q_lower))
    m_words = set(re.findall(r"\w+", memory.content.lower()))
    overlap_ratio = len(q_words.intersection(m_words)) / max(1, len(q_words))

    relevance = min(1.0, 0.5 * base_sim + 0.3 * overlap_ratio + slot_boost)
    return round(relevance, 4)


# ============================================================================
# 2. Dual-Store Memory Retrieval
# ============================================================================

def retrieve_memories(
    query: str,
    episodic_store: List[MemoryItem],
    semantic_store: List[MemoryItem],
    archive_store: Optional[List[MemoryItem]] = None,
    top_k: int = 5,
    weights: Optional[RetrievalWeights] = None,
    min_relevance_threshold: float = 0.15
) -> List[Tuple[MemoryItem, float]]:
    """
    Retrieves and ranks relevant memories from episodic and semantic stores (and archive if historical).

    Core Temporal Invariant:
    - Current queries: strictly exclude or penalize SUPERSEDED and EXPIRED memories;
      never treat superseded facts as current facts.
    - Historical queries: include and surface SUPERSEDED memories and historical predecessor chains.

    Returns:
        List of tuples: (MemoryItem, ranking_score) sorted descending by score.
    """
    if weights is None:
        weights = RetrievalWeights()
    if archive_store is None:
        archive_store = []

    is_hist = is_historical_query(query)

    # Candidate pool: active stores + archive if historical
    pool = list(episodic_store) + list(semantic_store)
    if is_hist:
        pool.extend(archive_store)

    scored_candidates: List[Tuple[MemoryItem, float]] = []

    for mem in pool:
        status = mem.lifecycle_status

        # Enforce temporal validity:
        if not is_hist:
            # Current query: NEVER treat superseded or expired items as current facts
            if status in (LifecycleStatus.SUPERSEDED, LifecycleStatus.EXPIRED):
                continue
        else:
            # Historical query: prioritize superseded items or items with predecessor history
            if status == LifecycleStatus.EXPIRED:
                continue

        relevance = compute_query_relevance(query, mem)
        if relevance < min_relevance_threshold:
            continue

        # Multi-factor score
        score = (
            weights.w_rel * relevance +
            weights.w_util * mem.utility_score +
            weights.w_rec * mem.recency +
            weights.w_stab * mem.stability
        )

        # Historical bonus when appropriate
        if is_hist:
            if status == LifecycleStatus.SUPERSEDED or mem.metadata.get("historical_predecessors"):
                score += 0.25

        scored_candidates.append((mem, round(score, 4)))

    # Sort descending by ranking score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[:top_k]


# ============================================================================
# 3. Prompt-Context Assembly
# ============================================================================

@dataclass
class ContextAssemblyResult:
    """Assembled prompt context with provenance and budget tracking."""
    formatted_context: str
    selected_memories: List[MemoryItem]
    context_token_cost: int
    context_budget: int
    system_prompt: str


def assemble_memory_context(
    query: str,
    retrieved_memories: List[Tuple[MemoryItem, float]],
    max_context_tokens: int = 512,
    include_provenance: bool = True
) -> ContextAssemblyResult:
    """
    Assembles long-term memory context for the LLM prompt under a strict prompt-context budget.
    Deduplicates context items, formats clear lifecycle provenance, and guarantees
    context_tokens <= max_context_tokens.

    Note: The prompt-context budget is strictly separate from the persistent storage budget.
    """
    if max_context_tokens <= 0:
        return ContextAssemblyResult(
            formatted_context="",
            selected_memories=[],
            context_token_cost=0,
            context_budget=max_context_tokens,
            system_prompt=""
        )

    system_prompt = (
        "You are an intelligent assistant with access to the user's persistent long-term memory.\n"
        "Use the retrieved memories below to answer the user's question accurately.\n"
        "IMPORTANT TEMPORARY & TEMPORAL INSTRUCTIONS:\n"
        "- Facts marked [CURRENT] represent the user's current situation or preferences.\n"
        "- Facts marked [HISTORICAL / SUPERSEDED] represent past information that has changed.\n"
        "- When the user asks about their current state, rely only on [CURRENT] facts.\n"
        "- When the user asks about past or previous states, explain what changed based on [HISTORICAL] facts."
    )

    base_overhead = estimate_token_cost(system_prompt) + 20
    available_budget = max(0, max_context_tokens - base_overhead)

    selected: List[MemoryItem] = []
    seen_contents: Set[str] = set()
    used_tokens = 0
    formatted_lines: List[str] = []

    for mem, rank_score in retrieved_memories:
        # Deduplication check
        norm_content = " ".join(mem.content.lower().split())
        if norm_content in seen_contents:
            continue

        # Format line with explicit lifecycle provenance
        if mem.lifecycle_status == LifecycleStatus.SUPERSEDED:
            status_tag = "[HISTORICAL / SUPERSEDED]"
            superseded_by = mem.metadata.get("superseded_by_id", "")
            extra_info = f" (Replaced by newer update)"
        elif mem.lifecycle_status in (LifecycleStatus.ACTIVE, LifecycleStatus.STABLE):
            status_tag = "[CURRENT]"
            preds = mem.metadata.get("historical_predecessors", [])
            if preds:
                prev_val = preds[-1].get("previous_value", "")
                extra_info = f" (Formerly: {prev_val})"
            else:
                extra_info = ""
        else:
            status_tag = f"[{mem.lifecycle_status.value.upper()}]"
            extra_info = ""

        type_tag = "Episodic" if mem.memory_type == MemoryType.EPISODIC else "Semantic"

        if include_provenance:
            line = f"- {status_tag} [{type_tag} | ID: {mem.id[:8]}] {mem.content}{extra_info}"
        else:
            line = f"- {status_tag} {mem.content}{extra_info}"

        line_cost = estimate_token_cost(line)

        # Strict budget enforcement
        if used_tokens + line_cost <= available_budget:
            formatted_lines.append(line)
            selected.append(mem)
            seen_contents.add(norm_content)
            used_tokens += line_cost

    formatted_context = "\n".join(formatted_lines) if formatted_lines else "No relevant memories found."
    total_token_cost = estimate_token_cost(formatted_context)

    return ContextAssemblyResult(
        formatted_context=formatted_context,
        selected_memories=selected,
        context_token_cost=total_token_cost,
        context_budget=max_context_tokens,
        system_prompt=system_prompt
    )


# ============================================================================
# 4. End-to-End Pipeline Runner
# ============================================================================

def run_phase3_pipeline(
    conversation_history: List[Dict[str, Any]],
    user_query: str,
    llm_adapter: BaseLLMAdapter,
    session_id: str = "session_01",
    storage_budget: int = 1000,
    max_context_tokens: int = 512,
    scoring_weights: Optional[ScoringWeights] = None,
    allocation_config: Optional[AllocationConfig] = None,
    existing_manager: Optional[AdaptiveMemoryManager] = None,
    current_timestamp: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end Phase 3 pipeline:
    1. Memory Extraction, Deduplication, Consolidation, Lifecycle Management (respecting storage_budget).
    2. Query Analysis & Dual-Store Retrieval.
    3. Context Assembly with Provenance & Temporal Labels (respecting max_context_tokens).
    4. Qwen / LLM Generation via Model Adapter.

    Storage budget and response-context budget are strictly separate and enforced.
    """
    if current_timestamp is None:
        current_timestamp = time.time()

    # 1. Memory Management & Storage
    if existing_manager is None:
        manager = AdaptiveMemoryManager(
            total_budget=storage_budget,
            scoring_weights=scoring_weights,
            allocation_config=allocation_config
        )
    else:
        manager = existing_manager

    session_res = manager.process_session(
        conversation_history=conversation_history,
        session_id=session_id,
        current_timestamp=current_timestamp
    )

    # 2. Retrieval
    retrieved = retrieve_memories(
        query=user_query,
        episodic_store=manager.episodic_store,
        semantic_store=manager.semantic_store,
        archive_store=manager.archive_store,
        top_k=5
    )

    # 3. Context Assembly under prompt-context budget
    context_res = assemble_memory_context(
        query=user_query,
        retrieved_memories=retrieved,
        max_context_tokens=max_context_tokens
    )

    # 4. Construct prompt for LLM
    messages = [
        {"role": "system", "content": f"{context_res.system_prompt}\n\nRELEVANT MEMORIES:\n{context_res.formatted_context}"},
        {"role": "user", "content": user_query}
    ]

    # 5. LLM Generation
    response_text = llm_adapter.generate_chat(
        messages=messages,
        max_tokens=256,
        temperature=0.7
    )

    return {
        "user_query": user_query,
        "response": response_text,
        "is_historical_query": is_historical_query(user_query),
        "retrieved_count": len(retrieved),
        "context_result": {
            "formatted_context": context_res.formatted_context,
            "selected_memories_count": len(context_res.selected_memories),
            "context_token_cost": context_res.context_token_cost,
            "max_context_tokens": context_res.context_budget,
        },
        "storage_status": {
            "total_storage_budget": manager.total_budget,
            "active_episodic_tokens": sum(m.token_cost for m in manager.episodic_store),
            "active_semantic_tokens": sum(m.token_cost for m in manager.semantic_store),
            "archive_count": len(manager.archive_store),
        },
        "manager": manager
    }
