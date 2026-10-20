"""
Adaptive Long-Term Memory for Language Models
Phase 3 (Hardened) & Phase 4: Query Processing, Temporal Reasoning & Budgeted Retrieval

Pipeline:
User Query
  -> Query Intent & Temporal Point-in-Time Analysis
  -> Stopword-Filtered Relevance & Slot Affinity Scoring
  -> Margin-Based Relevance Pruning (Distractor Exclusion)
  -> Prompt Budget Accounting (System + Query + Chat Template + Generation Allowance)
  -> Context Assembly with Strict Budget & Provenance Tags
  -> LLM Generation via Model Adapter
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
from model_adapter import BaseLLMAdapter, FakeLLMAdapter, heuristic_bpe_token_count


# Common English grammatical stopwords filtered during keyword relevance
COMMON_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "being", "below", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "right", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves", "help", "please"
}


# ============================================================================
# 1. Retrieval Configuration & Temporal Query Analysis
# ============================================================================

@dataclass
class RetrievalWeights:
    """Configurable weights for multi-factor memory retrieval ranking."""
    w_rel: float = 0.55    # Relevance to user query (content keywords & slot affinity)
    w_util: float = 0.20   # Intrinsic memory utility score U(m)
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
        r"\blast time\b", r"\byesterday\b", r"\bat timestamp\b", r"\bat time\b"
    ]
    return any(re.search(pat, q_lower) for pat in historical_keywords)


def extract_query_point_in_time(query: str) -> Optional[float]:
    """
    Extracts explicit timestamp reference if present in query.
    Examples: 'Where did I work at timestamp 1500.0?', 'at time 1200'
    """
    match = re.search(r"\b(?:at timestamp|at time|timestamp)\s+([0-9]+(?:\.[0-9]+)?)\b", query.lower())
    if match:
        return float(match.group(1))
    return None


def is_memory_valid_at_timestamp(memory: MemoryItem, query_timestamp: float) -> bool:
    """
    Evaluates whether a memory was active and valid at a specific historical point in time.
    """
    # Not yet created at query_timestamp
    if memory.timestamp > query_timestamp:
        return False

    # Check if superseded before query_timestamp
    if memory.lifecycle_status == LifecycleStatus.SUPERSEDED:
        superseded_at = memory.metadata.get("superseded_at")
        if superseded_at is not None and superseded_at <= query_timestamp:
            return False

    return True


# ============================================================================
# 2. Hardened Relevance Scoring & Distractor Pruning
# ============================================================================

def compute_query_relevance(query: str, memory: MemoryItem) -> float:
    """
    Calculates query-memory relevance combining:
    1. Stopword-filtered content keyword overlap (prevents 'now', 'do', 'i' matching unrelated tasks)
    2. Fact slot affinity (e.g. 'work'/'employer' aligning with 'employment' slot)
    3. Lexical text similarity
    4. Penalty for mismatched transient tasks when answering durable fact queries
    """
    q_lower = query.lower()
    mem_lower = memory.content.lower()

    # Extract non-stopword content words
    q_tokens = set(re.findall(r"\b[a-z0-9_]+\b", q_lower))
    m_tokens = set(re.findall(r"\b[a-z0-9_]+\b", mem_lower))

    q_content_words = q_tokens - COMMON_STOPWORDS
    m_content_words = m_tokens - COMMON_STOPWORDS

    if not q_content_words:
        q_content_words = q_tokens  # Fallback if query only contained stopwords

    # Keyword intersection
    intersection = q_content_words.intersection(m_content_words)
    content_overlap = len(intersection) / max(1, len(q_content_words))

    # Slot matching & domain affinity
    mem_slot = memory.metadata.get("slot_type")
    slot_boost = 0.0
    slot_keywords = {
        "employment": ["work", "job", "company", "employer", "employed", "office", "deepmind", "openai"],
        "location": ["live", "located", "city", "reside", "where", "bangalore", "london"],
        "ui_theme": ["theme", "mode", "dark", "light", "ui", "editor"],
        "favorite_language": ["language", "code", "programming", "python", "rust", "c++", "java"],
        "dl_framework": ["framework", "pytorch", "tensorflow", "jax", "cuda", "deep learning"],
    }

    query_matched_slots = []
    for slot_name, kws in slot_keywords.items():
        if any(kw in q_lower for kw in kws):
            query_matched_slots.append(slot_name)

    if mem_slot and mem_slot in query_matched_slots:
        slot_boost = 0.40
    elif query_matched_slots and (mem_slot is None or mem_slot not in query_matched_slots):
        # Query specifically targeted a recognized slot (e.g. employment),
        # but this memory is an unrelated slot or transient episodic task
        if memory.lifecycle_status == LifecycleStatus.TEMPORARY:
            slot_boost = -0.30

    base_sim = compute_text_similarity(query, memory.content)

    relevance = 0.50 * content_overlap + 0.20 * base_sim + slot_boost
    return round(max(0.0, min(1.0, relevance)), 4)


def retrieve_memories(
    query: str,
    episodic_store: List[MemoryItem],
    semantic_store: List[MemoryItem],
    archive_store: Optional[List[MemoryItem]] = None,
    top_k: int = 5,
    weights: Optional[RetrievalWeights] = None,
    min_relevance_threshold: float = 0.20,
    margin_pruning_ratio: float = 0.40
) -> List[Tuple[MemoryItem, float]]:
    """
    Hardened dual-store memory retrieval:
    - Current queries: strictly exclude SUPERSEDED and EXPIRED memories.
    - Historical queries: surface SUPERSEDED memories and predecessor links.
    - Point-in-time queries: filter by historical validity at target timestamp.
    - Margin pruning: excludes unrelated distractors (e.g. temporary tasks) when
      far superior evidence exists, ensuring low-relevance items don't leak into context.
    """
    if weights is None:
        weights = RetrievalWeights()
    if archive_store is None:
        archive_store = []

    is_hist = is_historical_query(query)
    point_in_time = extract_query_point_in_time(query)

    pool = list(episodic_store) + list(semantic_store)
    if is_hist or point_in_time is not None:
        pool.extend(archive_store)

    scored_candidates: List[Tuple[MemoryItem, float]] = []

    for mem in pool:
        status = mem.lifecycle_status

        # Temporal validity enforcement:
        if point_in_time is not None:
            if not is_memory_valid_at_timestamp(mem, point_in_time):
                continue
        elif not is_hist:
            # Current query: NEVER treat superseded or expired items as current facts
            if status in (LifecycleStatus.SUPERSEDED, LifecycleStatus.EXPIRED):
                continue
        else:
            # Historical query: exclude expired garbage
            if status == LifecycleStatus.EXPIRED:
                continue

        relevance = compute_query_relevance(query, mem)
        if relevance < min_relevance_threshold:
            continue

        score = (
            weights.w_rel * relevance +
            weights.w_util * mem.utility_score +
            weights.w_rec * mem.recency +
            weights.w_stab * mem.stability
        )

        if is_hist and (status == LifecycleStatus.SUPERSEDED or mem.metadata.get("historical_predecessors")):
            score += 0.25

        scored_candidates.append((mem, round(score, 4)))

    if not scored_candidates:
        return []

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # Dynamic margin pruning: prune items whose score is significantly below the top score
    top_score = scored_candidates[0][1]
    filtered_results = [
        (mem, score) for mem, score in scored_candidates
        if score >= top_score * margin_pruning_ratio
    ]

    return filtered_results[:top_k]


# ============================================================================
# 3. Exact Token Accounting & Context Assembly
# ============================================================================

@dataclass
class PromptBudgetTracker:
    """
    Tracks and enforces model context window bounds.
    Decoupled from persistent memory storage budget.
    """
    model_context_window: int = 2048       # Total model n_ctx
    generation_allowance: int = 256       # Output generation token budget
    system_prompt_tokens: int = 0
    query_tokens: int = 0
    chat_template_overhead: int = 16      # Formatting tokens (<|im_start|>, roles, newlines)
    max_memory_context_tokens: int = 512   # User-specified max budget for memory context

    def calculate_allowed_memory_budget(self) -> int:
        """
        Guarantees:
        system_tokens + query_tokens + template_overhead + memory_context_tokens + generation_allowance <= model_context_window
        """
        available = self.model_context_window - (
            self.system_prompt_tokens + self.query_tokens + self.chat_template_overhead + self.generation_allowance
        )
        return max(0, min(self.max_memory_context_tokens, available))


@dataclass
class ContextAssemblyResult:
    formatted_context: str
    selected_memories: List[MemoryItem]
    context_token_cost: int
    context_budget: int
    system_prompt: str
    tracker: PromptBudgetTracker


def assemble_memory_context(
    query: str,
    retrieved_memories: List[Tuple[MemoryItem, float]],
    max_context_tokens: int = 512,
    model_context_window: int = 2048,
    generation_allowance: int = 256,
    tokenizer_fn: Optional[Callable[[str], int]] = None,
    include_provenance: bool = True
) -> ContextAssemblyResult:
    """
    Assembles memory context with exact token accounting:
    1. Computes system prompt and query tokens.
    2. Calculates available budget for memory context without exceeding model context window.
    3. Formats memories with explicit lifecycle tags ([CURRENT], [HISTORICAL / SUPERSEDED], [TEMPORARY]).
    4. Deduplicates items to prevent redundant tokens.
    """
    if tokenizer_fn is None:
        tokenizer_fn = heuristic_bpe_token_count

    system_prompt = (
        "You are an intelligent assistant with access to the user's persistent long-term memory.\n"
        "Use the retrieved memories below to answer the user's question accurately.\n"
        "IMPORTANT TEMPORARY & TEMPORAL INSTRUCTIONS:\n"
        "- Facts marked [CURRENT] represent the user's current situation or preferences.\n"
        "- Facts marked [HISTORICAL / SUPERSEDED] represent past information that has changed.\n"
        "- When the user asks about their current state, rely only on [CURRENT] facts.\n"
        "- When the user asks about past or previous states, explain what changed based on [HISTORICAL] facts."
    )

    tracker = PromptBudgetTracker(
        model_context_window=model_context_window,
        generation_allowance=generation_allowance,
        system_prompt_tokens=tokenizer_fn(system_prompt),
        query_tokens=tokenizer_fn(query),
        max_memory_context_tokens=max_context_tokens
    )

    allowed_memory_tokens = tracker.calculate_allowed_memory_budget()

    if allowed_memory_tokens <= 0:
        return ContextAssemblyResult(
            formatted_context="",
            selected_memories=[],
            context_token_cost=0,
            context_budget=0,
            system_prompt=system_prompt,
            tracker=tracker
        )

    selected: List[Any] = []
    seen_contents: Set[str] = set()
    formatted_lines: List[str] = []

    for item_entry in retrieved_memories:
        if isinstance(item_entry, tuple) and len(item_entry) == 2:
            mem, rank_score = item_entry
        else:
            mem, rank_score = item_entry, 1.0

        content = getattr(mem, "content", None) or (mem.get("content") if isinstance(mem, dict) else str(mem))
        content = content.strip() if content else ""
        if not content:
            continue

        norm_content = " ".join(content.lower().split())
        if norm_content in seen_contents:
            continue

        # Strategy-neutral status handling
        raw_status = getattr(mem, "lifecycle_status", None) or (mem.get("lifecycle_status") if isinstance(mem, dict) else None)
        if raw_status == LifecycleStatus.SUPERSEDED or str(raw_status).lower() in ("superseded", "historical"):
            status_tag = "[HISTORICAL / SUPERSEDED]"
            extra_info = " (Replaced by newer update)"
        elif raw_status in (LifecycleStatus.ACTIVE, LifecycleStatus.STABLE) or str(raw_status).lower() in ("active", "stable", "current"):
            status_tag = "[CURRENT]"
            metadata = getattr(mem, "metadata", {}) or (mem.get("metadata", {}) if isinstance(mem, dict) else {})
            preds = metadata.get("historical_predecessors", []) if isinstance(metadata, dict) else []
            extra_info = f" (Formerly: {preds[-1].get('previous_value', '')})" if preds else ""
        elif raw_status is not None:
            val_str = getattr(raw_status, "value", str(raw_status))
            status_tag = f"[{val_str.upper()}]"
            extra_info = ""
        else:
            status_tag = "[CURRENT]"
            extra_info = ""

        # Strategy-neutral type handling
        raw_type = getattr(mem, "memory_type", None) or (mem.get("memory_type") if isinstance(mem, dict) else None)
        if raw_type is not None:
            type_tag = "Episodic" if (raw_type == MemoryType.EPISODIC or str(raw_type).lower() == "episodic") else "Semantic"
        else:
            type_tag = "Evidence"

        mem_id = getattr(mem, "id", None) or (mem.get("id") if isinstance(mem, dict) else None)
        if include_provenance and mem_id:
            line = f"- {status_tag} [{type_tag} | ID: {str(mem_id)[:8]}] {content}{extra_info}"
        else:
            line = f"- {status_tag} {content}{extra_info}"

        tentative_context = "\n".join(formatted_lines + [line])
        tentative_cost = tokenizer_fn(tentative_context)

        if tentative_cost <= allowed_memory_tokens:
            formatted_lines.append(line)
            selected.append(mem)
            seen_contents.add(norm_content)

    formatted_context = "\n".join(formatted_lines) if formatted_lines else "No relevant memories found."
    final_token_cost = tokenizer_fn(formatted_context) if formatted_lines else 0

    return ContextAssemblyResult(
        formatted_context=formatted_context,
        selected_memories=selected,
        context_token_cost=final_token_cost,
        context_budget=allowed_memory_tokens,
        system_prompt=system_prompt,
        tracker=tracker
    )


# ============================================================================
# 4. End-to-End Pipeline Execution
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
    current_timestamp: Optional[float] = None,
    admission_gate: Optional[Any] = None,
    candidate_extractor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes the hardened Phase 3 pipeline with native tokenizer support.
    """
    if current_timestamp is None:
        current_timestamp = time.time()

    if existing_manager is None:
        manager = AdaptiveMemoryManager(
            total_budget=storage_budget,
            scoring_weights=scoring_weights,
            allocation_config=allocation_config,
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor
        )
    else:
        manager = existing_manager

    if conversation_history:
        manager.process_session(
            conversation_history=conversation_history,
            session_id=session_id,
            current_timestamp=current_timestamp
        )

    # Retrieval
    retrieved = retrieve_memories(
        query=user_query,
        episodic_store=manager.episodic_store,
        semantic_store=manager.semantic_store,
        archive_store=manager.archive_store,
        top_k=5
    )

    # Context assembly using model's tokenizer function
    tokenizer_fn = getattr(llm_adapter, "count_tokens", heuristic_bpe_token_count)
    context_res = assemble_memory_context(
        query=user_query,
        retrieved_memories=retrieved,
        max_context_tokens=max_context_tokens,
        tokenizer_fn=tokenizer_fn
    )

    messages = [
        {"role": "system", "content": f"{context_res.system_prompt}\n\nRELEVANT MEMORIES:\n{context_res.formatted_context}"},
        {"role": "user", "content": user_query}
    ]

    response_text = llm_adapter.generate_chat(
        messages=messages,
        max_tokens=256,
        temperature=0.7
    )

    return {
        "user_query": user_query,
        "response": response_text,
        "is_historical_query": is_historical_query(user_query),
        "point_in_time": extract_query_point_in_time(user_query),
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
