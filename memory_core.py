"""
Adaptive Long-Term Memory for Language Models
Phase 1: Dynamic Memory-Budget Allocation under Evolving Information

Pipeline:
Conversation history
  -> Memory candidate extraction
  -> Utility scoring
  -> Adaptive budget allocation
  -> Episodic and semantic memory stores
  -> Persisted memory state
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# 1. Memory Schema and Enumerations
# ============================================================================

class MemoryType(str, Enum):
    """Classification of memory into episodic (contextual/events) vs semantic (durable knowledge)."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class LifecycleStatus(str, Enum):
    """Lifecycle state tracking validity and status of stored memories."""
    ACTIVE = "active"          # Currently valid and applicable
    STABLE = "stable"          # High-confidence, durable information
    TEMPORARY = "temporary"    # Transient context (e.g., active task)
    SUPERSEDED = "superseded"  # Replaced by newer, conflicting evidence
    EXPIRED = "expired"        # Outdated or past validity horizon


@dataclass
class MemoryItem:
    """Consistent memory representation across all pipeline phases."""
    id: str
    content: str
    memory_type: MemoryType
    source_session_id: str
    timestamp: float
    importance: float = 0.5            # I in [0, 1]: intrinsic significance
    recency: float = 1.0               # R in [0, 1]: temporal freshness
    frequency: int = 1                 # F >= 1: occurrence count
    stability: float = 0.5             # T in [0, 1]: temporal invariance/durability
    consistency_confidence: float = 1.0# C in [0, 1]: reliability/conflict-free confidence
    utility_score: float = 0.0         # U(m) in [0, 1]: aggregated priority score
    token_cost: int = 0                # Token footprint for budget management
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        data["lifecycle_status"] = self.lifecycle_status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryItem:
        data_copy = dict(data)
        data_copy["memory_type"] = MemoryType(data_copy["memory_type"])
        data_copy["lifecycle_status"] = LifecycleStatus(data_copy["lifecycle_status"])
        return cls(**data_copy)


# ============================================================================
# 2. Configuration & Hyperparameters
# ============================================================================

@dataclass
class ScoringWeights:
    """
    Configurable weights for the utility scoring function:
    U(m) = w_r * R + w_i * I + w_f * F_norm + w_t * T + w_c * C
    Must sum to 1.0 (+- 1e-4 tolerance).
    """
    w_r: float = 0.25   # Weight for Recency (freshness)
    w_i: float = 0.25   # Weight for Importance (intrinsic value)
    w_f: float = 0.20   # Weight for Frequency (repetition count)
    w_t: float = 0.15   # Weight for Temporal Stability (invariance over time)
    w_c: float = 0.15   # Weight for Consistency / Confidence

    def __post_init__(self):
        for name, val in [("w_r", self.w_r), ("w_i", self.w_i), ("w_f", self.w_f),
                          ("w_t", self.w_t), ("w_c", self.w_c)]:
            if val < 0.0:
                raise ValueError(f"Weight {name}={val} cannot be negative.")
        total = self.w_r + self.w_i + self.w_f + self.w_t + self.w_c
        if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"Scoring weights must sum to 1.0; got {total:.6f}")


@dataclass
class AllocationConfig:
    """
    Budget allocation bounds and total limits.
    Budget is measured in integer token units.
    """
    total_budget: int = 1000
    min_episodic_share: float = 0.20
    max_episodic_share: float = 0.80
    min_semantic_share: float = 0.20
    max_semantic_share: float = 0.80

    def __post_init__(self):
        if self.total_budget < 0:
            raise ValueError("total_budget cannot be negative.")
        if not (0.0 <= self.min_episodic_share <= self.max_episodic_share <= 1.0):
            raise ValueError("Invalid episodic share bounds.")
        if not (0.0 <= self.min_semantic_share <= self.max_semantic_share <= 1.0):
            raise ValueError("Invalid semantic share bounds.")
        if self.min_episodic_share + self.min_semantic_share > 1.0:
            raise ValueError("Minimum shares sum exceeds 1.0, making budget allocation impossible.")


# ============================================================================
# 3. Extraction & Token Estimation Heuristics
# ============================================================================

def estimate_token_cost(text: str) -> int:
    """
    Standard token cost estimator.
    Approximates tokens as ~1.3 tokens per whitespace-separated word (or ~4 chars per token).
    """
    if not text.strip():
        return 0
    words = text.strip().split()
    # Baseline: ceil(len(words) * 1.3), min 1 token per non-empty string
    estimated = max(1, math.ceil(len(words) * 1.3))
    return estimated


def _classify_candidate_heuristics(text: str) -> Tuple[MemoryType, float, float]:
    """
    Heuristics to determine memory_type, importance, and temporal stability.

    Conceptual separation:
    - Recency R: calculated during scoring based on time elapsed since last observed.
    - Stability T: indicates whether the information is permanent/enduring (semantic facts,
      user traits, core constraints) vs transient/session-bound (episodic requests, immediate steps).
    """
    text_lower = text.lower()

    # Semantic indicators: persistent preferences, traits, universal facts
    semantic_patterns = [
        r"\bi am\b", r"\bi like\b", r"\bi prefer\b", r"\bmy name is\b",
        r"\bmy job is\b", r"\bi work at\b", r"\balways\b", r"\bnever\b",
        r"\buser prefers\b", r"\blives in\b", r"\bspeaks\b", r"\bdefinition\b"
    ]

    # Episodic indicators: specific temporal actions, queries, one-off tasks
    episodic_patterns = [
        r"\btoday\b", r"\byesterday\b", r"\bnow\b", r"\bjust\b",
        r"\bcan you\b", r"\bhelp me\b", r"\bdebug\b", r"\bthis error\b",
        r"\brun this\b", r"\blast time\b", r"\bmoved to\b", r"\bcurrent task\b"
    ]

    is_semantic = any(re.search(pat, text_lower) for pat in semantic_patterns)
    is_episodic = any(re.search(pat, text_lower) for pat in episodic_patterns)

    if is_semantic and not is_episodic:
        memory_type = MemoryType.SEMANTIC
        importance = 0.8
        stability = 0.9  # Enduring fact
    elif is_episodic and not is_semantic:
        memory_type = MemoryType.EPISODIC
        importance = 0.6
        stability = 0.3  # Transient/event specific
    else:
        # Default fallback: check length and structure
        if len(text.split()) > 15:
            memory_type = MemoryType.EPISODIC
            importance = 0.5
            stability = 0.4
        else:
            memory_type = MemoryType.SEMANTIC
            importance = 0.6
            stability = 0.7

    return memory_type, importance, stability


def _is_filler_sentence(sent: str) -> bool:
    """Detect trivial conversational pleasantries, formatting artifacts, and fillers that provide no persistent value."""
    # Clean leading formatting (list numbers, markdown bullets, brackets)
    cleaned_sent = re.sub(r"^\s*(\d+[\.\)]|\*|\-|\+|\[|\#+)\s*", "", sent).strip()
    cleaned = re.sub(r"[^\w\s]", "", cleaned_sent.lower()).strip()
    words = cleaned.split()

    if not words:
        return True

    # Standalone numbers, roman numerals, single letters, or short fragments without factual anchor
    if len(words) < 3:
        core_predicates = {"work", "live", "name", "prefer", "like", "graduated", "degree", "use", "code"}
        if not any(kw in cleaned for kw in core_predicates):
            return True

    fillers = {
        "hello", "hi", "hey", "ok", "okay", "thanks", "thank you", "sure",
        "noted", "got it", "understood", "welcome", "you are welcome",
        "great", "sounds good", "alright", "yes", "no", "congratulations",
        "congrats", "good morning", "good evening", "nice to meet you",
        "pleased to meet you", "fascinating area"
    }
    if cleaned in fillers:
        return True
    if re.match(r"^(hello|hi|hey|congratulations on|congrats on|how can i help)\b", cleaned):
        return True
    return False


def extract_memories(
    conversation_history: List[Dict[str, Any]],
    session_id: str = "session_0",
    current_timestamp: Optional[float] = None,
    tokenizer_fn: Optional[Callable[[str], int]] = None
) -> List[MemoryItem]:
    """
    Extract candidate memories from a conversation history session.
    Deduplicates repeated candidates while preserving frequency and source provenance.
    Delegates to modular HeuristicCandidateExtractor.
    """
    from candidate_generation import HeuristicCandidateExtractor
    extractor = HeuristicCandidateExtractor(tokenizer_fn=tokenizer_fn)
    return extractor.extract_candidates(
        conversation_history=conversation_history,
        session_id=session_id,
        current_timestamp=current_timestamp,
        tokenizer_fn=tokenizer_fn
    )


# ============================================================================
# 4. Utility Scoring Function
# ============================================================================

def score_memories(
    memories: List[MemoryItem],
    weights: ScoringWeights = ScoringWeights(),
    current_timestamp: Optional[float] = None,
    half_life_seconds: float = 86400.0  # 24 hour exponential decay for recency
) -> List[MemoryItem]:
    """
    Computes utility score U(m) for each candidate memory:
        U(m) = w_r * R + w_i * I + w_f * F_norm + w_t * T + w_c * C

    Heuristic assumptions documented:
    1. Recency R: Exponential time-decay R = exp(-delta_t / half_life). Bounded in [0, 1].
    2. Importance I: Intrinsic value assigned during extraction or evaluation, bounded in [0, 1].
    3. Frequency F_norm: Normalized logarithmic saturation F_norm = min(1.0, log2(1 + F) / log2(11)),
       reaching 1.0 when frequency >= 10.
    4. Temporal Stability T: Separate from recency; reflects how timeless vs volatile the knowledge is.
    5. Consistency/Confidence C: Reliability score in [0, 1], penalizing conflicting statements.
    """
    if current_timestamp is None:
        current_timestamp = time.time()

    scored_list: List[MemoryItem] = []

    for mem in memories:
        # 1. Recency R
        delta_t = max(0.0, current_timestamp - mem.timestamp)
        recency = math.exp(-delta_t / max(1.0, half_life_seconds))
        recency = max(0.0, min(1.0, recency))
        mem.recency = recency

        # 2. Importance I
        importance = max(0.0, min(1.0, mem.importance))

        # 3. Frequency F_norm
        # log2(1 + 10) approx 3.4594
        f_norm = min(1.0, math.log2(1 + max(1, mem.frequency)) / math.log2(11))

        # 4. Stability T
        stability = max(0.0, min(1.0, mem.stability))

        # 5. Consistency C
        confidence = max(0.0, min(1.0, mem.consistency_confidence))

        # Linear combination
        u_score = (
            weights.w_r * recency +
            weights.w_i * importance +
            weights.w_f * f_norm +
            weights.w_t * stability +
            weights.w_c * confidence
        )
        mem.utility_score = round(max(0.0, min(1.0, u_score)), 6)
        scored_list.append(mem)

    return scored_list


# ============================================================================
# 5. Adaptive Budget Allocation
# ============================================================================

def allocate_budget(
    candidate_memories: List[MemoryItem],
    config: AllocationConfig
) -> Dict[MemoryType, int]:
    """
    Dynamically allocates a fixed total token budget between episodic and semantic memory
    according to their estimated demand, while enforcing min and max bounds.

    Demand formulation:
    Demand for memory type M is proportional to the aggregate utility of its candidates:
    Demand(M) = sum_{m in candidates of type M} U(m) * max(1, token_cost(m))

    If no candidates exist for a type or all demand is zero, falls back to a 50/50 baseline
    within the configured bounds.

    Guarantees:
    - allocated_episodic + allocated_semantic <= total_budget (and == total_budget if total_budget > 0)
    - min_share * total_budget <= allocation <= max_share * total_budget
    """
    total_budget = config.total_budget
    if total_budget <= 0:
        return {MemoryType.EPISODIC: 0, MemoryType.SEMANTIC: 0}

    episodic_candidates = [m for m in candidate_memories if m.memory_type == MemoryType.EPISODIC]
    semantic_candidates = [m for m in candidate_memories if m.memory_type == MemoryType.SEMANTIC]

    # Calculate demand weighted by utility and token cost
    demand_episodic = sum(max(0.01, m.utility_score) * max(1, m.token_cost) for m in episodic_candidates)
    demand_semantic = sum(max(0.01, m.utility_score) * max(1, m.token_cost) for m in semantic_candidates)
    total_demand = demand_episodic + demand_semantic

    if total_demand <= 0.0:
        # Default even demand share
        raw_episodic_share = 0.50
    else:
        raw_episodic_share = demand_episodic / total_demand

    # Enforce minimum and maximum bounds for episodic share
    clamped_episodic_share = min(config.max_episodic_share, max(config.min_episodic_share, raw_episodic_share))

    # Semantic share must also respect its bounds
    semantic_share = 1.0 - clamped_episodic_share
    if semantic_share < config.min_semantic_share:
        semantic_share = config.min_semantic_share
        clamped_episodic_share = 1.0 - semantic_share
    elif semantic_share > config.max_semantic_share:
        semantic_share = config.max_semantic_share
        clamped_episodic_share = 1.0 - semantic_share

    # Integer token allocation
    episodic_budget = int(round(clamped_episodic_share * total_budget))
    semantic_budget = total_budget - episodic_budget

    # Final sanity bounds enforcement on discrete integers
    min_episodic_tokens = math.ceil(config.min_episodic_share * total_budget)
    max_episodic_tokens = math.floor(config.max_episodic_share * total_budget)
    episodic_budget = max(min_episodic_tokens, min(max_episodic_tokens, episodic_budget))
    semantic_budget = total_budget - episodic_budget

    return {
        MemoryType.EPISODIC: episodic_budget,
        MemoryType.SEMANTIC: semantic_budget
    }


# ============================================================================
# 6. Memory Selection (Budget Enforcement)
# ============================================================================

def select_memories(
    memories: List[MemoryItem],
    budget: int
) -> Tuple[List[MemoryItem], int]:
    """
    Selects candidate memories that fit within the store's budget without silently exceeding it.
    Uses greedy knapsack heuristic ranking candidates by utility density: U(m) / token_cost(m).

    Returns:
        Tuple of (selected_memories, total_tokens_used)
    """
    if budget <= 0 or not memories:
        return [], 0

    # Rank by utility density (utility per token), tie-broken by raw utility score
    ranked = sorted(
        memories,
        key=lambda m: (m.utility_score / max(1, m.token_cost), m.utility_score),
        reverse=True
    )

    selected: List[MemoryItem] = []
    used_tokens = 0

    for item in ranked:
        cost = max(1, item.token_cost)
        if used_tokens + cost <= budget:
            selected.append(item)
            used_tokens += cost

    return selected, used_tokens


# ============================================================================
# 7. Storage Interfaces & Local JSON Persistence
# ============================================================================

class MemoryStore:
    """Abstract interface separating storage from memory-management logic."""
    def save(
        self,
        episodic_memories: List[MemoryItem],
        semantic_memories: List[MemoryItem],
        allocation_summary: Dict[str, Any]
    ) -> None:
        raise NotImplementedError

    def load(self) -> Tuple[List[MemoryItem], List[MemoryItem], Dict[str, Any]]:
        raise NotImplementedError


class LocalJsonMemoryStore(MemoryStore):
    """Simple, transparent local persistence engine writing structured JSON."""
    def __init__(self, filepath: str):
        self.filepath = filepath

    def save(
        self,
        episodic_memories: List[MemoryItem],
        semantic_memories: List[MemoryItem],
        allocation_summary: Dict[str, Any]
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        payload = {
            "version": "phase_1.0",
            "saved_at": time.time(),
            "allocation_summary": allocation_summary,
            "episodic_store": [m.to_dict() for m in episodic_memories],
            "semantic_store": [m.to_dict() for m in semantic_memories],
        }
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load(self) -> Tuple[List[MemoryItem], List[MemoryItem], Dict[str, Any]]:
        if not os.path.exists(self.filepath):
            return [], [], {}
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        episodic = [MemoryItem.from_dict(d) for d in data.get("episodic_store", [])]
        semantic = [MemoryItem.from_dict(d) for d in data.get("semantic_store", [])]
        summary = data.get("allocation_summary", {})
        return episodic, semantic, summary


def store_memories(
    episodic: List[MemoryItem],
    semantic: List[MemoryItem],
    allocation_summary: Dict[str, Any],
    filepath: str
) -> None:
    """Convenience helper for storing memories to a local JSON file."""
    store = LocalJsonMemoryStore(filepath)
    store.save(episodic, semantic, allocation_summary)


# ============================================================================
# 8. Complete Pipeline Runner
# ============================================================================

def run_phase1_pipeline(
    conversation_history: List[Dict[str, Any]],
    session_id: str = "session_0",
    total_budget: int = 1000,
    scoring_weights: Optional[ScoringWeights] = None,
    allocation_config: Optional[AllocationConfig] = None,
    persistence_path: Optional[str] = None,
    current_timestamp: Optional[float] = None,
    candidate_extractor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes the complete Phase 1 pipeline:
    Conversation history -> Extraction -> Utility Scoring -> Dynamic Allocation -> Selection -> Storage
    """
    if scoring_weights is None:
        scoring_weights = ScoringWeights()
    if allocation_config is None:
        allocation_config = AllocationConfig(total_budget=total_budget)
    else:
        allocation_config.total_budget = total_budget

    if current_timestamp is None:
        current_timestamp = time.time()

    # 1. Extraction
    if candidate_extractor is not None:
        candidates = candidate_extractor.extract_candidates(
            conversation_history=conversation_history,
            session_id=session_id,
            current_timestamp=current_timestamp
        )
    else:
        candidates = extract_memories(
            conversation_history=conversation_history,
            session_id=session_id,
            current_timestamp=current_timestamp
        )

    # 2. Utility Scoring
    scored_candidates = score_memories(
        memories=candidates,
        weights=scoring_weights,
        current_timestamp=current_timestamp
    )

    # 3. Dynamic Budget Allocation
    budget_split = allocate_budget(scored_candidates, allocation_config)
    episodic_budget = budget_split[MemoryType.EPISODIC]
    semantic_budget = budget_split[MemoryType.SEMANTIC]

    # Partition candidates by type
    episodic_candidates = [m for m in scored_candidates if m.memory_type == MemoryType.EPISODIC]
    semantic_candidates = [m for m in scored_candidates if m.memory_type == MemoryType.SEMANTIC]

    # 4. Selection within budgets
    selected_episodic, used_episodic_tokens = select_memories(episodic_candidates, episodic_budget)
    selected_semantic, used_semantic_tokens = select_memories(semantic_candidates, semantic_budget)

    # 5. Build structured summary
    allocation_summary = {
        "session_id": session_id,
        "total_budget": total_budget,
        "allocated_budget": {
            "episodic": episodic_budget,
            "semantic": semantic_budget,
        },
        "tokens_used": {
            "episodic": used_episodic_tokens,
            "semantic": used_semantic_tokens,
            "total": used_episodic_tokens + used_semantic_tokens,
        },
        "candidate_counts": {
            "total": len(scored_candidates),
            "episodic": len(episodic_candidates),
            "semantic": len(semantic_candidates),
        },
        "selected_counts": {
            "episodic": len(selected_episodic),
            "semantic": len(selected_semantic),
            "total": len(selected_episodic) + len(selected_semantic),
        },
        "scoring_weights": asdict(scoring_weights),
    }

    # 6. Persistence
    if persistence_path:
        store_memories(selected_episodic, selected_semantic, allocation_summary, persistence_path)

    return {
        "allocation_summary": allocation_summary,
        "selected_episodic": [m.to_dict() for m in selected_episodic],
        "selected_semantic": [m.to_dict() for m in selected_semantic],
    }
