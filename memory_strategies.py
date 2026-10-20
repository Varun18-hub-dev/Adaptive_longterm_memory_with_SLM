"""
Adaptive Long-Term Memory for Language Models
Phase 4: Common Memory Strategy Interface & Baseline Implementations

Strategies:
1. BaseMemoryStrategy: Unified interface for multi-session intake, storage budget enforcement, and retrieval.
2. FlatMemoryBaseline: Undivided single pool without hierarchical budget partition.
3. FixedHierarchicalStrategy: Static episodic / semantic budget split (e.g. 70/30, 50/50, 30/70).
4. AdaptiveHierarchicalStrategy: Dynamic demand-weighted budget split with share bounds.
5. AblatedMemoryStrategy: Configurable ablation runner for component ablation experiments.

Fairness Constraint: Total persistent-memory budget is strictly equal across all strategies.
All representations that consume storage count toward the budget.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory_core import (
    AllocationConfig,
    LifecycleStatus,
    MemoryItem,
    MemoryType,
    ScoringWeights,
    allocate_budget,
    estimate_token_cost,
    extract_memories,
    score_memories,
    select_memories,
)
from memory_lifecycle import (
    AdaptiveMemoryManager,
    AuditLogEntry,
    consolidate_episodic_memories,
    deduplicate_semantically,
    detect_conflicts_and_supersede,
    evaluate_lifecycle_transitions,
)
from memory_interface import (
    BaseMemoryStrategy,
    RetrievedEvidence,
    STRATEGY_REGISTRY,
    register_strategy,
    get_strategy,
    list_registered_strategies,
    MockMemoryStrategy,
)
from memory_retrieval import (
    RetrievalWeights,
    assemble_memory_context,
    retrieve_memories,
)


# ============================================================================
# 1. Flat Memory Baseline
# ============================================================================

@register_strategy("flat_memory")
@register_strategy("flat_memory_baseline")
class FlatMemoryBaseline(BaseMemoryStrategy):
    """
    Undivided memory baseline:
    - All memories live in a single flat pool without separate episodic or semantic stores.
    - No hierarchical budget allocator.
    - Pruning under budget selects top utility-density items across the entire undivided pool.
    - Uses identical candidate extraction, quality gating, and tokenizer accounting as hierarchical strategies.
    """

    def __init__(
        self,
        total_budget: int = 150,
        scoring_weights: Optional[ScoringWeights] = None,
        admission_gate: Optional[Any] = None,
        candidate_extractor: Optional[Any] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ):
        super().__init__(
            name="flat_memory_baseline",
            total_budget=total_budget,
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor,
            tokenizer_fn=tokenizer_fn
        )
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.memory_store: List[MemoryItem] = []
        self.archive_store: List[MemoryItem] = []
        self.audit_log: List[AuditLogEntry] = []
        self.rejected_candidates: List[Tuple[MemoryItem, Any, str]] = []

    def reset(self) -> None:
        self.memory_store.clear()
        self.archive_store.clear()
        self.audit_log.clear()
        self.rejected_candidates.clear()

    def process_session(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        current_timestamp: float
    ) -> Dict[str, Any]:
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

        scored_candidates = score_memories(
            memories=new_candidates,
            weights=self.scoring_weights,
            current_timestamp=current_timestamp
        )

        # Candidate Admission Gate (quality & confidence filtering)
        session_rejected: List[Tuple[MemoryItem, Any, str]] = []
        if self.admission_gate is not None:
            if hasattr(self.admission_gate, "filter_quality"):
                scored_candidates, session_rejected = self.admission_gate.filter_quality(
                    scored_candidates,
                    audit_entries=self.audit_log,
                    current_timestamp=current_timestamp
                )
            elif hasattr(self.admission_gate, "evaluate_candidates"):
                eval_res = self.admission_gate.evaluate_candidates(
                    scored_candidates,
                    existing_memories=self.memory_store,
                    current_timestamp=current_timestamp
                )
                scored_candidates = eval_res.admitted_candidates
                session_rejected = eval_res.rejected_candidates
                if eval_res.audit_entries:
                    self.audit_log.extend(eval_res.audit_entries)
            self.rejected_candidates.extend(session_rejected)

        # Soft deduplication against existing flat pool
        self.memory_store, distinct = deduplicate_semantically(
            incoming_memories=scored_candidates,
            existing_memories=self.memory_store,
            audit_entries=self.audit_log,
            current_timestamp=current_timestamp
        )
        self.memory_store.extend(distinct)

        # Conflict resolution and lifecycle
        detect_conflicts_and_supersede(self.memory_store, audit_entries=self.audit_log, current_timestamp=current_timestamp)
        self.memory_store = score_memories(self.memory_store, self.scoring_weights, current_timestamp)
        self.memory_store = evaluate_lifecycle_transitions(self.memory_store, current_timestamp=current_timestamp, audit_entries=self.audit_log)

        # Filter expired or superseded into archive
        active_items = []
        for m in self.memory_store:
            if m.lifecycle_status in (LifecycleStatus.EXPIRED, LifecycleStatus.SUPERSEDED):
                self.archive_store.append(m)
            else:
                active_items.append(m)

        # Select items that strictly fit total_budget in undivided pool
        selected, used_tokens = select_memories(active_items, self.total_budget)
        selected_ids = {m.id for m in selected}

        for m in active_items:
            if m.id not in selected_ids:
                self.archive_store.append(m)

        self.memory_store = selected

        return {
            "strategy": self.name,
            "session_id": session_id,
            "used_tokens": used_tokens,
            "total_budget": self.total_budget,
            "stored_count": len(self.memory_store),
            "archive_count": len(self.archive_store),
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        # In flat memory, all items reside in one store
        return retrieve_memories(
            query=query,
            episodic_store=[],
            semantic_store=self.memory_store,
            archive_store=self.archive_store,
            top_k=top_k
        )

    def get_persisted_memories(self) -> List[MemoryItem]:
        return list(self.memory_store)


# ============================================================================
# 2. Fixed Hierarchical Allocation Strategy
# ============================================================================

class FixedHierarchicalStrategy(BaseMemoryStrategy):
    """
    Fixed hierarchical baseline:
    - Maintains separate episodic and semantic stores with a STATIC budget ratio.
    - Examples: 70/30 (episodic heavy), 50/50 (even split), 30/70 (semantic heavy).
    """

    def __init__(
        self,
        total_budget: int = 150,
        episodic_share: float = 0.50,
        scoring_weights: Optional[ScoringWeights] = None,
        admission_gate: Optional[Any] = None,
        candidate_extractor: Optional[Any] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ):
        semantic_share = round(1.0 - episodic_share, 2)
        name = f"fixed_{int(episodic_share*100)}_{int(semantic_share*100)}"
        super().__init__(
            name=name,
            total_budget=total_budget,
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor,
            tokenizer_fn=tokenizer_fn
        )
        self.episodic_share = episodic_share
        self.semantic_share = semantic_share
        self.scoring_weights = scoring_weights or ScoringWeights()

        # Fixed integer budgets
        self.episodic_budget = int(round(self.episodic_share * total_budget))
        self.semantic_budget = total_budget - self.episodic_budget

        # Delegate state management to AdaptiveMemoryManager with fixed bounds
        self.config = AllocationConfig(
            total_budget=total_budget,
            min_episodic_share=episodic_share,
            max_episodic_share=episodic_share,
            min_semantic_share=semantic_share,
            max_semantic_share=semantic_share
        )
        self.manager = AdaptiveMemoryManager(
            total_budget=total_budget,
            scoring_weights=self.scoring_weights,
            allocation_config=self.config,
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor,
            tokenizer_fn=tokenizer_fn
        )

    def reset(self) -> None:
        self.manager.episodic_store.clear()
        self.manager.semantic_store.clear()
        self.manager.archive_store.clear()
        self.manager.audit_log.clear()
        self.manager.rejected_candidates.clear()

    def process_session(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        current_timestamp: float
    ) -> Dict[str, Any]:
        res = self.manager.process_session(
            conversation_history=conversation_history,
            session_id=session_id,
            current_timestamp=current_timestamp
        )
        res["strategy"] = self.name
        return res

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        return retrieve_memories(
            query=query,
            episodic_store=self.manager.episodic_store,
            semantic_store=self.manager.semantic_store,
            archive_store=self.manager.archive_store,
            top_k=top_k
        )

    def get_persisted_memories(self) -> List[MemoryItem]:
        return list(self.manager.episodic_store) + list(self.manager.semantic_store)


@register_strategy("fixed_70_30")
def create_fixed_70_30(total_budget: int = 500, **kwargs: Any) -> FixedHierarchicalStrategy:
    return FixedHierarchicalStrategy(total_budget=total_budget, episodic_share=0.70, **kwargs)


@register_strategy("fixed_50_50")
def create_fixed_50_50(total_budget: int = 500, **kwargs: Any) -> FixedHierarchicalStrategy:
    return FixedHierarchicalStrategy(total_budget=total_budget, episodic_share=0.50, **kwargs)


@register_strategy("fixed_30_70")
def create_fixed_30_70(total_budget: int = 500, **kwargs: Any) -> FixedHierarchicalStrategy:
    return FixedHierarchicalStrategy(total_budget=total_budget, episodic_share=0.30, **kwargs)


# ============================================================================
# 3. Adaptive Hierarchical Allocation Strategy
# ============================================================================

@register_strategy("adaptive_hierarchical")
class AdaptiveHierarchicalStrategy(BaseMemoryStrategy):
    """
    Adaptive hierarchical memory strategy:
    - Dynamically allocates storage budget between episodic and semantic stores
      proportional to estimated demand.
    - Consolidates reinforced episodic observations into compact semantic knowledge.
    - Enforces configurable share bounds [min_share, max_share].
    """

    def __init__(
        self,
        total_budget: int = 150,
        min_episodic_share: float = 0.20,
        max_episodic_share: float = 0.80,
        scoring_weights: Optional[ScoringWeights] = None,
        admission_gate: Optional[Any] = None,
        candidate_extractor: Optional[Any] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ):
        super().__init__(
            name="adaptive_hierarchical",
            total_budget=total_budget,
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor,
            tokenizer_fn=tokenizer_fn
        )
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.config = AllocationConfig(
            total_budget=total_budget,
            min_episodic_share=min_episodic_share,
            max_episodic_share=max_episodic_share,
            min_semantic_share=1.0 - max_episodic_share,
            max_semantic_share=1.0 - min_episodic_share
        )
        self.manager = AdaptiveMemoryManager(
            total_budget=total_budget,
            scoring_weights=self.scoring_weights,
            allocation_config=self.config,
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor,
            tokenizer_fn=tokenizer_fn
        )

    def reset(self) -> None:
        self.manager.episodic_store.clear()
        self.manager.semantic_store.clear()
        self.manager.archive_store.clear()
        self.manager.audit_log.clear()
        self.manager.rejected_candidates.clear()

    def process_session(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        current_timestamp: float
    ) -> Dict[str, Any]:
        res = self.manager.process_session(
            conversation_history=conversation_history,
            session_id=session_id,
            current_timestamp=current_timestamp
        )
        res["strategy"] = self.name
        return res

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        return retrieve_memories(
            query=query,
            episodic_store=self.manager.episodic_store,
            semantic_store=self.manager.semantic_store,
            archive_store=self.manager.archive_store,
            top_k=top_k
        )

    def get_persisted_memories(self) -> List[MemoryItem]:
        return list(self.manager.episodic_store) + list(self.manager.semantic_store)


# ============================================================================
# 4. Configurable Ablation Strategy
# ============================================================================

class AblatedMemoryStrategy(BaseMemoryStrategy):
    """
    Configurable ablation strategy:
    Allows selectively disabling core components to measure their individual impact:
    - disable_adaptive: falls back to static 50/50 split
    - disable_utility: sets uniform utility score U(m) = 0.50
    - disable_consolidation: disables promotion from episodic to semantic
    - disable_temporal_validity: does not filter or mark superseded memories
    - disable_reranking: retrieves by raw recency only without multi-factor reranking
    """

    def __init__(
        self,
        total_budget: int = 150,
        disable_adaptive: bool = False,
        disable_utility: bool = False,
        disable_consolidation: bool = False,
        disable_temporal_validity: bool = False,
        disable_reranking: bool = False,
        admission_gate: Optional[Any] = None,
        candidate_extractor: Optional[Any] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ):
        ablation_name_parts = []
        if disable_adaptive:
            ablation_name_parts.append("no_adaptive")
        if disable_utility:
            ablation_name_parts.append("no_utility")
        if disable_consolidation:
            ablation_name_parts.append("no_consolidation")
        if disable_temporal_validity:
            ablation_name_parts.append("no_temporal_validity")
        if disable_reranking:
            ablation_name_parts.append("no_reranking")

        name = "ablation_" + ("_".join(ablation_name_parts) if ablation_name_parts else "none")
        super().__init__(
            name=name,
            total_budget=total_budget,
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor,
            tokenizer_fn=tokenizer_fn
        )

        self.disable_adaptive = disable_adaptive
        self.disable_utility = disable_utility
        self.disable_consolidation = disable_consolidation
        self.disable_temporal_validity = disable_temporal_validity
        self.disable_reranking = disable_reranking

        # Underlying manager
        share_min = 0.50 if disable_adaptive else 0.20
        share_max = 0.50 if disable_adaptive else 0.80
        self.config = AllocationConfig(
            total_budget=total_budget,
            min_episodic_share=share_min,
            max_episodic_share=share_max,
            min_semantic_share=1.0 - share_max,
            max_semantic_share=1.0 - share_min
        )
        self.manager = AdaptiveMemoryManager(
            total_budget=total_budget,
            allocation_config=self.config,
            admission_gate=admission_gate,
            candidate_extractor=candidate_extractor,
            tokenizer_fn=tokenizer_fn,
            enable_consolidation=not disable_consolidation
        )

    def reset(self) -> None:
        self.manager.episodic_store.clear()
        self.manager.semantic_store.clear()
        self.manager.archive_store.clear()
        self.manager.audit_log.clear()
        self.manager.rejected_candidates.clear()

    def process_session(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        current_timestamp: float
    ) -> Dict[str, Any]:
        # If utility is disabled, override utility score
        res = self.manager.process_session(
            conversation_history=conversation_history,
            session_id=session_id,
            current_timestamp=current_timestamp
        )
        if self.disable_utility:
            for m in self.manager.episodic_store + self.manager.semantic_store:
                m.utility_score = 0.50

        res["strategy"] = self.name
        return res

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        weights = None
        if self.disable_reranking:
            # Rank strictly by recency
            weights = RetrievalWeights(w_rel=0.0, w_util=0.0, w_rec=1.0, w_stab=0.0)

        # If temporal validity disabled, treat all memories as active
        if self.disable_temporal_validity:
            all_mems = list(self.manager.episodic_store) + list(self.manager.semantic_store) + list(self.manager.archive_store)
            for m in all_mems:
                m.lifecycle_status = LifecycleStatus.ACTIVE
            return retrieve_memories(
                query=query,
                episodic_store=[],
                semantic_store=all_mems,
                archive_store=[],
                top_k=top_k,
                weights=weights
            )

        return retrieve_memories(
            query=query,
            episodic_store=self.manager.episodic_store,
            semantic_store=self.manager.semantic_store,
            archive_store=self.manager.archive_store,
            top_k=top_k,
            weights=weights
        )

    def get_persisted_memories(self) -> List[MemoryItem]:
        return list(self.manager.episodic_store) + list(self.manager.semantic_store)


@register_strategy("ablation_no_adaptive")
def create_ablation_no_adaptive(total_budget: int = 500, **kwargs: Any) -> AblatedMemoryStrategy:
    return AblatedMemoryStrategy(total_budget=total_budget, disable_adaptive=True, **kwargs)


@register_strategy("ablation_no_utility")
def create_ablation_no_utility(total_budget: int = 500, **kwargs: Any) -> AblatedMemoryStrategy:
    return AblatedMemoryStrategy(total_budget=total_budget, disable_utility=True, **kwargs)


@register_strategy("ablation_no_consolidation")
def create_ablation_no_consolidation(total_budget: int = 500, **kwargs: Any) -> AblatedMemoryStrategy:
    return AblatedMemoryStrategy(total_budget=total_budget, disable_consolidation=True, **kwargs)


@register_strategy("ablation_no_temporal_validity")
def create_ablation_no_temporal_validity(total_budget: int = 500, **kwargs: Any) -> AblatedMemoryStrategy:
    return AblatedMemoryStrategy(total_budget=total_budget, disable_temporal_validity=True, **kwargs)
