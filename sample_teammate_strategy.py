"""
Sample standalone memory strategy implemented by an independent researcher / teammate.
Demonstrates:
1. Strategy inherits from BaseMemoryStrategy.
2. Registration with @register_strategy("sample_teammate_memory").
3. Works out-of-the-box with:
   python run_experiments.py --strategy-module sample_teammate_strategy.py --strategy sample_teammate_memory
   OR via auto-discovery if placed in strategies/ or named strategy_*.py.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from memory_interface import BaseMemoryStrategy, RetrievedEvidence, register_strategy


@register_strategy("sample_teammate_memory")
class SampleTeammateMemory(BaseMemoryStrategy):
    """
    Teammate's custom memory implementation using an isolated flat evidence store.
    Requires no knowledge of episodic or semantic internals.
    """

    def __init__(
        self,
        total_budget: int = 500,
        tokenizer_fn: Optional[Callable[[str], int]] = None,
        **kwargs: Any
    ):
        super().__init__(
            name="sample_teammate_memory",
            total_budget=total_budget,
            tokenizer_fn=tokenizer_fn,
            **kwargs
        )
        self.evidence_store: List[RetrievedEvidence] = []

    def reset(self) -> None:
        self.evidence_store.clear()

    def process_session(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        current_timestamp: float
    ) -> Dict[str, Any]:
        for idx, turn in enumerate(conversation_history):
            content = turn.get("content", "").strip()
            role = turn.get("role", "user")
            ts = turn.get("timestamp", current_timestamp)
            if not content:
                continue

            item = RetrievedEvidence(
                id=f"{session_id}_t{idx}",
                content=content,
                score=1.0,
                timestamp=ts,
                lifecycle_status="current",
                metadata={"role": role, "session_id": session_id}
            )
            self.evidence_store.append(item)

        # Enforce budget cap
        while self.get_storage_usage() > self.total_budget and self.evidence_store:
            self.evidence_store.pop(0)

        return {
            "strategy": self.name,
            "session_id": session_id,
            "stored_count": len(self.evidence_store),
            "storage_tokens_used": self.get_storage_usage(),
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[RetrievedEvidence, float]]:
        q_words = set(query.lower().split())
        scored: List[Tuple[RetrievedEvidence, float]] = []
        for ev in self.evidence_store:
            ev_words = set(ev.content.lower().split())
            overlap = len(q_words & ev_words) / max(1, len(q_words))
            scored.append((ev, round(overlap, 3)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_persisted_memories(self) -> List[RetrievedEvidence]:
        return list(self.evidence_store)
