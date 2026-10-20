"""
Adaptive Long-Term Memory for Language Models
Memory Strategy Integration Contract & Registry

This module defines the minimal, strategy-neutral interface that every persistent-memory
implementation must fulfill to integrate with the shared experiment and evaluation pipeline.

Architecture Guarantee:
The shared pipeline does NOT mandate that a memory strategy use episodic or semantic stores,
consolidation, or any specific internal ranking/vector indexing algorithm.
A strategy can be a flat buffer, a static quota hierarchy, an adaptive allocator, or a custom neural/graph store.

Integration Lifecycle:
1. Registration: The strategy registers its name and factory via @register_strategy("name").
2. Instantiation: Initialized with total_budget, tokenizer_fn, and optional config dict.
3. Reset: reset() is called before each independent benchmark question to ensure zero cross-contamination.
4. Intake: process_session() ingests conversation history turns for each session sequentially.
5. Retrieval: retrieve() returns relevant memories / RetrievedEvidence for the question query.
6. Accounting: get_storage_usage() reports current persistent memory footprint in tokens.
7. Diagnostics: get_stats() reports optional strategy-specific metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib
import importlib.util
import inspect
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


def default_word_count_estimator(text: str) -> int:
    """Fallback token estimation: ~1.3 tokens per whitespace-separated word."""
    if not text or not text.strip():
        return 0
    words = text.strip().split()
    return max(1, math.ceil(len(words) * 1.3))


@dataclass
class RetrievedEvidence:
    """
    Strategy-neutral representation of evidence returned by a memory retrieval call.

    Attributes:
        content: The text assertion or snippet to be formatted into the prompt context.
        score: Relevance or utility score (higher = more relevant).
        id: Optional identifier of the memory item.
        timestamp: Optional timestamp when the observation occurred.
        lifecycle_status: Optional status string (e.g. 'current', 'superseded', 'temporary').
        metadata: Strategy-specific metadata (provenance, turn index, tags, etc.).
    """
    content: str
    score: float = 0.0
    id: Optional[str] = None
    timestamp: Optional[float] = None
    lifecycle_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "id": self.id,
            "timestamp": self.timestamp,
            "lifecycle_status": self.lifecycle_status,
            "metadata": self.metadata,
        }


class BaseMemoryStrategy(ABC):
    """
    Minimal contract for persistent-memory implementations.
    Independent teammates subclass this interface to integrate their strategy into the shared pipeline.
    """

    def __init__(
        self,
        name: str,
        total_budget: int = 500,
        tokenizer_fn: Optional[Callable[[str], int]] = None,
        config: Optional[Dict[str, Any]] = None,
        admission_gate: Optional[Any] = None,
        candidate_extractor: Optional[Any] = None,
        **kwargs: Any
    ):
        """
        Initialize strategy.
        Args:
            name: Unique name identifier for the strategy.
            total_budget: Persistent-memory budget limit in tokens (shared constraint).
            tokenizer_fn: Callable mapping text -> integer token count.
            config: Optional dictionary containing strategy-specific hyperparameters.
            admission_gate: Optional admission gate for quality filtering.
            candidate_extractor: Optional candidate extractor.
        """
        self.name = name
        self.total_budget = total_budget
        self.tokenizer_fn = tokenizer_fn or default_word_count_estimator
        self.config = config or {}
        self.admission_gate = admission_gate
        self.candidate_extractor = candidate_extractor

    def reset(self) -> None:
        """
        Reset all internal memory stores and state.
        Guarantees isolation: called before ingesting sessions for a new benchmark question.
        """
        pass

    @abstractmethod
    def process_session(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        current_timestamp: float
    ) -> Dict[str, Any]:
        """
        Ingest conversation history turns for a session while respecting total_budget.

        Args:
            conversation_history: List of turn dicts, e.g. [{"role": "user", "content": "...", "timestamp": ...}].
            session_id: Stable session identifier.
            current_timestamp: Timestamp of the session interaction.

        Returns:
            Dictionary with session processing metrics (e.g. used_tokens, stored_count).
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Any]:
        """
        Retrieve relevant memory evidence for a query.

        Returns:
            A list of retrieved items. Each item may be:
            - A RetrievedEvidence instance
            - A MemoryItem instance
            - A tuple of (item, score)
            - A dictionary containing 'content'
            - A plain string

            The shared context assembler normalizes all formats transparently.
        """
        pass

    @abstractmethod
    def get_persisted_memories(self) -> List[Any]:
        """
        Return all memories currently consuming persistent storage budget.
        Used by the evaluation harness to measure persistent memory utilization.
        """
        pass

    def get_storage_usage(self) -> int:
        """
        Return total tokens consumed by persistent memories under the strategy's tokenizer.
        """
        total = 0
        for m in self.get_persisted_memories():
            if hasattr(m, "token_cost") and m.token_cost is not None:
                total += m.token_cost
            elif isinstance(m, dict) and "token_cost" in m:
                total += m["token_cost"]
            else:
                content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else str(m))
                total += self.tokenizer_fn(content)
        return total

    def get_stats(self) -> Dict[str, Any]:
        """
        Optional diagnostic statistics for experiment logging and debugging.
        """
        return {
            "strategy": self.name,
            "total_budget": self.total_budget,
            "storage_tokens_used": self.get_storage_usage(),
            "stored_items_count": len(self.get_persisted_memories()),
        }


# ============================================================================
# Strategy Registry
# ============================================================================

STRATEGY_REGISTRY: Dict[str, Callable[..., BaseMemoryStrategy]] = {}


def register_strategy(name: str) -> Callable[[Any], Any]:
    """
    Decorator to register a memory strategy class or factory by name.
    Example:
        @register_strategy("my_custom_memory")
        class MyCustomMemory(BaseMemoryStrategy):
            ...
    """
    def decorator(cls_or_fn: Any) -> Any:
        STRATEGY_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator


def get_strategy(
    name: str,
    budget: int = 500,
    tokenizer_fn: Optional[Callable[[str], int]] = None,
    **kwargs: Any
) -> BaseMemoryStrategy:
    """
    Instantiate a registered memory strategy by name.
    Gracefully inspects parameters so that strategies not accepting arbitrary kwargs
    (such as admission_gate or config) still instantiate without TypeError.
    """
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise KeyError(
            f"Strategy '{name}' is not registered. Available strategies: {available}. "
            f"Register your strategy using @register_strategy('{name}')."
        )
    factory = STRATEGY_REGISTRY[name]

    call_kwargs = dict(total_budget=budget, tokenizer_fn=tokenizer_fn, **kwargs)

    # 1. Try direct call with full kwargs
    try:
        return factory(**call_kwargs)
    except TypeError:
        pass

    # 2. Filter kwargs against constructor signature if inspectable
    try:
        sig = inspect.signature(factory)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_var_keyword:
            filtered = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
            if "name" in sig.parameters and "name" not in filtered:
                filtered["name"] = name
            return factory(**filtered)
    except (ValueError, TypeError):
        pass

    # 3. Fallback attempts
    try:
        return factory(name=name, total_budget=budget, tokenizer_fn=tokenizer_fn)
    except TypeError:
        return factory(total_budget=budget, tokenizer_fn=tokenizer_fn)


def list_registered_strategies() -> List[str]:
    """Returns list of all currently registered memory strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())


def load_strategy_module(path_or_module_name: str) -> None:
    """
    Dynamically loads and executes a Python module from a file path or module name.
    Any @register_strategy decorators in the module will execute and register strategies.
    """
    p = Path(path_or_module_name)
    if p.exists() and p.suffix == ".py":
        module_name = p.stem
        # If already imported under this name, reload or return
        if module_name in sys.modules:
            module = sys.modules[module_name]
            importlib.reload(module)
            return
        spec = importlib.util.spec_from_file_location(module_name, str(p.resolve()))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return
    # Fallback to standard module import
    importlib.import_module(path_or_module_name)


def auto_discover_strategies(search_dirs: Sequence[str] = ("strategies",)) -> List[str]:
    """
    Auto-discovers and imports external strategy modules matching:
    - strategies/*.py
    - strategy_*.py (excluding test files, memory_strategies.py, and memory_interface.py)
    Returns list of discovered module paths.
    """
    discovered: List[str] = []

    # 1. Check dedicated strategies/ directory if present
    for sdir in search_dirs:
        p_dir = Path(sdir)
        if p_dir.is_dir():
            for py_file in p_dir.glob("*.py"):
                if py_file.name.startswith("__") or py_file.name.startswith("test_"):
                    continue
                try:
                    load_strategy_module(str(py_file))
                    discovered.append(str(py_file))
                except Exception:
                    pass

    # 2. Check root strategy_*.py
    for py_file in Path(".").glob("strategy_*.py"):
        if py_file.name.startswith("test_") or py_file.name in ("memory_strategies.py", "memory_interface.py"):
            continue
        try:
            load_strategy_module(str(py_file))
            if str(py_file) not in discovered:
                discovered.append(str(py_file))
        except Exception:
            pass

    return discovered


# ============================================================================
# Reference Mock Strategy for Tests & Teammate Onboarding
# ============================================================================

@register_strategy("mock")
class MockMemoryStrategy(BaseMemoryStrategy):
    """
    Minimal reference memory strategy for automated tests and developer onboarding.
    Does NOT require episodic/semantic partitioning or complex candidate pipelines.
    Simply stores incoming conversation turns up to the token budget and retrieves
    by keyword overlap.
    """

    def __init__(
        self,
        total_budget: int = 500,
        tokenizer_fn: Optional[Callable[[str], int]] = None,
        **kwargs: Any
    ):
        super().__init__(name="mock", total_budget=total_budget, tokenizer_fn=tokenizer_fn, **kwargs)
        self.memories: List[RetrievedEvidence] = []

    def reset(self) -> None:
        self.memories.clear()

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

            # Assistant chit-chat filtering
            if role == "assistant" and len(content.split()) < 5:
                continue

            evidence = RetrievedEvidence(
                id=f"{session_id}_turn_{idx}",
                content=content,
                score=1.0,
                timestamp=ts,
                lifecycle_status="current",
                metadata={"session_id": session_id, "role": role}
            )
            self.memories.append(evidence)

        # Truncate any single item that exceeds total_budget
        for m in self.memories:
            cost = self.tokenizer_fn(m.content)
            if cost > self.total_budget and self.total_budget > 0:
                words = m.content.split()
                ratio = self.total_budget / max(1, cost)
                target_words = max(1, int(len(words) * ratio * 0.9))
                m.content = " ".join(words[:target_words])

        # Enforce budget cap
        while self.get_storage_usage() > self.total_budget and self.memories:
            self.memories.pop(0)  # Evict oldest first

        return {
            "strategy": self.name,
            "session_id": session_id,
            "stored_count": len(self.memories),
            "storage_tokens_used": self.get_storage_usage()
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[RetrievedEvidence, float]]:
        query_words = set(query.lower().split())
        scored: List[Tuple[RetrievedEvidence, float]] = []

        for m in self.memories:
            mem_words = set(m.content.lower().split())
            overlap = len(query_words & mem_words)
            score = overlap / max(1, len(query_words))
            scored.append((m, round(score, 3)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_persisted_memories(self) -> List[RetrievedEvidence]:
        return list(self.memories)
