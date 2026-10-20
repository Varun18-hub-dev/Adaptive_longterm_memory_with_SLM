"""
Adaptive Long-Term Memory for Language Models
Phase 5: Modular Memory Candidate Generation & Hardened Admission Pipeline

Components:
1. BaseCandidateExtractor: Abstract base class for candidate extraction.
2. HeuristicCandidateExtractor: High-speed, rule-based extractor with:
   - Multi-fact clause decomposition
   - Conversational filler & distractor suppression
   - Uncertainty & hedging detection (confidence discounting)
   - Negation & constraint detection
   - Temporal anchor & historical fact recognition
   - MemoryType (semantic vs episodic) classification
3. SLMCandidateExtractor: Model-guided candidate extraction using BaseLLMAdapter:
   - Structured JSON prompt schema
   - Graceful heuristic fallback if JSON is unparseable
4. HybridCandidateExtractor: Cost-aware extractor that runs fast heuristics first
   and selectively invokes SLM only when high clause complexity or ambiguity is detected.
5. CandidateAdmissionGate: Admission gate that filters, gates, and audits candidates:
   - Usefulness & informative threshold
   - Confidence & uncertainty gating
   - Redundancy / soft deduplication check
   - Contradiction & temporal conflict resolution (superseding stale facts)
   - Budget-aware storage management
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    AuditAction,
    AuditLogEntry,
    compute_text_similarity,
    deduplicate_semantically,
    detect_conflicts_and_supersede,
    extract_fact_slot,
)
from model_adapter import BaseLLMAdapter, heuristic_bpe_token_count


# ============================================================================
# 1. Linguistic & Heuristic Helpers
# ============================================================================

REACTION_PATTERNS = [
    r"^(that'?s|it'?s|this is)\s+(really|very|so)?\s*(cool|interesting|helpful|great|awesome|nice|good|amazing|fascinating|wonderful|fine)\b",
    r"^(thanks|thank you)\b",
    r"^(sounds|looks)\s+(great|good|cool|awesome|nice|fine)\b",
    r"^(sure thing|no problem|you'?re welcome|anytime|my pleasure)\b",
    r"^(make it|can you make it)\s+(longer|shorter|better)\b",
    r"^(hello|hi|hey|good morning|good afternoon|good evening)\b",
    r"^(congratulations|congrats|welcome)\b",
    r"^(how can i help|what can i do for you)\b",
]

UNCERTAINTY_PATTERNS = [
    r"\b(not sure|unsure|uncertain|maybe|perhaps|might|i think maybe|possibly|tentatively|not certain)\b",
    r"\b(i suppose|i guess|could be|not confident)\b",
]

NEGATION_PATTERNS = [
    r"\b(never|don't|do not|cannot|can't|won't|will not|dislike|hate|refuse|stop|no longer|neither|nor)\b"
]

HISTORICAL_PATTERNS = [
    r"\b(worked at|lived in|used to|formerly|previously|was born|graduated from|from \d{4} to \d{4}|in \d{4})\b",
    r"\b(my previous|my former|back in \d{4}|last year|years ago)\b"
]


def is_filler_sentence(sent: str) -> bool:
    """
    Detects trivial conversational pleasantries, formatting artifacts, and fillers
    that provide zero durable personal memory value.
    """
    cleaned_sent = re.sub(r"^\s*(\d+[\.\)]|\*|\-|\+|\[|\#+)\s*", "", sent).strip()
    cleaned = re.sub(r"[^\w\s]", "", cleaned_sent.lower()).strip()
    words = cleaned.split()

    if not words:
        return True

    # Standalone numbers, roman numerals, single letters, or short fragments without factual anchor
    if len(words) < 3:
        core_predicates = {
            "work", "worked", "live", "lived", "name", "prefer", "like",
            "graduated", "degree", "use", "code", "born", "joined", "moved"
        }
        if not any(kw in cleaned for kw in core_predicates):
            return True

    if any(re.search(p, cleaned) for p in REACTION_PATTERNS):
        return True

    return False


def decompose_multi_fact_sentence(sentence: str) -> List[str]:
    """
    Decomposes compound sentences with multiple coordinated clauses into atomic propositions.
    Example: 'I live in Seattle, work at Amazon, and prefer using vim.'
    -> ['I live in Seattle', 'I work at Amazon', 'I prefer using vim']
    """
    s = sentence.strip()
    user_subj_match = re.match(r"^(i|we|my\s+\w+)\s+(.+)$", s, re.IGNORECASE)
    if not user_subj_match:
        return [s]

    subj = user_subj_match.group(1)
    body = user_subj_match.group(2)

    # Split by coordinating conjunctions
    parts = re.split(r",\s*and\s+|\s+and\s+|,\s*", body)
    if len(parts) <= 1:
        return [s]

    verb_indicators = (
        r"\b(am|was|live|lived|work|worked|prefer|like|use|used|joined|moved|"
        r"graduated|study|studied|speak|have|had|drink|eat|code|program)\b"
    )

    propositions: List[str] = []
    for p in parts:
        p_clean = p.strip()
        if not p_clean:
            continue
        if re.search(verb_indicators, p_clean, re.IGNORECASE):
            if re.match(r"^(i|we|my)\b", p_clean, re.IGNORECASE):
                propositions.append(p_clean)
            else:
                propositions.append(f"{subj} {p_clean}")

    if len(propositions) >= 2:
        return propositions
    return [s]


def detect_uncertainty(text: str) -> bool:
    """Detects whether an utterance expresses doubt, tentativeness, or hedging."""
    return any(re.search(p, text.lower()) for p in UNCERTAINTY_PATTERNS)


def detect_negation(text: str) -> bool:
    """Detects whether an utterance asserts a negative preference or constraint."""
    return any(re.search(p, text.lower()) for p in NEGATION_PATTERNS)


def detect_historical_intent(text: str) -> bool:
    """Detects whether an utterance refers to a past or historical fact."""
    return any(re.search(p, text.lower()) for p in HISTORICAL_PATTERNS)


# ============================================================================
# 2. Candidate Extractor Interfaces & Implementations
# ============================================================================

class BaseCandidateExtractor(ABC):
    """Abstract interface for extracting candidate memories from conversation sessions."""

    @abstractmethod
    def extract_candidates(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str = "session_0",
        current_timestamp: Optional[float] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ) -> List[MemoryItem]:
        """Extract candidate MemoryItem objects from conversation history turns."""
        pass


class HeuristicCandidateExtractor(BaseCandidateExtractor):
    """
    Hardened heuristic candidate extractor:
    - Splits compound sentences into atomic propositions.
    - Suppresses conversational pleasantries and filler.
    - Detects uncertainty to avoid false stability.
    - Preserves negative constraints and historical temporal evidence.
    - Assigns MemoryType (semantic vs episodic), importance, and stability.
    """

    def __init__(self, tokenizer_fn: Optional[Callable[[str], int]] = None):
        self.tokenizer_fn = tokenizer_fn or estimate_token_cost

    @staticmethod
    def _is_admissible_assistant_turn(text: str) -> bool:
        """
        Policy for assistant turns:
        Generic assistant advice, code explanations, pleasantries, and commentary
        must never pollute personal memory stores.
        Assistant turns are only admitted if they contain an identifiable entity-attribute fact slot
        or an explicit user-referential fact assertion (e.g. 'You work at DeepMind', 'Your dog's name is Luna').
        """
        # 1. Check if it matches a structured entity-attribute fact slot
        if extract_fact_slot(text) is not None:
            return True

        text_lower = text.lower()
        # 2. Check for explicit user-referential assertion
        user_ref = re.search(r"\b(you|your|user)\b", text_lower)
        if user_ref:
            user_fact_predicates = [
                r"\b(work at|live in|moved to|prefer|like|use|own|drive|joined|graduated from)\b",
                r"\b(name is|dog is|cat is|pet is|car is|job is|employer is|title is)\b",
                r"\b(favorite|preference|degree|budget|hometown)\b"
            ]
            if any(re.search(pat, text_lower) for pat in user_fact_predicates):
                return True

        return False

    def _classify_candidate(self, text: str) -> Tuple[MemoryType, float, float, float, LifecycleStatus, Dict[str, Any]]:
        text_lower = text.lower()
        meta: Dict[str, Any] = {}

        is_uncertain = detect_uncertainty(text)
        is_neg = detect_negation(text)
        is_hist = detect_historical_intent(text)

        meta["uncertainty_detected"] = is_uncertain
        meta["is_negation"] = is_neg
        meta["is_historical_fact"] = is_hist

        semantic_patterns = [
            r"\bi am\b", r"\bi like\b", r"\bi prefer\b", r"\bmy name is\b",
            r"\bmy job is\b", r"\bi work at\b", r"\balways\b", r"\bnever\b",
            r"\buser prefers\b", r"\blives in\b", r"\bspeaks\b", r"\bdefinition\b",
            r"\bi (graduated|studied|bought|lived|live|moved|own|drive|commute)\b",
            r"\bmy (degree|commute|car|house|family|wife|husband|dog|cat|project|hobby|budget|laptop|computer)\b"
        ]

        episodic_patterns = [
            r"\btoday\b", r"\byesterday\b", r"\bnow\b", r"\bjust\b",
            r"\bcan you\b", r"\bhelp me\b", r"\bdebug\b", r"\bthis error\b",
            r"\brun this\b", r"\blast time\b", r"\bcurrent task\b"
        ]

        is_sem = any(re.search(pat, text_lower) for pat in semantic_patterns)
        is_epi = any(re.search(pat, text_lower) for pat in episodic_patterns)

        if is_sem and not is_epi:
            mem_type = MemoryType.SEMANTIC
            importance = 0.85 if not is_uncertain else 0.50
            stability = 0.90 if not is_uncertain else 0.40
        elif is_epi and not is_sem:
            mem_type = MemoryType.EPISODIC
            importance = 0.60
            stability = 0.30
        else:
            if len(text.split()) > 15:
                mem_type = MemoryType.EPISODIC
                importance = 0.50
                stability = 0.40
            else:
                mem_type = MemoryType.SEMANTIC
                importance = 0.65
                stability = 0.70

        if is_uncertain:
            confidence = 0.45
            status = LifecycleStatus.ACTIVE
        else:
            confidence = 1.0
            status = LifecycleStatus.STABLE if stability >= 0.8 else LifecycleStatus.ACTIVE

        return mem_type, importance, stability, confidence, status, meta

    def extract_candidates(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str = "session_0",
        current_timestamp: Optional[float] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ) -> List[MemoryItem]:
        if current_timestamp is None:
            current_timestamp = time.time()
        fn_tokens = tokenizer_fn or self.tokenizer_fn

        candidates_by_content: Dict[str, MemoryItem] = {}

        for turn_idx, turn in enumerate(conversation_history):
            content = turn.get("content", "").strip()
            role = turn.get("role", "unknown")
            turn_timestamp = turn.get("timestamp", current_timestamp)

            if not content:
                continue

            # Split turn into sentences
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", content) if s.strip()]
            if not raw_sentences:
                raw_sentences = [content]

            # Decompose multi-fact sentences into atomic propositions
            atomic_propositions: List[str] = []
            for raw_s in raw_sentences:
                cleaned_prefix = re.sub(r"^\s*(\d+[\.\)]|\*|\-|\+|\[|\#+)\s*", "", raw_s).strip()
                if not cleaned_prefix or is_filler_sentence(cleaned_prefix):
                    continue
                decomposed = decompose_multi_fact_sentence(cleaned_prefix)
                atomic_propositions.extend(decomposed)

            for prop in atomic_propositions:
                prop_clean = prop.strip()
                if not prop_clean or is_filler_sentence(prop_clean):
                    continue

                # Policy: Assistant turns must contain identifiable user facts or attributes
                if role == "assistant" and not self._is_admissible_assistant_turn(prop_clean):
                    continue

                normalized_key = " ".join(prop_clean.lower().split())

                if normalized_key in candidates_by_content:
                    existing = candidates_by_content[normalized_key]
                    existing.frequency += 1
                    existing.timestamp = max(existing.timestamp, turn_timestamp)
                    existing.metadata["provenance"].append({
                        "session_id": session_id,
                        "turn_index": turn_idx,
                        "role": role,
                        "timestamp": turn_timestamp
                    })
                else:
                    mem_type, imp, stab, conf, status, extra_meta = self._classify_candidate(prop_clean)
                    mem_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{session_id}:{normalized_key}"))
                    cost = fn_tokens(prop_clean)

                    meta = {
                        "normalized_key": normalized_key,
                        "provenance": [{
                            "session_id": session_id,
                            "turn_index": turn_idx,
                            "role": role,
                            "timestamp": turn_timestamp
                        }]
                    }
                    meta.update(extra_meta)

                    candidates_by_content[normalized_key] = MemoryItem(
                        id=mem_id,
                        content=prop_clean,
                        memory_type=mem_type,
                        source_session_id=session_id,
                        timestamp=turn_timestamp,
                        importance=imp,
                        recency=1.0,
                        frequency=1,
                        stability=stab,
                        consistency_confidence=conf,
                        utility_score=0.0,
                        token_cost=cost,
                        lifecycle_status=status,
                        metadata=meta
                    )

        return list(candidates_by_content.values())


class SLMCandidateExtractor(BaseCandidateExtractor):
    """
    Model-guided candidate extraction using BaseLLMAdapter.
    Extracts atomic, structured memory candidates via prompt engineering.
    Falls back gracefully to HeuristicCandidateExtractor if the model response
    is invalid or cannot be parsed.
    """

    EXTRACTION_SYSTEM_PROMPT = (
        "You are an expert memory candidate extraction module for a long-term interactive assistant.\n"
        "Given the conversation turns, extract atomic, self-contained personal facts, user preferences, "
        "and temporal tasks.\n"
        "Do not extract conversational pleasantries, reactions, or generic assistant advice.\n"
        "Return ONLY a JSON array of candidate objects with this schema:\n"
        "[\n"
        "  {\n"
        "    \"content\": \"atomic fact string\",\n"
        "    \"memory_type\": \"semantic\" | \"episodic\",\n"
        "    \"importance\": 0.1 - 1.0,\n"
        "    \"temporal_stability\": 0.1 - 1.0,\n"
        "    \"confidence\": 0.1 - 1.0,\n"
        "    \"is_uncertain\": false | true,\n"
        "    \"is_historical\": false | true\n"
        "  }\n"
        "]"
    )

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        fallback_extractor: Optional[BaseCandidateExtractor] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ):
        self.llm_adapter = llm_adapter
        self.fallback = fallback_extractor or HeuristicCandidateExtractor()
        self.tokenizer_fn = tokenizer_fn or getattr(llm_adapter, "count_tokens", heuristic_bpe_token_count)
        self.last_fallback: bool = False
        self.last_error: Optional[str] = None

    def extract_candidates(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str = "session_0",
        current_timestamp: Optional[float] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ) -> List[MemoryItem]:
        if current_timestamp is None:
            current_timestamp = time.time()
        fn_tokens = tokenizer_fn or self.tokenizer_fn

        # Format conversation transcript for model
        transcript_lines = []
        for t in conversation_history:
            role = t.get("role", "user")
            content = t.get("content", "").strip()
            if content:
                transcript_lines.append(f"{role.upper()}: {content}")

        if not transcript_lines:
            return []

        user_content = "\n".join(transcript_lines)
        messages = [
            {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract memory candidates from this session:\n\n{user_content}"}
        ]

        try:
            raw_response = self.llm_adapter.generate_chat(
                messages=messages,
                max_tokens=256,
                temperature=0.0
            )

            # Extract JSON block
            json_text = raw_response.strip()
            code_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", json_text, re.DOTALL)
            if code_block_match:
                json_text = code_block_match.group(1).strip()
            elif "[" in json_text and "]" in json_text:
                json_text = json_text[json_text.find("["):json_text.rfind("]") + 1].strip()

            parsed = json.loads(json_text)
            if not isinstance(parsed, list):
                raise ValueError("Parsed JSON is not a list")

            candidates: List[MemoryItem] = []
            for item in parsed:
                content_str = item.get("content", "").strip()
                if not content_str or is_filler_sentence(content_str):
                    continue

                m_type = MemoryType.EPISODIC if item.get("memory_type") == "episodic" else MemoryType.SEMANTIC
                imp = float(item.get("importance", 0.7))
                stab = float(item.get("temporal_stability", 0.8))
                conf = float(item.get("confidence", 1.0))
                is_unc = bool(item.get("is_uncertain", False))
                is_hist = bool(item.get("is_historical", False))

                if is_unc:
                    conf = min(conf, 0.45)
                    status = LifecycleStatus.ACTIVE
                else:
                    status = LifecycleStatus.STABLE if stab >= 0.8 else LifecycleStatus.ACTIVE

                mem_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{session_id}:{content_str.lower()}"))
                cost = fn_tokens(content_str)

                candidates.append(MemoryItem(
                    id=mem_id,
                    content=content_str,
                    memory_type=m_type,
                    source_session_id=session_id,
                    timestamp=current_timestamp,
                    importance=imp,
                    recency=1.0,
                    frequency=1,
                    stability=stab,
                    consistency_confidence=conf,
                    utility_score=0.0,
                    token_cost=cost,
                    lifecycle_status=status,
                    metadata={
                        "extracted_by": "slm",
                        "uncertainty_detected": is_unc,
                        "is_historical_fact": is_hist,
                        "provenance": [{
                            "session_id": session_id,
                            "role": "dialogue",
                            "timestamp": current_timestamp
                        }]
                    }
                ))

            if candidates:
                return candidates

        except Exception as exc:
            # Graceful fallback to heuristic extractor on model generation or parse failure
            self.last_fallback = True
            self.last_error = str(exc)

        return self.fallback.extract_candidates(
            conversation_history=conversation_history,
            session_id=session_id,
            current_timestamp=current_timestamp,
            tokenizer_fn=fn_tokens
        )


class HybridCandidateExtractor(BaseCandidateExtractor):
    """
    Cost-aware hybrid candidate extractor:
    Runs fast heuristics on every turn; selectively routes to SLM only when
    high clause complexity or ambiguity is detected.
    """

    def __init__(
        self,
        llm_adapter: Optional[BaseLLMAdapter] = None,
        trigger_slm_on_multi_clause: bool = False
    ):
        self.heuristic = HeuristicCandidateExtractor()
        self.slm = SLMCandidateExtractor(llm_adapter=llm_adapter) if llm_adapter else None
        self.trigger_slm = trigger_slm_on_multi_clause
        self.last_fallback: bool = False
        self.last_error: Optional[str] = None

    def extract_candidates(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str = "session_0",
        current_timestamp: Optional[float] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None
    ) -> List[MemoryItem]:
        self.last_fallback = False
        self.last_error = None

        # Fast heuristic pass
        candidates = self.heuristic.extract_candidates(
            conversation_history=conversation_history,
            session_id=session_id,
            current_timestamp=current_timestamp,
            tokenizer_fn=tokenizer_fn
        )

        # If SLM is available and multi-clause complexity trigger is active
        if self.slm and self.trigger_slm:
            has_complex_turn = any(
                len(t.get("content", "").split()) > 20 and "," in t.get("content", "")
                for t in conversation_history
            )
            if has_complex_turn:
                res = self.slm.extract_candidates(
                    conversation_history=conversation_history,
                    session_id=session_id,
                    current_timestamp=current_timestamp,
                    tokenizer_fn=tokenizer_fn
                )
                self.last_fallback = getattr(self.slm, "last_fallback", False)
                self.last_error = getattr(self.slm, "last_error", None)
                return res

        return candidates


# ============================================================================
# 3. Candidate Admission Gate
# ============================================================================

class AdmissionDecision(str, Enum):
    ADMIT = "admit"
    REJECT_FILLER = "reject_filler"
    REJECT_LOW_CONFIDENCE = "reject_low_confidence"
    REJECT_DUPLICATE = "reject_duplicate"
    SUPERSEDE_EXISTING = "supersede_existing"


@dataclass
class AdmissionResult:
    admitted_candidates: List[MemoryItem]
    rejected_candidates: List[Tuple[MemoryItem, AdmissionDecision, str]]
    audit_entries: List[AuditLogEntry]


class CandidateAdmissionGate:
    """
    Evaluates candidate memories prior to storage store admission:
    1. Usefulness & informative value check (rejects conversational noise).
    2. Confidence & uncertainty gating (penalizes uncertain statements).
    3. Redundancy check (merges duplicates rather than duplicating storage).
    4. Contradiction & temporal conflict resolution (marks old facts as SUPERSEDED,
       preserving historical links for point-in-time queries).
    5. Storage budget awareness.
    """

    def __init__(
        self,
        min_importance_threshold: float = 0.20,
        min_confidence_threshold: float = 0.30,
        dedup_similarity_threshold: float = 0.80,
    ):
        self.min_importance_threshold = min_importance_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.dedup_similarity_threshold = dedup_similarity_threshold

    def filter_quality(
        self,
        incoming_candidates: List[MemoryItem],
        audit_entries: Optional[List[AuditLogEntry]] = None,
        current_timestamp: Optional[float] = None
    ) -> Tuple[List[MemoryItem], List[Tuple[MemoryItem, AdmissionDecision, str]]]:
        """
        Quality and confidence filter:
        Filters out trivial conversational fillers, low-importance pleasantries,
        and low-confidence uncertain assertions before store partitioning.
        """
        if current_timestamp is None:
            current_timestamp = time.time()

        clean_candidates: List[MemoryItem] = []
        rejected: List[Tuple[MemoryItem, AdmissionDecision, str]] = []

        for cand in incoming_candidates:
            if is_filler_sentence(cand.content):
                dec = AdmissionDecision.REJECT_FILLER
                reason = "Trivial filler or pleasantry"
                rejected.append((cand, dec, reason))
                if audit_entries is not None:
                    audit_entries.append(AuditLogEntry(
                        id=str(uuid.uuid4()),
                        timestamp=current_timestamp,
                        action=AuditAction.REJECT_CANDIDATE,
                        target_memory_id=cand.id,
                        related_memory_ids=[],
                        reason=reason,
                        metadata={"decision": dec.value, "content": cand.content}
                    ))
                continue
            if cand.consistency_confidence < self.min_confidence_threshold:
                dec = AdmissionDecision.REJECT_LOW_CONFIDENCE
                reason = f"Confidence {cand.consistency_confidence:.2f} below threshold {self.min_confidence_threshold:.2f}"
                rejected.append((cand, dec, reason))
                if audit_entries is not None:
                    audit_entries.append(AuditLogEntry(
                        id=str(uuid.uuid4()),
                        timestamp=current_timestamp,
                        action=AuditAction.REJECT_CANDIDATE,
                        target_memory_id=cand.id,
                        related_memory_ids=[],
                        reason=reason,
                        metadata={"decision": dec.value, "content": cand.content, "confidence": cand.consistency_confidence}
                    ))
                continue
            if cand.importance < self.min_importance_threshold:
                dec = AdmissionDecision.REJECT_FILLER
                reason = f"Importance {cand.importance:.2f} below threshold {self.min_importance_threshold:.2f}"
                rejected.append((cand, dec, reason))
                if audit_entries is not None:
                    audit_entries.append(AuditLogEntry(
                        id=str(uuid.uuid4()),
                        timestamp=current_timestamp,
                        action=AuditAction.REJECT_CANDIDATE,
                        target_memory_id=cand.id,
                        related_memory_ids=[],
                        reason=reason,
                        metadata={"decision": dec.value, "content": cand.content, "importance": cand.importance}
                    ))
                continue
            clean_candidates.append(cand)

        return clean_candidates, rejected

    def evaluate_candidates(
        self,
        incoming_candidates: List[MemoryItem],
        existing_memories: List[MemoryItem],
        current_timestamp: Optional[float] = None
    ) -> AdmissionResult:
        if current_timestamp is None:
            current_timestamp = time.time()

        audit_entries: List[AuditLogEntry] = []

        # 1. Filter trivial fillers and low-importance noise
        clean_candidates, rejected = self.filter_quality(
            incoming_candidates=incoming_candidates,
            audit_entries=audit_entries,
            current_timestamp=current_timestamp
        )

        # 2. Redundancy / Soft Deduplication against existing memories
        updated_existing, distinct = deduplicate_semantically(
            incoming_memories=clean_candidates,
            existing_memories=existing_memories,
            similarity_threshold=self.dedup_similarity_threshold,
            audit_entries=audit_entries,
            current_timestamp=current_timestamp
        )

        distinct_ids = {c.id for c in distinct}
        for cand in clean_candidates:
            if cand.id not in distinct_ids:
                rejected.append((cand, AdmissionDecision.REJECT_DUPLICATE, "Soft-merged into existing memory"))

        # 3. Contradiction & Temporal Conflict Resolution
        all_combined = list(existing_memories) + list(distinct)
        detect_conflicts_and_supersede(
            all_memories=all_combined,
            audit_entries=audit_entries,
            current_timestamp=current_timestamp
        )

        admitted = distinct

        return AdmissionResult(
            admitted_candidates=admitted,
            rejected_candidates=rejected,
            audit_entries=audit_entries
        )
