# Adaptive Long-Term Memory for Language Models

**Dynamic Memory-Budget Allocation under Evolving Information**

---

## 1. Research Objective

This research investigates whether dynamically allocating a fixed persistent-memory budget between episodic and semantic memory retains more useful information than fixed allocation strategies, particularly in multi-session environments where information evolves or changes over time.

Instead of treating external memory as an unbounded store or applying static 50/50 episodic/semantic splits, this system models memory under a hard token budget constraint $B$ and continuously adapts the capacity allocated to:
1. **Episodic Memory**: Detailed, contextual records of recent interactions, tasks, and queries.
2. **Semantic Memory**: Compact, durable knowledge, consolidated user preferences, and stable facts.

---

## 2. Pipeline Architecture (Phases 1 & 2)

```
Multi-Session Conversation History
                 │
                 ▼
[ 1. Memory Candidate Extraction & Soft Deduplication ]
  - Turn proposition segmentation & pleasantry/filler filtering
  - Semantic similarity matching (Jaccard + Gestalt sequence matching)
  - Deduplication, frequency tracking, and alias linking
  - Comprehensive source provenance logging (turns, roles, timestamps)
                 │
                 ▼
[ 2. Conflict Detection & Temporal Superseding ]
  - Slot-based entity/attribute extraction (e.g. employment, preferences)
  - Temporal ordering: newer facts supersede older facts (ACTIVE vs SUPERSEDED)
  - Historical evidence preservation: predecessor links retained for retroactive queries
                 │
                 ▼
[ 3. Episodic-to-Semantic Consolidation ]
  - High-frequency / high-stability episodic memories promoted to semantic knowledge
  - Compression of repeated interactions into compact representations
  - Source observations tracked in metadata['source_evidence']
                 │
                 ▼
[ 4. Utility Scoring & Lifecycle Evaluation ]
  - Recency (R) exponential time-decay
  - Importance (I), Frequency saturation (F_norm), Temporal Stability (T), Consistency (C)
  - Justified status transitions: ACTIVE, STABLE, TEMPORARY, SUPERSEDED, EXPIRED
                 │
                 ▼
[ 5. Adaptive Budget Allocation & Eviction Maintenance ]
  - Demand-weighted dynamic budget split within configurable share bounds
  - Eviction of expired/superseded items to archive pool
  - Knapsack utility-density selection (U(m) / token_cost(m))
  - Hard constraint: active_tokens <= total_budget
                 │
                 ▼
[ 6. Auditing & Structured Local Persistence ]
  - Every merge, consolidation, superseding, transition, and eviction recorded in AuditLog
  - Local JSON state serialization (active stores, archive, summary, audit log)
```

---

## 3. Mathematical Foundations

### Utility Scoring Function

For each candidate memory $m$, the utility $U(m) \in [0, 1]$ is computed as:

$$U(m) = w_r R(m) + w_i I(m) + w_f F_{\text{norm}}(m) + w_t T(m) + w_c C(m)$$

Where:
- $\sum w = 1.0$, with $w_r, w_i, w_f, w_t, w_c \ge 0$.
- **Recency ($R \in [0, 1]$)**: Temporal freshness computed via exponential decay:
  $$R(m) = \exp\left(-\frac{t_{\text{current}} - t_{\text{last\_seen}}}{\tau}\right)$$
  (default half-life $\tau = 86,400\text{ s}$ / 24 hours).
- **Importance ($I \in [0, 1]$)**: Intrinsic heuristic significance of the memory proposition.
- **Frequency ($F_{\text{norm}} \in [0, 1]$)**: Logarithmically saturated occurrence frequency:
  $$F_{\text{norm}}(m) = \min\left(1.0, \frac{\log_2(1 + F)}{\log_2(11)}\right)$$
- **Temporal Stability ($T \in [0, 1]$)**: **Conceptually distinct from recency.** Indicates the permanence/timelessness of the knowledge. Enduring facts (e.g., *"I work at DeepMind"*, *"I prefer Python"*) receive high stability ($T \approx 0.9$), whereas transient states (e.g., *"I am debugging this error right now"*) receive low stability ($T \approx 0.2$), regardless of when they occurred.
- **Consistency / Confidence ($C \in [0, 1]$)**: Confidence score penalizing contradictions or uncertain claims.

### Adaptive Allocation & Budget Enforcement

Given a total storage budget $B$ in tokens:
1. **Demand Estimation**:
   $$\text{Demand}(M) = \sum_{m \in \mathcal{C}_M} U(m) \times \max(1, \text{cost}(m))$$
   where $M \in \{\text{episodic}, \text{semantic}\}$.
2. **Share Clamping**:
   $$s_{\text{episodic}} = \text{clip}\left(\frac{\text{Demand}(\text{episodic})}{\text{Demand}(\text{episodic}) + \text{Demand}(\text{semantic})}, s_{\min}, s_{\max}\right)$$
3. **Discrete Budget Assignment**:
   $$B_{\text{episodic}} = \lfloor s_{\text{episodic}} \times B \rceil, \quad B_{\text{semantic}} = B - B_{\text{episodic}}$$
   guaranteeing $B_{\text{episodic}} + B_{\text{semantic}} \le B$ at all times.

---

## 4. Repository Structure & Module Responsibilities

```
adaptive_mememory_management/
├── .gitignore                  # Excludes datasets, model weights, checkpoints, logs, virtualenvs
├── README.md                   # Research specifications, math formulation, and usage guide
├── memory_core.py              # Phase 1: schemas, extraction, scoring, allocation, storage
├── memory_lifecycle.py         # Phase 2: deduplication, conflicts, consolidation, transitions, auditing
├── run_pipeline.py             # CLI driver for running Phase 1 & Phase 2 pipelines
├── test_memory_core.py         # Phase 1 unit and integration test suite (11 tests)
├── test_memory_lifecycle.py    # Phase 2 lifecycle and maintenance test suite (8 tests)
└── outputs/                    # Output directory for structured persisted states (git-ignored)
```

### Module Responsibilities:
- `memory_core.py`:
  - `MemoryItem` & Enums: Core memory representation across all phases.
  - `ScoringWeights`: Parameter container with sum-to-1 validation.
  - `AllocationConfig`: Budget bounds and share constraints.
  - `extract_memories(...)`: Candidate parsing, filler filtering, deduplication, and provenance tracking.
  - `score_memories(...)`: Heuristic scoring implementing $U(m)$.
  - `allocate_budget(...)`: Dynamic episodic vs. semantic budget allocation.
  - `select_memories(...)`: Budget enforcement and greedy knapsack selection.
  - `LocalJsonMemoryStore`: Decoupled persistence interface saving and loading structured JSON.
  - `run_phase1_pipeline(...)`: End-to-end integration orchestrator for Phase 1.
- `memory_lifecycle.py`:
  - `deduplicate_semantically(...)`: Soft deduplication combining Jaccard and sequence matching.
  - `detect_conflicts_and_supersede(...)`: Temporal conflict resolution marking older items `SUPERSEDED` and linking predecessors.
  - `consolidate_episodic_memories(...)`: Merging repeated episodic items into semantic knowledge while retaining `metadata['source_evidence']`.
  - `evaluate_lifecycle_transitions(...)`: Status state machine (`ACTIVE`, `STABLE`, `TEMPORARY`, `SUPERSEDED`, `EXPIRED`).
  - `maintain_memory_budget(...)`: Eviction of expired/superseded items and budget maintenance.
  - `AdaptiveMemoryManager`: Stateful multi-session manager with complete audit trail persistence.
  - `run_phase2_pipeline(...)`: High-level procedural runner for Phase 2.

---

## 5. Execution Instructions

### Running Tests

Execute all tests across Phase 1 and Phase 2:
```bash
python -m unittest discover -v -s . -p "test_*.py"
```

Or run modules individually:
```bash
python -m unittest -v test_memory_core.py
python -m unittest -v test_memory_lifecycle.py
```

### Running the Pipeline

Run Phase 2 (multi-session lifecycle demo with evolving information and audit trails):
```bash
python run_pipeline.py --phase 2 --budget 120
```

Run Phase 1 (single-session baseline allocation):
```bash
python run_pipeline.py --phase 1 --budget 120
```

Run with custom conversation JSON, weights, and budget:
```bash
python run_pipeline.py \
  --phase 2 \
  --input path/to/conversation.json \
  --output outputs/my_state.json \
  --budget 200 \
  --weights 0.25 0.25 0.20 0.15 0.15 \
  --seed 42
```

---

## 6. Research Constraints & Next Phases

- **Phase 1 & Phase 2 Scope**: Rule-based extraction, soft deduplication, temporal conflict resolution, consolidation with historical provenance, dynamic allocation, and audit trail logging.
- **Subsequent Phases**:
  - *Phase 3*: Budgeted retrieval and SLM/LLM temporal validity verification against queries.
  - *Phase 4*: Controlled empirical benchmarking on LongMemEval against flat-vector and static-allocation baselines.
