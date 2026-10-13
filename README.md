# Adaptive Long-Term Memory for Language Models

**Dynamic Memory-Budget Allocation under Evolving Information**

---

## 1. Research Objective

This research investigates whether dynamically allocating a fixed persistent-memory budget between episodic and semantic memory retains more useful information than fixed allocation strategies, particularly in multi-session environments where information evolves or changes over time.

Instead of treating external memory as an unbounded store or applying static 50/50 episodic/semantic splits, this system models memory under a hard token budget constraint $B$ and continuously adapts the capacity allocated to:
1. **Episodic Memory**: Detailed, contextual records of recent interactions, tasks, and queries.
2. **Semantic Memory**: Compact, durable knowledge, consolidated user preferences, and stable facts.

---

## 2. Phase 1 Pipeline Architecture

```
Conversation History
        │
        ▼
[ 1. Memory Candidate Extraction & Deduplication ]
  - Turn proposition segmentation
  - Pleasantry/filler filtering
  - Deduplication & occurrence frequency tracking
  - Source provenance logging (sessions, turns, timestamps)
        │
        ▼
[ 2. Utility Controller Scoring ]
  - Recency (R) exponential time-decay
  - Importance (I) intrinsic value
  - Frequency (F) saturation normalization
  - Temporal Stability (T) invariance over time
  - Consistency (C) confidence score
        │
        ▼
[ 3. Adaptive Budget Allocation ]
  - Estimated demand calculation per memory type
  - Clamping within configurable [min_share, max_share] bounds
  - Hard total token budget enforcement
        │
        ▼
[ 4. Memory Selection (Store-Budget Fit) ]
  - Knapsack greedy density ranking: U(m) / token_cost(m)
  - Zero-overrun guarantee per store
        │
        ▼
[ 5. Local Structured Persistence ]
  - Transparent JSON storage (stores, metadata, allocation summary)
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
├── .gitignore               # Excludes datasets, model weights, checkpoints, logs, virtualenvs
├── README.md                # Research specifications, math formulation, and usage guide
├── memory_core.py           # Core Phase 1 engine: schema, extraction, scoring, allocation, storage
├── run_pipeline.py          # Command-line driver for running Phase 1 on conversations
├── test_memory_core.py      # Comprehensive unit and integration test suite
└── outputs/                 # Output directory for structured persisted memory states (git-ignored)
```

### Module Responsibilities:
- `memory_core.py`:
  - `MemoryItem` & Enums: Complete schema definition.
  - `ScoringWeights`: Parameter container with sum-to-1 validation.
  - `AllocationConfig`: Budget bounds and share constraints.
  - `extract_memories(...)`: Candidate parsing, filler filtering, deduplication, and provenance tracking.
  - `score_memories(...)`: Heuristic scoring implementing $U(m)$.
  - `allocate_budget(...)`: Dynamic episodic vs. semantic budget allocation.
  - `select_memories(...)`: Budget enforcement and greedy knapsack selection.
  - `LocalJsonMemoryStore`: Decoupled persistence interface saving and loading structured JSON.
  - `run_phase1_pipeline(...)`: End-to-end integration orchestrator.

---

## 5. Execution Instructions

### Running Tests

Execute the unit test suite:
```bash
python -m unittest test_memory_core.py
```

All 11 tests verify:
- Schema serialization and lifecycle statuses
- Extraction, filler suppression, deduplication, and provenance
- Utility scoring, weight validation, and recency/stability independence
- Adaptive allocation bounds under extreme demand skews
- Selection strictness under tight token budgets
- JSON persistence roundtrip fidelity
- Full deterministic pipeline integration

### Running the Pipeline

Run the pipeline using default sample conversations:
```bash
python run_pipeline.py --budget 120
```

Run with custom budget, weights, and input JSON:
```bash
python run_pipeline.py \
  --input path/to/conversation.json \
  --output outputs/my_state.json \
  --budget 250 \
  --min-episodic 0.25 \
  --max-episodic 0.75 \
  --weights 0.25 0.25 0.20 0.15 0.15 \
  --seed 42
```

---

## 6. Research Constraints & Next Phases

- **Phase 1 Scope**: Focuses strictly on the local core (candidate extraction, scoring, budget allocation, knapsack selection, local JSON persistence).
- **Subsequent Phases**:
  - *Phase 2*: Memory consolidation (clustering & summarization) and state transitions (`ACTIVE` $\rightarrow$ `SUPERSEDED` upon conflicting updates).
  - *Phase 3*: Budgeted retrieval and temporal validity checking.
  - *Phase 4*: Controlled empirical benchmarking on LongMemEval against fixed-allocation and flat-vector baselines.
