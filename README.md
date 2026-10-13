# Adaptive Long-Term Memory for Language Models

**Dynamic Memory-Budget Allocation under Evolving Information**

---

## 1. Research Objective

This research investigates whether dynamically allocating a fixed persistent-memory budget between episodic and semantic memory retains more useful information than fixed allocation strategies, particularly in multi-session environments where information evolves or changes over time.

Instead of treating external memory as an unbounded store or applying static 50/50 episodic/semantic splits, this system models memory under a hard token budget constraint $B$ and continuously adapts the capacity allocated to:
1. **Episodic Memory**: Detailed, contextual records of recent interactions, tasks, and queries.
2. **Semantic Memory**: Compact, durable knowledge, consolidated user preferences, and stable facts.

---

## 2. Complete Pipeline Architecture (Phases 1, 2, & 3)

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
[ 5. Adaptive Storage Budget Allocation & Maintenance ]
  - Demand-weighted dynamic budget split within configurable share bounds
  - Eviction of expired/superseded items to archive pool
  - Knapsack utility-density selection (U(m) / token_cost(m))
  - Hard constraint: active_storage_tokens <= storage_budget
                 │
                 ▼
[ 6. Query Processing & Dual-Store Retrieval (Phase 3) ]
  - Query intent classification: current state vs historical evolution
  - Multi-factor ranking: relevance, utility, recency, and stability
  - Strict temporal validity: NEVER treat superseded or expired facts as current
  - Surface superseded memories only when query is historical
                 │
                 ▼
[ 7. Prompt-Context Assembly (Phase 3) ]
  - Deduplicated memory formatting with explicit lifecycle provenance tags:
    [CURRENT], [HISTORICAL / SUPERSEDED], [TEMPORARY]
  - Strict enforcement: prompt_context_tokens <= max_context_tokens
  - Prompt-context budget is strictly decoupled from persistent storage budget
                 │
                 ▼
[ 8. Model Adapter & LLM Response Generation (Phase 3) ]
  - Local Qwen GGUF inference via llama-cpp-python (LlamaCppAdapter)
  - Deterministic FakeLLMAdapter for automated testing without model weights
```

---

## 3. Mathematical Foundations & Retrieval Formulation

### Utility Scoring Function

For each candidate memory $m$, the utility $U(m) \in [0, 1]$ is computed as:

$$U(m) = w_r R(m) + w_i I(m) + w_f F_{\text{norm}}(m) + w_t T(m) + w_c C(m)$$

Where:
- $\sum w = 1.0$, with $w_r, w_i, w_f, w_t, w_c \ge 0$.
- **Recency ($R \in [0, 1]$)**: Temporal freshness computed via exponential decay:
  $$R(m) = \exp\left(-\frac{t_{\text{current}} - t_{\text{last\_seen}}}{\tau}\right)$$
- **Temporal Stability ($T \in [0, 1]$)**: Conceptually separate from recency; reflects intrinsic durability/invariance over time.

### Query Retrieval Ranking

For a query $q$ and candidate memory $m$:

$$\text{Score}_{\text{retrieval}}(m, q) = w_{\text{rel}} \text{Rel}(m, q) + w_{\text{util}} U(m) + w_{\text{rec}} R(m) + w_{\text{stab}} T(m) + \text{ValidityBonus}(m, q)$$

Where:
- If $q$ is a **current state query**: any memory with status `SUPERSEDED` or `EXPIRED` is disqualified (never returned as current facts).
- If $q$ is a **historical query** (e.g. *"Where did I work previously?"*): memories with status `SUPERSEDED` or historical predecessors receive a validity bonus $+0.25$.

---

## 4. Repository Structure & Module Responsibilities

```
adaptive_mememory_management/
├── .gitignore                  # Excludes datasets, model weights, checkpoints, logs, virtualenvs
├── README.md                   # Research specifications, math formulation, and usage guide
├── memory_core.py              # Phase 1: schemas, extraction, scoring, allocation, storage
├── memory_lifecycle.py         # Phase 2: deduplication, conflicts, consolidation, transitions, auditing
├── model_adapter.py            # Phase 3: BaseLLMAdapter, LlamaCppAdapter, FakeLLMAdapter
├── memory_retrieval.py         # Phase 3: query processing, dual-store retrieval, context assembly
├── run_pipeline.py             # CLI driver for running Phase 1, 2, and 3 pipelines
├── test_memory_core.py         # Phase 1 unit and integration test suite (11 tests)
├── test_memory_lifecycle.py    # Phase 2 lifecycle and maintenance test suite (8 tests)
├── test_memory_retrieval.py    # Phase 3 retrieval, temporal validity, and context tests (10 tests)
└── outputs/                    # Output directory for structured persisted states (git-ignored)
```

### Module Responsibilities:
- `memory_core.py`: Foundational dataclasses (`MemoryItem`, `MemoryType`, `LifecycleStatus`), token cost estimation, proposition extraction, utility scoring, and adaptive budget allocation.
- `memory_lifecycle.py`: Multi-session `AdaptiveMemoryManager`, soft deduplication, conflict resolution, episodic-to-semantic consolidation with historical evidence tracking, lifecycle transitions, and audit logging.
- `model_adapter.py`: Decoupled LLM interface (`BaseLLMAdapter`), local GGUF model loader using `llama-cpp-python` (`LlamaCppAdapter`), and mock generator (`FakeLLMAdapter`) for fast deterministic unit tests.
- `memory_retrieval.py`: Query intent detector (`is_historical_query`), multi-factor dual-store retrieval (`retrieve_memories`), prompt-context assembler with strict token budget enforcement (`assemble_memory_context`), and end-to-end runner (`run_phase3_pipeline`).

---

## 5. Execution Instructions

### Running Tests

Execute all 29 unit and integration tests across Phases 1, 2, and 3:
```bash
python -m unittest discover -v -s . -p "test_*.py"
```

Or run individual test modules:
```bash
python -m unittest -v test_memory_core.py
python -m unittest -v test_memory_lifecycle.py
python -m unittest -v test_memory_retrieval.py
```

### Running the Pipeline

**Phase 3 with Mock LLM Adapter (Fast Automated Mode)**:
```bash
python run_pipeline.py --phase 3 --use-fake-llm --query "Where do I work now?"
```

**Phase 3 Manual Smoke-Test with Local Qwen GGUF Model**:
```bash
python run_pipeline.py \
  --phase 3 \
  --model-path "C:\Users\Varun S\.ollama\models\blobs\sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6" \
  --query "Where do I work now?"
```

**Historical Query Smoke-Test with Local Qwen GGUF Model**:
```bash
python run_pipeline.py \
  --phase 3 \
  --model-path "C:\Users\Varun S\.ollama\models\blobs\sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6" \
  --query "Where did I work previously before DeepMind?"
```

---

## 6. Research Constraints & Next Phases

- **Phase 1, 2, & 3 Scope**: Verified candidate extraction, soft deduplication, temporal conflict resolution, consolidation with historical provenance, dynamic allocation, audit trail logging, query retrieval, prompt context assembly, and local Qwen inference.
- **Phase 4**: Controlled empirical benchmarking on LongMemEval against flat-vector and static-allocation baselines to evaluate retention under evolving multi-session information.
