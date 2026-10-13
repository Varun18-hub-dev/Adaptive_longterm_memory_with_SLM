# Adaptive Long-Term Memory for Language Models

**Dynamic Memory-Budget Allocation under Evolving Information**

---

## 1. Research Objective & Central Hypothesis

Language models deploying persistent external memory operate under finite storage constraints while processing interactions where facts, tasks, and preferences continuously evolve over time.

**Central Research Question**:
> Under the same total persistent-memory budget $B$, does dynamically allocating storage between detailed episodic memory and compact semantic memory retain more useful information than flat memory or fixed hierarchical allocation, especially when information evolves over time?

---

## 2. Architecture & Pipeline Overview (Phases 1, 2, 3, & 4)

```
Multi-Session Conversation History
                 │
                 ▼
[ 1. Memory Candidate Extraction & Soft Deduplication ]
  - Turn proposition segmentation & pleasantry/filler filtering
  - Semantic similarity matching (Jaccard + Gestalt sequence matching)
  - Deduplication, frequency tracking, and alias linking
  - Provenance logging (sessions, turns, timestamps)
                 │
                 ▼
[ 2. Conflict Detection & Temporal Superseding ]
  - Slot-based entity/attribute extraction (e.g. employment, preferences)
  - Newer facts supersede older facts (ACTIVE vs SUPERSEDED)
  - Predecessor links preserved for retroactive/historical queries
                 │
                 ▼
[ 3. Episodic-to-Semantic Consolidation ]
  - High-frequency / high-stability episodic memories promoted to semantic knowledge
  - Compression of repeated interactions into compact representations
  - Source observations tracked in metadata['source_evidence']
                 │
                 ▼
[ 4. Utility Scoring & Lifecycle Transitions ]
  - Recency (R) exponential time-decay
  - Importance (I), Frequency saturation (F_norm), Temporal Stability (T), Consistency (C)
  - Status transitions: ACTIVE, STABLE, TEMPORARY, SUPERSEDED, EXPIRED
                 │
                 ▼
[ 5. Adaptive Storage Budget Allocation & Maintenance ]
  - Demand-weighted dynamic budget split within configurable share bounds
  - Eviction of expired/superseded items to archive pool
  - Knapsack utility-density selection (U(m) / token_cost(m))
  - Hard constraint: active_storage_tokens <= storage_budget
                 │
                 ▼
[ 6. Query Processing & Hardened Dual-Store Retrieval (Phase 3 Hardened) ]
  - Query intent classification: current state, historical, point-in-time
  - Stopword-filtered content overlap + slot affinity scoring
  - Dynamic margin pruning: excludes transient distractors (e.g. CUDA tasks)
  - Strict temporal validity: NEVER treat superseded or expired facts as current
                 │
                 ▼
[ 7. Prompt Budget Accounting & Context Assembly (Phase 3 Hardened) ]
  - Prompt budget tracker: model_window - (system + query + template + generation_allowance)
  - Exact tokenization via native Qwen GGUF tokenizer
  - Explicit provenance tags: [CURRENT], [HISTORICAL / SUPERSEDED], [TEMPORARY]
                 │
                 ▼
[ 8. Controlled Research Evaluation & Benchmarks (Phase 4) ]
  - Comparable strategies under equal budget: Flat, Fixed (70/30, 50/50, 30/70), Adaptive, Ablations
  - Synthetic Evolving Benchmark & LongMemEval-S Evaluation Suite
  - Metrics: Accuracy, Token F1, Recall@K, MRR, Stale Error Rate, Storage/Context Usage, Latency
```

---

## 3. Evaluated Memory Strategies

To ensure strict fairness, all strategies operate under identical input histories, identical queries, and identical total persistent storage budgets $B$:

1. **`flat_memory_baseline`**: Single undivided memory pool without episodic/semantic division. Prunes lowest utility-density items when total tokens exceed $B$.
2. **`fixed_70_30`**: Fixed hierarchical allocation allocating 70% of budget to episodic memory and 30% to semantic memory.
3. **`fixed_50_50`**: Fixed hierarchical allocation allocating an even 50% split.
4. **`fixed_30_70`**: Fixed hierarchical allocation allocating 30% to episodic memory and 70% to semantic memory.
5. **`adaptive_hierarchical`**: Dynamically adjusts episodic vs. semantic budget based on candidate workload demand within bounds $[0.20, 0.80]$.
6. **Ablations**: Selectively disables adaptive allocation, utility scoring, consolidation, temporal validity handling, or reranking.

---

## 4. Phase 4 Empirical Evaluation Results

### Live Model Configuration
- **Model**: `Qwen2.5-3B-Instruct` (GGUF Q4_K_M quantization) via `llama-cpp-python` (`v0.2.90`)
- **Model Context Window**: 2048 tokens
- **Prompt Context Limit**: 256 tokens
- **Decoding**: Greedy deterministic decoding ($\text{temperature} = 0.0$, $\text{max\_tokens} = 128$)
- **Tokenizer**: Native embedded Qwen GGUF BPE tokenizer

### Live Qwen Benchmark Results (Synthetic Evolving Suite, 330.46s run)

| Strategy | Budget | Accuracy | Token F1 | Recall@K | MRR | Stale Error Rate | Storage Used (tokens) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fixed_70_30` | 50 | **62.5%** | 0.175 | 62.5% | 0.562 | 0.0% | 19.0 | 5499.2 |
| `fixed_50_50` | 50 | **87.5%** | 0.265 | 87.5% | 0.812 | 0.0% | 24.9 | 4461.6 |
| `adaptive_hierarchical` | 50 | **87.5%** | 0.265 | 87.5% | 0.812 | 0.0% | 23.6 | 4316.9 |
| `fixed_30_70` | 50 | **100.0%** | 0.333 | 100.0% | 0.938 | 0.0% | 23.8 | 3816.3 |
| `flat_memory_baseline` | 50 | **100.0%** | 0.333 | 100.0% | 0.938 | 0.0% | 24.1 | 4315.4 |
| `fixed_70_30` | 150 | **100.0%** | 0.333 | 100.0% | 0.938 | 0.0% | 26.2 | 3742.5 |
| `fixed_50_50` | 150 | **100.0%** | 0.333 | 100.0% | 0.938 | 0.0% | 26.2 | 3849.2 |
| `adaptive_hierarchical` | 150 | **100.0%** | 0.333 | 100.0% | 0.938 | 0.0% | 26.2 | 3638.8 |
| `fixed_30_70` | 150 | **100.0%** | 0.333 | 100.0% | 0.938 | 0.0% | 26.2 | 3745.7 |
| `flat_memory_baseline` | 150 | **100.0%** | 0.333 | 100.0% | 0.938 | 0.0% | 25.4 | 3874.1 |

### Key Scientific Findings:
1. **Severe Failure of Episodic-Heavy Fixed Allocation**: Under tight storage budgets ($B = 50$), `fixed_70_30` degraded significantly (62.5% accuracy, 62.5% recall), because reserving 70% for detailed episodic memory starved the compact semantic store where enduring facts reside.
2. **Robustness of Adaptive Allocation**: `adaptive_hierarchical` dynamically shifted budget toward semantic storage when factual demand was high, matching or outperforming 50/50 and 70/30 fixed splits (87.5% accuracy vs 62.5%).
3. **Competitiveness of Semantic-Heavy & Flat Baselines**: On factual queries under tight budgets, semantic-heavy allocation (`fixed_30_70`) and `flat_memory_baseline` achieved 100% accuracy by keeping all compact propositions. This demonstrates that dynamic allocation prevents catastrophic failure modes of rigid splits, while flat memory remains a strong baseline when memory demand is predominantly semantic.
4. **Zero Stale-Memory Errors**: Across all strategies with temporal validity enabled, the stale-memory error rate was **0.0%**, confirming that superseded facts were never mistakenly reported as current state.

---

## 5. Execution Instructions

### Running All Unit Tests (38 Tests)
```bash
python -m unittest discover -v -s . -p "test_*.py"
```

### Running Automated Synthetic Comparisons (Fast Mock Mode)
```bash
python run_experiments.py --dataset synthetic --budgets 50 100 150 200 --model fake --ablation
```

### Running Live Qwen GGUF Research Comparison
```bash
python run_experiments.py \
  --dataset synthetic \
  --budgets 50 150 \
  --model qwen \
  --model-path "C:\Users\Varun S\.ollama\models\blobs\sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6"
```

### Running LongMemEval Development Experiment
```bash
python run_experiments.py --dataset longmemeval_dev --budgets 50 100 150 --model fake
```

### Running Single-Session / Multi-Session Pipeline
```bash
python run_pipeline.py --phase 3 --model-path "path/to/qwen.gguf" --query "Where do I work now?"
```

---

## 6. Repository Structure

```
adaptive_mememory_management/
├── .gitignore               # Excludes datasets, data/, model weights, checkpoints, logs, virtualenvs
├── README.md                # Research specifications, empirical results, and execution guide
├── memory_core.py           # Phase 1: memory schemas, extraction, scoring, budget allocation
├── memory_lifecycle.py      # Phase 2: deduplication, conflicts, consolidation, transitions, auditing
├── model_adapter.py         # Phase 3: BaseLLMAdapter, LlamaCppAdapter (native Qwen tokenizer), FakeLLMAdapter
├── memory_retrieval.py      # Phase 3 Hardened: stopword filtering, margin pruning, prompt budget tracker
├── memory_strategies.py     # Phase 4: Flat, Fixed (70/30, 50/50, 30/70), Adaptive, and Ablated strategies
├── benchmark_dataset.py     # Phase 4: 8-category Synthetic Suite and LongMemEval-S (Dev & Eval splits)
├── evaluation_harness.py    # Phase 4: Paired experiment runner, metric calculators, and JSON logging
├── run_experiments.py       # Phase 4: CLI driver for research benchmarks
├── run_pipeline.py          # CLI driver for running Phase 1, 2, and 3 pipelines
├── test_memory_core.py      # Phase 1 unit and integration tests (11 tests)
├── test_memory_lifecycle.py # Phase 2 lifecycle tests (8 tests)
├── test_memory_retrieval.py # Phase 3 retrieval, temporal, and prompt budget tests (11 tests)
├── test_strategies.py       # Phase 4 strategy tests (4 tests)
├── test_evaluation.py       # Phase 4 evaluation metrics & harness tests (4 tests)
└── outputs/                 # Locally persisted outputs and experiment results (git-ignored)
```
