# Developer & Teammate Integration Guide

## Adaptive Long-Term Memory for Language Models: Dynamic Memory-Budget Allocation under Evolving Information

Welcome to the shared research pipeline. This guide explains how to integrate and evaluate your own independent persistent-memory strategy without modifying shared infrastructure, dataset loaders, model adapters, or other teammates' implementations.

---

## 1. Architecture Overview (3-Layer Separation)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Layer 1: Shared Infrastructure                       │
│  - Dataset Loaders & Schema Validation (benchmark_dataset.py)          │
│  - Model Adapters & Native Tokenizer (model_adapter.py)               │
│  - Context Assembly & Budget Bounding (memory_retrieval.py)            │
│  - Evaluation Engine & Metrics (evaluation_harness.py)                 │
│  - CLI Runner & Isolation (run_experiments.py)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Memory Integration Contract
┌───────────────────────────────────▼────────────────────────────────────┐
│                  Layer 2: Memory Integration Contract                  │
│  - memory_interface.BaseMemoryStrategy (abstract interface)            │
│  - memory_interface.RetrievedEvidence (strategy-neutral evidence format)│
│  - memory_interface.register_strategy (decorator / registry)           │
│  - memory_interface.MockMemoryStrategy (reference mock implementation) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Subclassed by independent strategies
┌───────────────────────────────────▼────────────────────────────────────┐
│             Layer 3: Independent Strategy Implementations              │
│  - Flat Memory Baseline (FlatMemoryBaseline)                           │
│  - Fixed Hierarchical Split (FixedHierarchicalStrategy: 70/30, 50/50)  │
│  - Adaptive Hierarchical Allocation (AdaptiveHierarchicalStrategy)     │
│  - Teammate Custom Strategies (e.g. Graph, Vector, Recency, etc.)      │
└────────────────────────────────────────────────────────────────────────┘
```

**Key Principle:** The shared infrastructure does **not** assume episodic/semantic partitioning, consolidation, or any specific indexing algorithm. Your strategy only needs to conform to `BaseMemoryStrategy`.

---

## 2. Environment Setup & Dependencies

Python 3.10+ is recommended.

```bash
# Clone and enter directory
cd adaptive_mememory_management

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 3. Local Model (Qwen) Configuration

Our primary model is **Qwen 2.5 (GGUF)** running locally via `llama-cpp-python`.

### A. Install llama-cpp-python (Optional for Fake LLM tests)
```bash
pip install llama-cpp-python
```

### B. Configure Model Path
Set the environment variable or pass `--model-path`:
```bash
# Windows PowerShell:
$env:QWEN_MODEL_PATH = "C:\Users\<user>\.ollama\models\blobs\<sha256-blob-name>"

# Linux / macOS:
export QWEN_MODEL_PATH="/path/to/qwen2.5-7b-instruct.Q4_K_M.gguf"
```

If the Qwen model file is not found, the runner raises a clear `FileNotFoundError` rather than silently substituting fake weights.

---

## 4. Benchmark Datasets

Datasets are located in `data/longmemeval/`:
- `synthetic` (8 controlled scenarios covering information updates, task distractors, temporal shifts).
- `longmemeval_dev` (30 questions from LongMemEval-S for threshold calibration and fast testing).
- `longmemeval_eval` (470 held-out questions for final scientific comparison).

### Data Schema (`BenchmarkExample`):
Each benchmark example provides:
- `id`: Unique identifier (e.g., `"synth_01_stable_fact"`, `"94f70d80"`).
- `category`: Scenario type (e.g., `"information_update"`, `"temporal_reasoning"`).
- `sessions`: List of sessions, each containing turns `[{"role": "user"|"assistant", "content": str, "timestamp": float}]`.
- `query`: The evaluation question.
- `expected_answer`: Reference answer (used **only** for scoring; never passed to memory ingestion).
- `acceptable_answers`: List of valid alternative substrings or phrases.
- `supporting_evidence`: Ground truth evidence text for retrieval Recall@K and MRR evaluation.

---

## 5. How to Implement a New Strategy (Step-by-Step)

### Step 1: Subclass `BaseMemoryStrategy`
Create a new file, e.g. `my_memory_strategy.py`:

```python
from typing import Any, Dict, List, Optional
from memory_interface import BaseMemoryStrategy, RetrievedEvidence, register_strategy

@register_strategy("my_strategy_name")
class MyCustomMemoryStrategy(BaseMemoryStrategy):
    """
    Example strategy implementation.
    """
    def __init__(
        self,
        total_budget: int = 500,
        tokenizer_fn: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__(
            name="my_strategy_name",
            total_budget=total_budget,
            tokenizer_fn=tokenizer_fn,
            config=config,
            **kwargs
        )
        self.store: List[RetrievedEvidence] = []

    def reset(self) -> None:
        """Called automatically before each benchmark question to guarantee zero data leakage."""
        self.store.clear()

    def process_session(
        self,
        conversation_history: List[Dict[str, Any]],
        session_id: str,
        current_timestamp: float
    ) -> Dict[str, Any]:
        """
        Ingest conversation history turns for this session.
        Respect self.total_budget.
        """
        for turn in conversation_history:
            content = turn.get("content", "").strip()
            role = turn.get("role", "user")
            if not content:
                continue

            item = RetrievedEvidence(
                id=f"{session_id}_{len(self.store)}",
                content=content,
                score=1.0,
                timestamp=turn.get("timestamp", current_timestamp),
                lifecycle_status="current",
                metadata={"role": role}
            )
            self.store.append(item)

        # Enforce budget cap
        while self.get_storage_usage() > self.total_budget and self.store:
            self.store.pop(0)  # Eviction policy

        return {
            "strategy": self.name,
            "session_id": session_id,
            "stored_count": len(self.store),
            "storage_tokens_used": self.get_storage_usage()
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedEvidence]:
        """
        Return the top_k most relevant memories for the user query.
        """
        # Your custom scoring or vector search
        return self.store[-top_k:]

    def get_persisted_memories(self) -> List[Any]:
        """
        Return all memories currently occupying storage.
        """
        return list(self.store)
```

### Step 2: Strategy Discovery (Zero-Modification Workflow)
You do **NOT** need to edit `run_experiments.py` or any shared file.

You have three convenient ways to make your strategy available:
1. **Dedicated Directory (Recommended):** Put your file in `strategies/` (e.g. `strategies/my_memory.py`). It is auto-discovered on launch.
2. **Filename Convention:** Name your file `strategy_<name>.py` in the root workspace. It is auto-discovered on launch.
3. **Explicit CLI Flag:** Pass `--strategy-module <path/to/file.py>` on the command line:

```bash
# Check registered strategies (including your auto-discovered or module strategy)
python run_experiments.py --list-strategies
python run_experiments.py --strategy-module my_memory_strategy.py --list-strategies

# Run your strategy on synthetic smoke test
python run_experiments.py --strategy my_strategy_name --dataset synthetic --budgets 500 --limit 1
python run_experiments.py --strategy-module my_memory_strategy.py --strategy my_strategy_name --dataset synthetic --budgets 500 --limit 1
```

---

## 6. How to Run Tests

```bash
# Run all unit and integration tests (90+ tests)
python -m unittest discover -v -s . -p "test_*.py"

# Run only shared pipeline tests
python -m unittest -v test_shared_pipeline.py

# Run strategy hardening tests
python -m unittest -v test_strategy_hardening.py
```

---

## 7. Execution CLI Commands & Options

```bash
# 1. Fast smoke test (1 question, fake adapter)
python run_experiments.py --strategy mock --dataset synthetic --budgets 500 --limit 1

# 2. Evaluate single strategy on 30-question development split (Fake LLM)
python run_experiments.py --strategy flat_memory --dataset longmemeval_dev --budgets 500

# 3. Evaluate single strategy on 30-question development split (Local Qwen)
python run_experiments.py --strategy adaptive_hierarchical --dataset longmemeval_dev --budgets 500 --model qwen

# 4. Run comparative suite across all 5 strategies under 500 tokens
python run_experiments.py --strategy all --dataset longmemeval_dev --budgets 500 --limit 5
```

---

## 8. Strategy Hyperparameter Configuration Template

Teammates should record strategy-specific configuration in a JSON or Python dict passed to `--strategy`:

```json
{
  "strategy_name": "adaptive_hierarchical",
  "total_budget": 500,
  "min_episodic_share": 0.20,
  "max_episodic_share": 0.80,
  "scoring_weights": {
    "w_recency": 0.25,
    "w_importance": 0.25,
    "w_frequency": 0.20,
    "w_stability": 0.15,
    "w_consistency": 0.15
  },
  "retrieval_weights": {
    "w_relevance": 0.50,
    "w_utility": 0.25,
    "w_recency": 0.15,
    "w_stability": 0.10
  },
  "min_frequency_to_consolidate": 2,
  "dedup_similarity_threshold": 0.80
}
```

---

## 9. Output Locations and Trace Artifacts

- Structured benchmark JSON results: `outputs/experiments/<dataset>_<timestamp>.json`
- SQLite trace database (if web UI trace active): `data/traces.db`
- Per-question metrics saved:
  - Exact match (`exact_match`)
  - Substring accuracy (`substring_accuracy`)
  - Token F1 (`token_f1`)
  - Retrieval Recall@K (`recall_at_k`)
  - Retrieval MRR (`mrr`)
  - Stale memory error rate (`stale_memory_error`)
  - Storage tokens used (`storage_tokens_used`)
  - Context tokens used (`context_tokens_used`)
  - Retrieval latency (`retrieval_latency_ms`)
  - Generation latency (`generation_latency_ms`)
  - Error logs if question processing failed (`error`)

---

## 10. Troubleshooting

1. **`FileNotFoundError: Configured Qwen model file does not exist`**
   - Check the path passed to `--model-path` or the value in `$env:QWEN_MODEL_PATH`.
   - For automated verification without local weights, use `--model fake`.
2. **`KeyError: Strategy 'xyz' is not registered`**
   - Ensure your strategy file was imported or registered using `@register_strategy("xyz")`.
   - Run `python run_experiments.py --list-strategies` to inspect registered names.
3. **`Memory budget exceeded`**
   - In `process_session()`, ensure your pruning or knapsack loop stops when `self.get_storage_usage() <= self.total_budget`.

---

## 11. Team Workflow, Branching & Pull Requests

To ensure team members work in parallel without merge conflicts or breaking shared evaluation infrastructure:

### A. Dedicated Strategy Branches
Each teammate should branch off the `shared-pipeline-handoff` or `main` branch:

```bash
git checkout -b feature/<strategy-name>
# Examples:
# git checkout -b feature/fixed-ratio-hierarchical
# git checkout -b feature/flat-vector-baseline
# git checkout -b feature/adaptive-demand-allocation
```

### B. Development Isolation Rules
1. **Never edit Layer 1 Shared Files:** Do not modify `evaluation_harness.py`, `benchmark_dataset.py`, `model_adapter.py`, or `memory_retrieval.py`.
2. **Place code in `strategies/`:** Implement your memory strategy in a standalone file, e.g. `strategies/<my_strategy>.py`.
3. **Register via `@register_strategy("<name>")`:** The pipeline will auto-discover your file without any edits to `run_experiments.py`.

### C. Pre-PR Checklist
Before pushing your branch or submitting a pull request:
1. Run all unit and integration tests:
   ```bash
   python -m unittest discover -v -s . -p "test_*.py"
   ```
2. Verify strategy listing:
   ```bash
   python run_experiments.py --list-strategies
   ```
3. Run a smoke test:
   ```bash
   python run_experiments.py --strategy <my_strategy> --dataset synthetic --budgets 500 --limit 1
   ```
4. Push your branch to GitHub:
   ```bash
   git push -u origin feature/<strategy-name>
   ```
5. Open a Pull Request targeting `main`. Include the Markdown evaluation results table generated by the harness in your PR description.

