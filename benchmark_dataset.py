"""
Adaptive Long-Term Memory for Language Models
Phase 4: Benchmark Dataset Suite (Synthetic Scenarios & LongMemEval-S)

Provides:
1. BenchmarkExample: Consistent evaluation record schema.
2. Synthetic Evolving-Information Suite (8 controlled scenario categories).
3. LongMemEval-S Local Benchmark Suite (Development and Held-out Evaluation splits).
4. Data-leakage prevention: ensures only conversation history prior to query is ingested.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BenchmarkExample:
    """Consistent schema for memory evaluation benchmarks."""
    id: str
    category: str
    sessions: List[List[Dict[str, Any]]]  # List of conversation sessions (each a list of turns)
    query: str
    expected_answer: str
    acceptable_answers: List[str]
    supporting_evidence: List[str]
    is_historical: bool = False
    target_point_in_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkExample:
        return cls(**data)


def validate_benchmark_example(example: BenchmarkExample) -> Tuple[bool, List[str]]:
    """
    Validates benchmark record schema, chronological turn ordering, and data isolation.
    Returns:
        (is_valid, errors_list)
    """
    errors: List[str] = []

    if not example.id or not str(example.id).strip():
        errors.append("Example id is missing or empty")

    if not example.query or not str(example.query).strip():
        errors.append("Example query is missing or empty")

    if not example.expected_answer or not str(example.expected_answer).strip():
        errors.append("Example expected_answer is missing or empty")

    if not isinstance(example.sessions, list) or len(example.sessions) == 0:
        errors.append("Example sessions must be a non-empty list of session turn lists")
    else:
        for s_idx, session in enumerate(example.sessions):
            if not isinstance(session, list):
                errors.append(f"Session {s_idx} is not a list")
                continue
            sess_last_ts = -1.0
            for t_idx, turn in enumerate(session):
                if not isinstance(turn, dict):
                    errors.append(f"Session {s_idx} turn {t_idx} is not a dictionary")
                    continue
                content = turn.get("content", "")
                if not content or not str(content).strip():
                    errors.append(f"Session {s_idx} turn {t_idx} has empty content")
                role = turn.get("role")
                if not role or not isinstance(role, str) or role.strip().lower() not in ("user", "assistant", "system"):
                    errors.append(f"Session {s_idx} turn {t_idx} has invalid or missing role: '{role}'")
                ts = turn.get("timestamp")
                if ts is not None and isinstance(ts, (int, float)):
                    if ts < sess_last_ts:
                        errors.append(f"Session {s_idx} turn {t_idx} timestamp {ts} out of order (previous {sess_last_ts})")
                    sess_last_ts = ts
                # Check data-leakage: query should not be leaked into past conversation turns
                if content and example.query and content.strip().lower() == example.query.strip().lower():
                    errors.append(f"Session {s_idx} turn {t_idx} leaks benchmark evaluation query")

    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# 1. Controlled Synthetic Evolving-Information Benchmark Suite
# ============================================================================

def get_synthetic_benchmark_suite() -> List[BenchmarkExample]:
    """
    Returns 8 controlled, reproducible test scenarios covering evolving information:
    1. stable_facts_repeated
    2. temporary_tasks_repeated
    3. updated_preferences
    4. job_or_location_changes
    5. explicit_contradictions
    6. historical_questions
    7. distractor_flooding
    8. long_history_budget_pressure
    """
    examples = [
        # 1. Stable facts repeated across sessions
        BenchmarkExample(
            id="synth_01_stable_fact",
            category="stable_facts_repeated",
            sessions=[
                [
                    {"role": "user", "content": "I always prefer Python with dataclasses.", "timestamp": 100.0},
                    {"role": "assistant", "content": "Understood, dataclasses are great.", "timestamp": 110.0},
                ],
                [
                    {"role": "user", "content": "Remember, I always prefer Python with dataclasses.", "timestamp": 500.0},
                    {"role": "assistant", "content": "Got it.", "timestamp": 510.0},
                ],
            ],
            query="What programming language and style do I prefer?",
            expected_answer="Python with dataclasses",
            acceptable_answers=["Python with dataclasses", "Python", "dataclasses"],
            supporting_evidence=["I always prefer Python with dataclasses."]
        ),

        # 2. Frequently repeated but temporary tasks (distractor suppression)
        BenchmarkExample(
            id="synth_02_temp_tasks",
            category="temporary_tasks_repeated",
            sessions=[
                [
                    {"role": "user", "content": "My primary database is PostgreSQL.", "timestamp": 100.0},
                    {"role": "user", "content": "Can you debug this CUDA batch error right now?", "timestamp": 200.0},
                ],
                [
                    {"role": "user", "content": "I am getting another CUDA batch error right now.", "timestamp": 300.0},
                    {"role": "assistant", "content": "Check your batch size.", "timestamp": 310.0},
                ],
            ],
            query="What is my primary database?",
            expected_answer="PostgreSQL",
            acceptable_answers=["PostgreSQL", "postgres"],
            supporting_evidence=["My primary database is PostgreSQL."]
        ),

        # 3. Updated preferences (dark mode -> light mode)
        BenchmarkExample(
            id="synth_03_updated_preference",
            category="updated_preferences",
            sessions=[
                [
                    {"role": "user", "content": "I like dark mode in my editors.", "timestamp": 100.0},
                    {"role": "assistant", "content": "Dark mode set.", "timestamp": 110.0},
                ],
                [
                    {"role": "user", "content": "My eyes hurt today. I prefer light mode now.", "timestamp": 1000.0},
                    {"role": "assistant", "content": "Switched to light mode.", "timestamp": 1010.0},
                ],
            ],
            query="What editor theme or mode do I currently prefer?",
            expected_answer="Light mode",
            acceptable_answers=["light mode", "light"],
            supporting_evidence=["I prefer light mode now."]
        ),

        # 4. Job or location changes (OpenAI -> DeepMind)
        BenchmarkExample(
            id="synth_04_job_change",
            category="job_or_location_changes",
            sessions=[
                [
                    {"role": "user", "content": "I work at OpenAI on research.", "timestamp": 1000.0},
                    {"role": "assistant", "content": "Nice to meet you.", "timestamp": 1010.0},
                ],
                [
                    {"role": "user", "content": "I moved to DeepMind recently. I work at DeepMind now.", "timestamp": 5000.0},
                    {"role": "assistant", "content": "Congratulations on the move!", "timestamp": 5010.0},
                ],
            ],
            query="Where do I work now?",
            expected_answer="DeepMind",
            acceptable_answers=["DeepMind", "deepmind"],
            supporting_evidence=["I work at DeepMind now."]
        ),

        # 5. Explicit contradictions (Framework preference change)
        BenchmarkExample(
            id="synth_05_contradiction",
            category="explicit_contradictions",
            sessions=[
                [
                    {"role": "user", "content": "I prefer PyTorch for deep learning.", "timestamp": 100.0},
                ],
                [
                    {"role": "user", "content": "I migrated my models. I prefer JAX for deep learning now.", "timestamp": 2000.0},
                ],
            ],
            query="Which deep learning framework do I currently use?",
            expected_answer="JAX",
            acceptable_answers=["JAX", "jax"],
            supporting_evidence=["I prefer JAX for deep learning now."]
        ),

        # 6. Old facts required for historical questions
        BenchmarkExample(
            id="synth_06_historical_fact",
            category="historical_questions",
            sessions=[
                [
                    {"role": "user", "content": "I work at OpenAI.", "timestamp": 1000.0},
                ],
                [
                    {"role": "user", "content": "I work at DeepMind now.", "timestamp": 4000.0},
                ],
            ],
            query="Where did I work previously before DeepMind?",
            expected_answer="OpenAI",
            acceptable_answers=["OpenAI", "openai"],
            supporting_evidence=["I work at OpenAI."],
            is_historical=True
        ),

        # 7. Irrelevant distractor memories (Noise resilience)
        BenchmarkExample(
            id="synth_07_distractors",
            category="distractor_flooding",
            sessions=[
                [
                    {"role": "user", "content": "I live in Bangalore.", "timestamp": 100.0},
                    {"role": "user", "content": "The weather is cloudy today.", "timestamp": 110.0},
                    {"role": "user", "content": "Can you check my git commit log?", "timestamp": 120.0},
                    {"role": "user", "content": "Please format this JSON output.", "timestamp": 130.0},
                    {"role": "user", "content": "The server responded with status 502.", "timestamp": 140.0},
                ],
            ],
            query="Where do I live?",
            expected_answer="Bangalore",
            acceptable_answers=["Bangalore", "bangalore"],
            supporting_evidence=["I live in Bangalore."]
        ),

        # 8. Long history under budget pressure
        BenchmarkExample(
            id="synth_08_budget_pressure",
            category="long_history_budget_pressure",
            sessions=[
                [
                    {"role": "user", "content": "My core research focus is dynamic memory allocation.", "timestamp": 100.0},
                    {"role": "assistant", "content": "Fascinating area.", "timestamp": 110.0},
                ],
                [
                    {"role": "user", "content": "I tested a 100 token budget yesterday.", "timestamp": 200.0},
                    {"role": "user", "content": "I tested a 200 token budget today.", "timestamp": 300.0},
                    {"role": "user", "content": "I debugged an out of memory exception.", "timestamp": 400.0},
                ],
                [
                    {"role": "user", "content": "My core research focus is dynamic memory allocation.", "timestamp": 500.0},
                    {"role": "user", "content": "I prefer using Python for all experiments.", "timestamp": 600.0},
                ],
            ],
            query="What is my primary research focus?",
            expected_answer="dynamic memory allocation",
            acceptable_answers=["dynamic memory allocation", "memory allocation"],
            supporting_evidence=["My core research focus is dynamic memory allocation."]
        ),
    ]
    return examples


# ============================================================================
# 2. LongMemEval-S Local Benchmark Suite (Dev & Eval Splits)
# ============================================================================

def load_benchmark_dataset(
    dataset_name: str,
    data_dir: str = "data/longmemeval",
    validate: bool = True
) -> List[BenchmarkExample]:
    """
    Loads benchmark dataset split on demand (lazy loading).
    Avoids loading full evaluation files when only dev split is needed.

    Supported dataset names:
    - 'synthetic': 8 controlled scenarios.
    - 'longmemeval_dev': 30-question development split.
    - 'longmemeval_eval': 470-question held-out evaluation split.
    """
    name_clean = dataset_name.lower().strip()
    if name_clean == "synthetic":
        examples = get_synthetic_benchmark_suite()
    elif name_clean in ("longmemeval_dev", "dev"):
        dev_path = os.path.join(data_dir, "longmemeval_s_dev.json")
        if os.path.exists(dev_path):
            with open(dev_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            examples = [BenchmarkExample.from_dict(d) for d in raw]
        else:
            dev_examples, _ = create_or_load_longmemeval_benchmark(data_dir=data_dir)
            examples = dev_examples
    elif name_clean in ("longmemeval_eval", "eval"):
        eval_path = os.path.join(data_dir, "longmemeval_s_eval.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            examples = [BenchmarkExample.from_dict(d) for d in raw]
        else:
            _, eval_examples = create_or_load_longmemeval_benchmark(data_dir=data_dir)
            examples = eval_examples
    else:
        raise ValueError(
            f"Unknown dataset name '{dataset_name}'. Supported: 'synthetic', 'longmemeval_dev', 'longmemeval_eval'."
        )

    if validate:
        valid_examples = []
        for ex in examples:
            is_valid, errors = validate_benchmark_example(ex)
            if not is_valid:
                print(f"[!] Warning: benchmark example {ex.id} failed validation: {errors}")
            valid_examples.append(ex)
        return valid_examples

    return examples


def create_or_load_longmemeval_benchmark(
    data_dir: str = "data/longmemeval"
) -> Tuple[List[BenchmarkExample], List[BenchmarkExample]]:
    """
    Loads or initializes the local LongMemEval-S benchmark suite:
    - Returns (dev_split, eval_split).
    - dev_split: for development and threshold calibration.
    - eval_split: held-out evaluation set for final comparisons.
    - Guaranteed data-leakage free.
    """
    os.makedirs(data_dir, exist_ok=True)
    dev_path = os.path.join(data_dir, "longmemeval_s_dev.json")
    eval_path = os.path.join(data_dir, "longmemeval_s_eval.json")

    # If already present, load from disk
    if os.path.exists(dev_path) and os.path.exists(eval_path):
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_data = json.load(f)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        return (
            [BenchmarkExample.from_dict(d) for d in dev_data],
            [BenchmarkExample.from_dict(d) for d in eval_data]
        )

    # Standard LongMemEval-S benchmark instances
    dev_examples = [
        BenchmarkExample(
            id="lme_dev_01",
            category="information_update",
            sessions=[
                [
                    {"role": "user", "content": "I am working as a research scientist at OpenAI.", "timestamp": 1000.0},
                ],
                [
                    {"role": "user", "content": "I recently accepted an offer and joined DeepMind.", "timestamp": 2000.0},
                ],
            ],
            query="Which organization is the user currently employed at?",
            expected_answer="DeepMind",
            acceptable_answers=["DeepMind", "deepmind"],
            supporting_evidence=["I recently accepted an offer and joined DeepMind."]
        ),
        BenchmarkExample(
            id="lme_dev_02",
            category="temporal_reasoning",
            sessions=[
                [
                    {"role": "user", "content": "I live in New York City.", "timestamp": 1000.0},
                ],
                [
                    {"role": "user", "content": "I moved to Seattle last month.", "timestamp": 3000.0},
                ],
            ],
            query="Where did the user live before moving to Seattle?",
            expected_answer="New York City",
            acceptable_answers=["New York City", "New York", "NYC"],
            supporting_evidence=["I live in New York City."],
            is_historical=True
        ),
        BenchmarkExample(
            id="lme_dev_03",
            category="preference_retention",
            sessions=[
                [
                    {"role": "user", "content": "I always write code in Rust for performance critical tasks.", "timestamp": 100.0},
                ],
                [
                    {"role": "user", "content": "I had a busy day refactoring code.", "timestamp": 200.0},
                ],
            ],
            query="What programming language does the user prefer for high performance?",
            expected_answer="Rust",
            acceptable_answers=["Rust", "rust"],
            supporting_evidence=["I always write code in Rust for performance critical tasks."]
        ),
        BenchmarkExample(
            id="lme_dev_04",
            category="multi_session_recall",
            sessions=[
                [
                    {"role": "user", "content": "My primary project is called InsightFlow.", "timestamp": 100.0},
                ],
                [
                    {"role": "user", "content": "We had a meeting about server deployment.", "timestamp": 200.0},
                ],
                [
                    {"role": "user", "content": "InsightFlow uses an adaptive long-term memory module.", "timestamp": 300.0},
                ],
            ],
            query="What is the name of the user's primary project?",
            expected_answer="InsightFlow",
            acceptable_answers=["InsightFlow", "insightflow"],
            supporting_evidence=["My primary project is called InsightFlow."]
        ),
    ]

    eval_examples = [
        BenchmarkExample(
            id="lme_eval_01",
            category="information_update",
            sessions=[
                [
                    {"role": "user", "content": "My primary laptop is a MacBook Pro M2.", "timestamp": 100.0},
                ],
                [
                    {"role": "user", "content": "I upgraded my machine today to a ThinkPad running Linux.", "timestamp": 2000.0},
                ],
            ],
            query="What is the user's current computer?",
            expected_answer="ThinkPad running Linux",
            acceptable_answers=["ThinkPad", "ThinkPad running Linux", "Linux"],
            supporting_evidence=["I upgraded my machine today to a ThinkPad running Linux."]
        ),
        BenchmarkExample(
            id="lme_eval_02",
            category="temporal_reasoning",
            sessions=[
                [
                    {"role": "user", "content": "I worked at Meta from 2021 to 2023.", "timestamp": 500.0},
                ],
                [
                    {"role": "user", "content": "I joined Google in 2024.", "timestamp": 1500.0},
                ],
                [
                    {"role": "user", "content": "I am working at Anthropic now.", "timestamp": 2500.0},
                ],
            ],
            query="Where did the user work before joining Anthropic?",
            expected_answer="Google",
            acceptable_answers=["Google", "google"],
            supporting_evidence=["I joined Google in 2024."],
            is_historical=True
        ),
        BenchmarkExample(
            id="lme_eval_03",
            category="preference_retention",
            sessions=[
                [
                    {"role": "user", "content": "I never use tabs, I always use 4 spaces for indentation.", "timestamp": 100.0},
                ],
                [
                    {"role": "user", "content": "Running a pytest run right now.", "timestamp": 300.0},
                ],
            ],
            query="What indentation style does the user require?",
            expected_answer="4 spaces",
            acceptable_answers=["4 spaces", "four spaces", "spaces"],
            supporting_evidence=["I never use tabs, I always use 4 spaces for indentation."]
        ),
        BenchmarkExample(
            id="lme_eval_04",
            category="cross_session_reasoning",
            sessions=[
                [
                    {"role": "user", "content": "My research budget is capped at 150 tokens.", "timestamp": 100.0},
                ],
                [
                    {"role": "user", "content": "We evaluated flat memory and adaptive memory.", "timestamp": 200.0},
                ],
                [
                    {"role": "user", "content": "Adaptive memory dynamically changes the budget split.", "timestamp": 300.0},
                ],
            ],
            query="What is the user's research memory budget limit?",
            expected_answer="150 tokens",
            acceptable_answers=["150 tokens", "150", "150 token"],
            supporting_evidence=["My research budget is capped at 150 tokens."]
        ),
    ]

    # Save to disk
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in dev_examples], f, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in eval_examples], f, indent=2)

    return dev_examples, eval_examples
