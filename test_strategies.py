"""
Unit Tests for Phase 4 Memory Strategies and Baselines:
Flat, Fixed Hierarchical, Adaptive Hierarchical, and Ablated Strategies.
"""

import unittest

from memory_strategies import (
    AblatedMemoryStrategy,
    AdaptiveHierarchicalStrategy,
    FixedHierarchicalStrategy,
    FlatMemoryBaseline,
)


class TestMemoryStrategies(unittest.TestCase):
    def setUp(self):
        self.session_1 = [
            {"role": "user", "content": "I work at OpenAI. I love PyTorch.", "timestamp": 1000.0},
            {"role": "assistant", "content": "Noted.", "timestamp": 1010.0},
        ]
        self.session_2 = [
            {"role": "user", "content": "I moved to DeepMind. I work at DeepMind now.", "timestamp": 2000.0},
            {"role": "assistant", "content": "Congrats on DeepMind!", "timestamp": 2010.0},
            {"role": "user", "content": "Can you debug this CUDA issue right now?", "timestamp": 2020.0},
        ]

    def test_flat_memory_baseline(self):
        strategy = FlatMemoryBaseline(total_budget=100)
        strategy.process_session(self.session_1, "s1", 1020.0)
        strategy.process_session(self.session_2, "s2", 2050.0)

        # Budget strictly enforced
        self.assertLessEqual(strategy.get_storage_usage(), 100)

        # Retrieval works
        retrieved = strategy.retrieve("Where do I work now?")
        self.assertTrue(len(retrieved) > 0)
        top_mem = retrieved[0][0]
        self.assertIn("DeepMind", top_mem.content)

    def test_fixed_hierarchical_splits(self):
        for ep_share in [0.70, 0.50, 0.30]:
            strat = FixedHierarchicalStrategy(total_budget=100, episodic_share=ep_share)
            strat.process_session(self.session_1, "s1", 1020.0)
            strat.process_session(self.session_2, "s2", 2050.0)

            self.assertLessEqual(strat.get_storage_usage(), 100)
            retrieved = strat.retrieve("Where do I work now?")
            self.assertTrue(any("DeepMind" in m[0].content for m in retrieved))

    def test_adaptive_hierarchical_strategy(self):
        strat = AdaptiveHierarchicalStrategy(total_budget=100)
        strat.process_session(self.session_1, "s1", 1020.0)
        strat.process_session(self.session_2, "s2", 2050.0)

        self.assertLessEqual(strat.get_storage_usage(), 100)
        retrieved = strat.retrieve("Where do I work now?")
        self.assertTrue(len(retrieved) > 0)
        self.assertIn("DeepMind", retrieved[0][0].content)

    def test_fairness_equal_budget_enforcement_across_all_strategies(self):
        budget = 75
        strategies = [
            FlatMemoryBaseline(total_budget=budget),
            FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.70),
            FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.50),
            FixedHierarchicalStrategy(total_budget=budget, episodic_share=0.30),
            AdaptiveHierarchicalStrategy(total_budget=budget),
            AblatedMemoryStrategy(total_budget=budget, disable_adaptive=True),
            AblatedMemoryStrategy(total_budget=budget, disable_consolidation=True),
        ]

        for s in strategies:
            s.process_session(self.session_1, "s1", 1020.0)
            s.process_session(self.session_2, "s2", 2050.0)
            usage = s.get_storage_usage()
            self.assertLessEqual(
                usage, budget,
                f"Strategy {s.name} exceeded total budget {budget}: used {usage}"
            )


if __name__ == "__main__":
    unittest.main()
