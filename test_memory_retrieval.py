"""
Comprehensive Unit and Integration Tests for Phase 3 (Hardened):
Query Retrieval, Relevance Filtering, Temporal Validity, Context Assembly, and Model Adapter.
"""

import unittest

from memory_core import (
    LifecycleStatus,
    MemoryItem,
    MemoryType,
    estimate_token_cost,
)
from memory_lifecycle import AdaptiveMemoryManager
from memory_retrieval import (
    PromptBudgetTracker,
    RetrievalWeights,
    assemble_memory_context,
    compute_query_relevance,
    extract_query_point_in_time,
    is_historical_query,
    is_memory_valid_at_timestamp,
    retrieve_memories,
    run_phase3_pipeline,
)
from model_adapter import FakeLLMAdapter


class TestModelAdapter(unittest.TestCase):
    def test_fake_llm_adapter_basic_and_canned(self):
        adapter = FakeLLMAdapter(canned_response="Mock answer.")
        res = adapter.generate("Hello?")
        self.assertEqual(res, "Mock answer.")

        # Test chat generation
        chat_res = adapter.generate_chat([{"role": "user", "content": "test"}])
        self.assertEqual(chat_res, "Mock answer.")
        self.assertEqual(len(adapter.call_history), 2)

    def test_fake_llm_adapter_token_counting(self):
        adapter = FakeLLMAdapter()
        tokens = adapter.tokenize("Where do I work now?")
        self.assertGreater(len(tokens), 0)
        self.assertEqual(adapter.count_tokens("Where do I work now?"), len(tokens))

        chat_tokens = adapter.count_chat_tokens([{"role": "user", "content": "hello"}])
        self.assertGreater(chat_tokens, adapter.count_tokens("hello"))

    def test_fake_llm_adapter_heuristic_answering(self):
        adapter = FakeLLMAdapter()
        ans_current = adapter.generate("Context: DeepMind\nWhere do I work?")
        self.assertIn("DeepMind", ans_current)

        ans_prev = adapter.generate("Context: OpenAI, DeepMind\nWhere did I work previously?")
        self.assertIn("OpenAI", ans_prev)


class TestQueryIntentAndRelevance(unittest.TestCase):
    def test_is_historical_query_detection(self):
        self.assertTrue(is_historical_query("Where did I work previously?"))
        self.assertTrue(is_historical_query("What was my old preference?"))
        self.assertTrue(is_historical_query("What company did I work at before?"))
        self.assertFalse(is_historical_query("Where do I work now?"))
        self.assertFalse(is_historical_query("What framework should I use today?"))

    def test_relevance_computation(self):
        mem_work = MemoryItem(
            id="1", content="I work at DeepMind.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s", timestamp=0, metadata={"slot_type": "employment"}
        )
        mem_code = MemoryItem(
            id="2", content="I always prefer PyTorch.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s", timestamp=0, metadata={"slot_type": "dl_framework"}
        )

        rel_work = compute_query_relevance("What company do I work for?", mem_work)
        rel_code = compute_query_relevance("What company do I work for?", mem_code)

        self.assertGreater(rel_work, rel_code)
        self.assertGreater(rel_work, 0.40)


class TestHardenedRetrievalAndDistractorExclusion(unittest.TestCase):
    def setUp(self):
        self.employer_mem = MemoryItem(
            id="cur-1", content="I work at DeepMind now.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s2", timestamp=2000.0, utility_score=0.85, recency=1.0,
            stability=0.8, lifecycle_status=LifecycleStatus.ACTIVE,
            metadata={"slot_type": "employment", "slot_value": "deepmind"}
        )
        self.cuda_debug_task = MemoryItem(
            id="task-1", content="Can you help me debug this CUDA issue right now?",
            memory_type=MemoryType.EPISODIC, source_session_id="s1", timestamp=1000.0,
            utility_score=0.40, recency=0.8, stability=0.2,
            lifecycle_status=LifecycleStatus.TEMPORARY
        )
        self.superseded_employer = MemoryItem(
            id="old-1", content="I work at OpenAI.", memory_type=MemoryType.SEMANTIC,
            source_session_id="s1", timestamp=1000.0, utility_score=0.5, recency=0.4,
            stability=0.8, lifecycle_status=LifecycleStatus.SUPERSEDED,
            metadata={"slot_type": "employment", "slot_value": "openai", "superseded_at": 2000.0}
        )

    def test_distractor_task_excluded_under_competition(self):
        # Query asking for employer should retrieve employer and PRUNE the CUDA task
        query = "Where do I work now?"
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[self.cuda_debug_task],
            semantic_store=[self.employer_mem],
            archive_store=[self.superseded_employer],
            top_k=5
        )

        retrieved_ids = [m[0].id for m in retrieved]
        self.assertIn("cur-1", retrieved_ids, "Relevant employer memory must be retrieved")
        self.assertNotIn("task-1", retrieved_ids, "Irrelevant transient task must be excluded")
        self.assertNotIn("old-1", retrieved_ids, "Superseded memory must be excluded for current query")

    def test_point_in_time_query(self):
        # At timestamp 1500.0, OpenAI was valid, DeepMind was not yet created
        query = "Where did I work at timestamp 1500.0?"
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[],
            semantic_store=[self.employer_mem],
            archive_store=[self.superseded_employer],
            top_k=5
        )

        retrieved_ids = [m[0].id for m in retrieved]
        self.assertIn("old-1", retrieved_ids, "At t=1500, OpenAI was the active fact")
        self.assertNotIn("cur-1", retrieved_ids, "DeepMind (created at t=2000) was in the future at t=1500")

    def test_lola_dual_expense_retrieval(self):
        # Regression test for Lola example:
        # Two distinct expenses (vet visit $50 and flea medication $25) in archive must both be retrieved
        vet_visit = MemoryItem(
            id="lola-vet-1",
            content="By the way, I remember when I took Lola to the vet last week, the consultation fee was $50, which was a discounted price for regular customers.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
            utility_score=0.8,
            recency=0.2,
            stability=0.6,
            lifecycle_status=LifecycleStatus.ACTIVE
        )
        flea_med = MemoryItem(
            id="lola-flea-1",
            content="By the way, I also got her flea and tick prevention medication, it was $25 for a 3-month supply.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s2",
            timestamp=200.0,
            utility_score=0.8,
            recency=0.3,
            stability=0.6,
            lifecycle_status=LifecycleStatus.ACTIVE
        )
        vet_paraphrase = MemoryItem(
            id="lola-vet-dup",
            content="By the way, I just took Lola to the vet last week and got a discounted consultation fee of $50 as a regular customer.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s3",
            timestamp=300.0,
            utility_score=0.7,
            recency=0.4,
            stability=0.5,
            lifecycle_status=LifecycleStatus.ACTIVE
        )
        unrelated_recent = MemoryItem(
            id="art-visit-1",
            content="It's a must-visit for any street art enthusiast!",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s40",
            timestamp=4000.0,
            utility_score=0.7,
            recency=1.0,
            stability=0.5,
            lifecycle_status=LifecycleStatus.ACTIVE
        )

        query = "What is the total cost of Lola's vet visit and flea medication?"
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[unrelated_recent],
            semantic_store=[],
            archive_store=[vet_visit, flea_med, vet_paraphrase],
            top_k=5
        )

        retrieved_ids = [m[0].id for m in retrieved]
        has_vet = "lola-vet-1" in retrieved_ids or "lola-vet-dup" in retrieved_ids
        self.assertTrue(has_vet, "Vet visit expense ($50) must be retrieved")
        self.assertIn("lola-flea-1", retrieved_ids, "Flea medication expense ($25) must be retrieved")
        self.assertNotIn("art-visit-1", retrieved_ids, "Unrelated street art distractor must be excluded")

    def test_ordinary_non_historical_query_retrieves_archived_fact(self):
        # Ordinary query without historical keywords should retrieve valid archived facts
        archived_fact = MemoryItem(
            id="db-fact",
            content="My primary database is PostgreSQL.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s1",
            timestamp=100.0,
            utility_score=0.85,
            recency=0.2,
            stability=0.9,
            lifecycle_status=LifecycleStatus.ACTIVE,
            metadata={"slot_type": "database", "slot_value": "postgresql"}
        )
        active_unrelated = MemoryItem(
            id="ui-today",
            content="We discussed the design system color palette today.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s10",
            timestamp=2000.0,
            utility_score=0.5,
            recency=1.0,
            stability=0.3,
            lifecycle_status=LifecycleStatus.ACTIVE
        )

        query = "What is my primary database?"
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[active_unrelated],
            semantic_store=[],
            archive_store=[archived_fact],
            top_k=5
        )

        retrieved_ids = [m[0].id for m in retrieved]
        self.assertIn("db-fact", retrieved_ids, "Valid archived fact must be retrieved for ordinary query")
        self.assertNotIn("ui-today", retrieved_ids, "Unrelated active memory must not displace relevant archived fact")

    def test_superseded_fact_in_archive_not_retrieved_for_current_query(self):
        # A superseded fact in archive must NEVER be retrieved for a current-state query
        older_superseded = MemoryItem(
            id="old-city",
            content="I live in New York City.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s1",
            timestamp=1000.0,
            utility_score=0.7,
            recency=0.2,
            stability=0.8,
            lifecycle_status=LifecycleStatus.SUPERSEDED,
            metadata={"slot_type": "location", "slot_value": "new york city", "superseded_at": 3000.0}
        )
        newer_active = MemoryItem(
            id="cur-city",
            content="I moved to Seattle last month and live in Seattle now.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s2",
            timestamp=3000.0,
            utility_score=0.9,
            recency=1.0,
            stability=0.85,
            lifecycle_status=LifecycleStatus.ACTIVE,
            metadata={"slot_type": "location", "slot_value": "seattle"}
        )

        # Current query: only Seattle must be returned
        current_query = "Where do I live right now?"
        retrieved_current = retrieve_memories(
            query=current_query,
            episodic_store=[],
            semantic_store=[newer_active],
            archive_store=[older_superseded],
            top_k=5
        )
        current_ids = [m[0].id for m in retrieved_current]
        self.assertIn("cur-city", current_ids, "Current city must be retrieved")
        self.assertNotIn("old-city", current_ids, "Superseded city must NOT be retrieved for current query")

        # Historical query: superseded fact must be surfaced
        hist_query = "Where did I live before moving to Seattle?"
        retrieved_hist = retrieve_memories(
            query=hist_query,
            episodic_store=[],
            semantic_store=[newer_active],
            archive_store=[older_superseded],
            top_k=5
        )
        hist_ids = [m[0].id for m in retrieved_hist]
        self.assertIn("old-city", hist_ids, "Superseded city must be retrieved for historical query")

    def test_unrelated_location_memory_without_query_overlap_not_boosted(self):
        # Query contains "Where" which matches location slot keywords, but asks about yoga classes
        query = "Where do I take yoga classes?"

        # Unrelated location memory with zero content keyword overlap
        loc_distractor = MemoryItem(
            id="loc-kolkovna",
            content="*Kolkovna:** This restaurant is located in the heart of Prague and serves traditional Czech cuisine.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s1",
            timestamp=1000.0,
            utility_score=0.8,
            recency=0.5,
            stability=0.9,
            lifecycle_status=LifecycleStatus.ACTIVE,
            metadata={"slot_type": "location"}
        )

        # Relevant target memory with keyword overlap
        yoga_target = MemoryItem(
            id="yoga-serenity",
            content="It's been super helpful for me, especially on days when I can't make it to Serenity Yoga.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s2",
            timestamp=2000.0,
            utility_score=0.7,
            recency=0.6,
            stability=0.7,
            lifecycle_status=LifecycleStatus.ACTIVE
        )

        # Legitimate location memory with keyword overlap for location query
        legit_location = MemoryItem(
            id="legit-loc",
            content="I live in Bangalore, Karnataka.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s3",
            timestamp=3000.0,
            utility_score=0.9,
            recency=1.0,
            stability=0.9,
            lifecycle_status=LifecycleStatus.ACTIVE,
            metadata={"slot_type": "location"}
        )

        rel_distractor = compute_query_relevance(query, loc_distractor)
        rel_yoga = compute_query_relevance(query, yoga_target)
        self.assertLess(rel_distractor, 0.18, "Unrelated location distractor with 0 overlap must fall below threshold")
        self.assertGreater(rel_yoga, rel_distractor, "Relevant yoga memory must have higher relevance than unrelated location distractor")

        # Legitimate location query must still receive slot boost and high relevance
        rel_legit = compute_query_relevance("Where do I live?", legit_location)
        self.assertGreaterEqual(rel_legit, 0.80, "Legitimate location memory must achieve high relevance")

        # In full retrieval, unrelated location memory must not displace relevant memory
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[],
            semantic_store=[yoga_target],
            archive_store=[loc_distractor],
            top_k=5
        )
        retrieved_ids = [m[0].id for m in retrieved]
        self.assertNotIn("loc-kolkovna", retrieved_ids, "Unrelated location distractor must not be retrieved")
        self.assertIn("yoga-serenity", retrieved_ids, "Relevant yoga target must be retrieved")

    def test_serenity_yoga_named_entity_ranks_above_apps_and_slot_distractors(self):
        # Query: "Where do I take yoga classes?"
        query = "Where do I take yoga classes?"

        serenity_brunch = MemoryItem(
            id="serenity-brunch",
            content="Do you have any recommendations for healthy brunch spots near Serenity Yoga?",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s51",
            timestamp=6000.0,
            utility_score=0.73,
            recency=1.0,
            stability=0.7,
            lifecycle_status=LifecycleStatus.ACTIVE
        )
        serenity_practice = MemoryItem(
            id="serenity-practice",
            content="It's been super helpful for me, especially on days when I can't make it to Serenity Yoga.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s51",
            timestamp=6020.0,
            utility_score=0.64,
            recency=1.0,
            stability=0.4,
            lifecycle_status=LifecycleStatus.ACTIVE
        )
        app_dylan = MemoryItem(
            id="app-dylan",
            content="*Dylan Werner Yoga**: A unique app that focuses on dynamic, challenging flows and offers a variety of classes.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s10",
            timestamp=2000.0,
            utility_score=0.64,
            recency=0.5,
            stability=0.4,
            lifecycle_status=LifecycleStatus.ACTIVE
        )
        app_peloton = MemoryItem(
            id="app-peloton",
            content="*Peloton Digital**: While known for their cycling classes, Peloton also offers a vast yoga library with classes ranging from beginner to advanced.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s10",
            timestamp=2000.0,
            utility_score=0.64,
            recency=0.5,
            stability=0.4,
            lifecycle_status=LifecycleStatus.ACTIVE
        )
        loc_kolkovna = MemoryItem(
            id="loc-kolkovna",
            content="*Kolkovna:** This restaurant is located in the heart of Prague and serves traditional Czech cuisine.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s1",
            timestamp=1000.0,
            utility_score=0.8,
            recency=0.5,
            stability=0.9,
            lifecycle_status=LifecycleStatus.ACTIVE,
            metadata={"slot_type": "location"}
        )

        retrieved = retrieve_memories(
            query=query,
            episodic_store=[],
            semantic_store=[],
            archive_store=[serenity_brunch, serenity_practice, app_dylan, app_peloton, loc_kolkovna],
            top_k=5
        )
        retrieved_ids = [m[0].id for m in retrieved]

        # Serenity Yoga facts must place in the top 5
        self.assertIn("serenity-brunch", retrieved_ids, "Serenity Yoga brunch reference must enter top 5")
        self.assertIn("serenity-practice", retrieved_ids, "Serenity Yoga practice reference must enter top 5")

        # Unrelated location distractor must not be retrieved
        self.assertNotIn("loc-kolkovna", retrieved_ids, "Unrelated location distractor must not be in top 5")

        # Serenity Yoga facts must outrank generic app recommendations
        rank_brunch = retrieved_ids.index("serenity-brunch")
        rank_practice = retrieved_ids.index("serenity-practice")
        if "app-dylan" in retrieved_ids:
            self.assertLess(rank_brunch, retrieved_ids.index("app-dylan"), "Serenity Yoga must outrank Dylan Werner app")
        if "app-peloton" in retrieved_ids:
            self.assertLess(rank_brunch, retrieved_ids.index("app-peloton"), "Serenity Yoga must outrank Peloton app")


class TestContextAssemblyAndPromptBudgetAccounting(unittest.TestCase):
    def test_budget_tracker_accounting(self):
        tracker = PromptBudgetTracker(
            model_context_window=1000,
            generation_allowance=200,
            system_prompt_tokens=150,
            query_tokens=50,
            chat_template_overhead=20,
            max_memory_context_tokens=400
        )
        # Available = 1000 - (150 + 50 + 20 + 200) = 580
        # Capped by max_memory_context_tokens = 400
        allowed = tracker.calculate_allowed_memory_budget()
        self.assertEqual(allowed, 400)

        # When window is tight:
        tracker_tight = PromptBudgetTracker(
            model_context_window=500,
            generation_allowance=200,
            system_prompt_tokens=150,
            query_tokens=50,
            chat_template_overhead=20,
            max_memory_context_tokens=400
        )
        # Available = 500 - 420 = 80
        self.assertEqual(tracker_tight.calculate_allowed_memory_budget(), 80)

    def test_prompt_budget_strictly_enforced(self):
        items = [
            (MemoryItem(id=f"mem-{i}", content=f"User preference statement number {i} with extra text words.",
                        memory_type=MemoryType.SEMANTIC, source_session_id="s", timestamp=0,
                        token_cost=15, lifecycle_status=LifecycleStatus.ACTIVE), 0.9 - i * 0.05)
            for i in range(10)
        ]

        res = assemble_memory_context("query", items, max_context_tokens=75)
        self.assertLessEqual(res.context_token_cost, 75)
        self.assertTrue(len(res.selected_memories) < len(items))

    def test_duplicate_context_suppressed(self):
        mem1 = MemoryItem(id="1", content="I prefer PyTorch.", memory_type=MemoryType.SEMANTIC,
                          source_session_id="s", timestamp=0, lifecycle_status=LifecycleStatus.ACTIVE)
        mem2 = MemoryItem(id="2", content="I prefer PyTorch.", memory_type=MemoryType.SEMANTIC,
                          source_session_id="s", timestamp=0, lifecycle_status=LifecycleStatus.ACTIVE)

        res = assemble_memory_context("query", [(mem1, 0.9), (mem2, 0.85)], max_context_tokens=256)
        self.assertEqual(len(res.selected_memories), 1, "Duplicate memory content should only appear once in context")


class TestEndToEndPhase3Pipeline(unittest.TestCase):
    def test_full_pipeline_with_fake_adapter(self):
        conversation = [
            {"role": "user", "content": "I work at OpenAI. I love PyTorch.", "timestamp": 1000.0},
            {"role": "assistant", "content": "Noted.", "timestamp": 1010.0},
            {"role": "user", "content": "I moved to DeepMind. I work at DeepMind now.", "timestamp": 2000.0},
            {"role": "assistant", "content": "Congrats on DeepMind!", "timestamp": 2010.0},
            {"role": "user", "content": "Can you debug this CUDA issue right now?", "timestamp": 2020.0},
        ]

        fake_llm = FakeLLMAdapter()

        # Query 1: Current company (must retrieve DeepMind and exclude CUDA issue)
        result_current = run_phase3_pipeline(
            conversation_history=conversation,
            user_query="Where do I work now?",
            llm_adapter=fake_llm,
            storage_budget=200,
            max_context_tokens=256,
            current_timestamp=2050.0
        )

        self.assertFalse(result_current["is_historical_query"])
        self.assertIn("DeepMind", result_current["response"])
        self.assertIn("DeepMind", result_current["context_result"]["formatted_context"])
        self.assertNotIn("CUDA", result_current["context_result"]["formatted_context"])

        # Query 2: Historical company
        result_prev = run_phase3_pipeline(
            conversation_history=[],
            user_query="Where did I work previously before DeepMind?",
            llm_adapter=fake_llm,
            storage_budget=200,
            max_context_tokens=256,
            existing_manager=result_current["manager"],
            current_timestamp=2050.0
        )

        self.assertTrue(result_prev["is_historical_query"])
        self.assertIn("OpenAI", result_prev["response"])




class TestBatch2TargetedBridgingAndStemming(unittest.TestCase):
    """
    Regression tests for Batch 2 generalizable retrieval refinements:
    1. Plural / inflection suffix normalization (kits/kit, lunches/lunch, worked/working)
    2. Unambiguous purchase synonym bridging (purchased <-> bought <-> order)
    3. Agricultural feed / poultry feed bridging (feed <-> scratch grains)
    4. Faith and religious activity semantic bridging (faith <-> church/mass/bible)
    5. Food delivery services entity bridging (food delivery <-> uber eats)
    """

    def test_suffix_stemming_plurals_and_inflections(self):
        # Query uses plural 'model kits' and past tense 'worked'
        query = "How many model kits have I worked on?"
        mem1 = MemoryItem(
            id="m1",
            content="I finished a 1/72 scale B-29 bomber model kit.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        mem2 = MemoryItem(
            id="m2",
            content="I am working on a 1/16 scale German Tiger I tank.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s2",
            timestamp=200.0,
        )
        rel1 = compute_query_relevance(query, mem1)
        rel2 = compute_query_relevance(query, mem2)
        self.assertGreaterEqual(rel1, 0.18)
        self.assertGreaterEqual(rel2, 0.18)

    def test_purchase_synonym_bridging(self):
        query = "What feed did I purchase for the animals?"
        mem_bought = MemoryItem(
            id="mb",
            content="I bought 20 pounds of organic scratch grains for my chickens.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        mem_unrelated = MemoryItem(
            id="mu",
            content="I like to take a walk in the evening.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s2",
            timestamp=100.0,
        )
        rel_bought = compute_query_relevance(query, mem_bought)
        rel_unrelated = compute_query_relevance(query, mem_unrelated)
        self.assertGreaterEqual(rel_bought, 0.18)
        self.assertLess(rel_unrelated, 0.10)

    def test_faith_activity_semantic_bridging(self):
        query = "How many days did I spend participating in faith-related activities in December?"
        mem_church = MemoryItem(
            id="mc",
            content="I attended midnight mass at St. Mary's Church on December 24th.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        mem_bible = MemoryItem(
            id="mb",
            content="I led a Bible study at my church on December 17th.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s2",
            timestamp=100.0,
        )
        mem_general_activity = MemoryItem(
            id="mga",
            content="I participated in various outdoor activities during summer.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s3",
            timestamp=100.0,
        )
        rel_church = compute_query_relevance(query, mem_church)
        rel_bible = compute_query_relevance(query, mem_bible)
        rel_general = compute_query_relevance(query, mem_general_activity)

        self.assertGreaterEqual(rel_church, 0.18)
        self.assertGreaterEqual(rel_bible, 0.18)

        # In end-to-end retrieval, faith memories are successfully retrieved
        retrieved = retrieve_memories(
            query=query,
            episodic_store=[mem_church, mem_bible, mem_general_activity],
            semantic_store=[],
            archive_store=[],
            top_k=3
        )
        retrieved_ids = {m.id for m, _ in retrieved}
        self.assertIn("mc", retrieved_ids)
        self.assertIn("mb", retrieved_ids)

    def test_food_delivery_entity_bridging(self):
        query = "How many different types of food delivery services have I used recently?"
        mem_uber = MemoryItem(
            id="mu",
            content="My weekends have been all about Uber Eats lately, it has been a lifesaver.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        rel_uber = compute_query_relevance(query, mem_uber)
        self.assertGreaterEqual(rel_uber, 0.18)




class TestBatch3PreferenceAndDomainBridging(unittest.TestCase):
    """
    Regression tests for Batch 3 generalizable retrieval refinements:
    1. Conversational meta-word suppression (filtering 'recommend', 'thinking', 'suggest' from dominating topical keywords)
    2. Academic research & publication semantic bridging (publications/conferences <-> advancements/research/papers)
    3. Culinary recipe & preparation bridging (recipe <-> making/ingredients)
    """

    def test_conversational_meta_word_suppression(self):
        query = "I was thinking of trying a new coffee creamer recipe. Any recommendations?"
        mem_topical = MemoryItem(
            id="m_topical",
            content="I started making my own flavored creamer with almond milk and honey.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        mem_conversational_distractor = MemoryItem(
            id="m_dist",
            content="I'm thinking of customizing my new Birkenstocks with some colorful straps.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s2",
            timestamp=100.0,
        )
        rel_topical = compute_query_relevance(query, mem_topical)
        rel_dist = compute_query_relevance(query, mem_conversational_distractor)

        self.assertGreaterEqual(rel_topical, 0.18)
        self.assertGreater(rel_topical, rel_dist)

    def test_academic_publication_bridging(self):
        query = "Can you recommend some recent publications or conferences that I might find interesting?"
        mem_academic = MemoryItem(
            id="m_acad",
            content="Can you give me an overview of the recent advancements in this field of deep learning for medical image analysis?",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        mem_distractor = MemoryItem(
            id="m_batman",
            content="If you're interested in preparing for the new Batman film, I'd recommend re-watching the previous ones.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s2",
            timestamp=100.0,
        )
        rel_academic = compute_query_relevance(query, mem_academic)
        rel_dist = compute_query_relevance(query, mem_distractor)

        self.assertGreaterEqual(rel_academic, 0.18)
        self.assertGreater(rel_academic, rel_dist)

    def test_culinary_recipe_bridging(self):
        query = "Any suggestions for new recipes for dinner?"
        mem_recipe = MemoryItem(
            id="m_rec",
            content="I've been making roasted vegetable bowls with quinoa for dinner.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        rel_recipe = compute_query_relevance(query, mem_recipe)
        self.assertGreaterEqual(rel_recipe, 0.18)
    def test_adversarial_topical_query_preserves_thinking_concept(self):
        """Topical queries mentioning 'thinking' (e.g. 'critical thinking') must retain 'thinking' and outrank distractors."""
        query = "What was that book on critical thinking I mentioned wanting to read?"
        mem_target = MemoryItem(
            id="m_think",
            content="I've been wanting to read that classic book on critical thinking, Thinking Fast and Slow.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        mem_distractor = MemoryItem(
            id="m_care",
            content="I'm reading a book on critical care medicine for my clinical rotations.",
            memory_type=MemoryType.SEMANTIC,
            source_session_id="s2",
            timestamp=100.0,
        )
        rel_target = compute_query_relevance(query, mem_target)
        rel_dist = compute_query_relevance(query, mem_distractor)
        self.assertGreaterEqual(rel_target, 0.18)
        self.assertGreater(rel_target, rel_dist)

    def test_adversarial_episodic_query_preserves_recommend_action(self):
        """Episodic queries asking about a past user recommendation must retain 'recommend' as a distinguishing predicate."""
        query = "What movie did I recommend to Sarah last week?"
        mem_target = MemoryItem(
            id="m_rec",
            content="I recommend that movie Interstellar to Sarah because she loves space exploration.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s1",
            timestamp=100.0,
        )
        mem_distractor = MemoryItem(
            id="m_cinema",
            content="Sarah and I went to see a movie at the cinema last week.",
            memory_type=MemoryType.EPISODIC,
            source_session_id="s2",
            timestamp=100.0,
        )
        # Without conversational request framing (it's not 'can you recommend'), 'recommend' remains an active keyword
        from memory_retrieval import clean_conversational_framing
        self.assertIn("recommend", clean_conversational_framing(query.lower()))
        rel_target = compute_query_relevance(query, mem_target)
        self.assertGreaterEqual(rel_target, 0.18)



if __name__ == "__main__":
    unittest.main()
