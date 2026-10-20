"""Auto-discovered strategy in strategies/ directory."""
from memory_interface import BaseMemoryStrategy, register_strategy, RetrievedEvidence

@register_strategy("auto_discovered_plugin")
class AutoDiscoveredPlugin(BaseMemoryStrategy):
    def __init__(self, total_budget=500, **kwargs):
        super().__init__(name="auto_discovered_plugin", total_budget=total_budget, **kwargs)
        self.items = []

    def reset(self):
        self.items.clear()

    def process_session(self, conversation_history, session_id, current_timestamp):
        for turn in conversation_history:
            self.items.append(RetrievedEvidence(content=turn.get("content", "")))
        return {"stored_count": len(self.items)}

    def retrieve(self, query, top_k=5):
        return [(item, 1.0) for item in self.items[:top_k]]

    def get_persisted_memories(self):
        return self.items
