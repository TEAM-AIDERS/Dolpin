from agents.causality_agent import CausalityAgent

items = [
    {"id": "A", "referenced_id": None, "created_at": 1, "text": "원글"},
    {"id": "B", "referenced_id": "A", "created_at": 2, "text": "A 반응"},
    {"id": "C", "referenced_id": "B", "created_at": 3, "text": "B 반응"},
    {"id": "D", "referenced_id": "A", "created_at": 4, "text": "A 다른 반응"},
]

agent = CausalityAgent(items)
print(agent.build_chains())
