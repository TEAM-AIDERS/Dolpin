from agents.causality_agent import CausalityAgent

def run_sample():
    items = [
        {
            "id": "A",
            "referenced_id": None,
            "created_at": 1,
            "text": "원글"
        },
        {
            "id": "B",
            "referenced_id": "A",
            "created_at": 2,
            "text": "A에 대한 반응"
        },
        {
            "id": "C",
            "referenced_id": "B",
            "created_at": 3,
            "text": "B에 대한 반응"
        },
        {
            "id": "D",
            "referenced_id": "A",
            "created_at": 4,
            "text": "A에 대한 또 다른 반응"
        }
    ]

    agent = CausalityAgent(items)
    chains = agent.build_chains()

    for c in chains:
        print(c)


if __name__ == "__main__":
    run_sample()
