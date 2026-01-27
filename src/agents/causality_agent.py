from typing import List, Dict, Set
from collections import defaultdict


class CausalityAgent:
    def __init__(self, items: List[Dict]):
        self.items = items
        self.id_map = {item["id"]: item for item in items}
        self.reverse_map = self._build_reverse_map()

    def _build_reverse_map(self):
        reverse = defaultdict(list)
        for item in self.items:
            ref = item.get("referenced_id")
            if ref is not None:
                reverse[ref].append(item["id"])
        return reverse

    def trace_upstream(self, start_id: str):
        chain = []
        visited: Set[str] = set()
        current = start_id

        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            current = self.id_map.get(current, {}).get("referenced_id")

        return list(reversed(chain))

    def trace_downstream(self, start_id: str):
        results = []

        def dfs(node_id: str, path, visited):
            if node_id in visited:
                return
            visited.add(node_id)
            children = self.reverse_map.get(node_id, [])
            if not children:
                results.append(path + [node_id])
                return
            for child in children:
                dfs(child, path + [node_id], visited.copy())

        dfs(start_id, [], set())
        return results

    def _sort_chain(self, chain):
        return sorted(chain, key=lambda x: self.id_map[x]["created_at"])

    def build_chains(self):
        chains = []
        for item in self.items:
            if item.get("referenced_id") is None:
                downstream = self.trace_downstream(item["id"])
                for chain in downstream:
                    ordered = self._sort_chain(chain)
                    chains.append({
                        "root_id": ordered[0],
                        "chain": ordered,
                        "length": len(ordered)
                    })
        return chains


def run_causality_agent(state: dict) -> dict:
    issue_id = state.get("issue_id")
    items = state.get("items", [])

    agent = CausalityAgent(items)
    chains = agent.build_chains()

    return {
        "issue_id": issue_id,
        "causality": {
            "chains": chains
        }
    }
