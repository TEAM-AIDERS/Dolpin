"""
CausalityAgent v0.4
- 구조적 체인(v0.1)을 기반으로
- type 흐름, 인과 패턴, 구조적 signal을 해석
- 판단/대응/위험 판정은 수행하지 않음
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

import networkx as nx


@dataclass(frozen=True)
class CausalityPattern:
    name: str
    sequence: Tuple[str, ...]
    meaning: str


DEFAULT_PATTERNS: List[CausalityPattern] = [
    CausalityPattern("tone_shift", ("meme_positive", "meme_negative"), "밈 소비 톤 전환"),
    CausalityPattern("mobilization", ("meme_negative", "boycott_action"), "불만의 행동화"),
    CausalityPattern("fanwar_escalation", ("fanwar_target", "fanwar_action"), "갈등 격화"),
    CausalityPattern("irony_to_negative_meme", ("irony_cue", "meme_negative"), "조롱 확산"),
    CausalityPattern("evasion_spread", ("search_evasion", "search_evasion"), "검색 회피 확산"),
]


class CausalityAgent:
    def __init__(
        self,
        items: List[Dict[str, Any]],
        patterns: Optional[List[CausalityPattern]] = None,
        enable_graph: bool = True,
        top_k: int = 10,
    ):
        self.items = items or []
        self.patterns = patterns if patterns is not None else DEFAULT_PATTERNS
        self.enable_graph = enable_graph
        self.top_k = top_k

        self.id_map: Dict[str, Dict[str, Any]] = {it["id"]: it for it in self.items if "id" in it}
        self.reverse_map: Dict[str, List[str]] = self._build_reverse_map()

    def _build_reverse_map(self) -> Dict[str, List[str]]:
        rev = defaultdict(list)
        for it in self.items:
            it_id = it.get("id")
            ref = it.get("referenced_id")
            if it_id is None:
                continue
            if ref is not None:
                rev[ref].append(it_id)
        return dict(rev)

    def _get_created_at(self, node_id: str) -> Any:
        return self.id_map.get(node_id, {}).get("created_at")

    def _get_types(self, node_id: str) -> List[str]:
        t = self.id_map.get(node_id, {}).get("types", [])
        if t is None:
            return []
        return list(t)

    # sentiment is auxiliary metadata (NOT used for causality decisions)
    def _get_sentiment(self, node_id: str) -> Optional[str]:
        s = self.id_map.get(node_id, {}).get("sentiment")
        return s

    def _roots(self) -> List[str]:
        roots = []
        for it in self.items:
            if it.get("id") is None:
                continue
            if it.get("referenced_id") is None:
                roots.append(it["id"])
        return roots

    def trace_downstream_paths(self, start_id: str) -> List[List[str]]:
        results: List[List[str]] = []

        def dfs(node_id: str, path: List[str], visited: Set[str]):
            if node_id in visited:
                return
            visited.add(node_id)

            children = self.reverse_map.get(node_id, [])
            if not children:
                results.append(path + [node_id])
                return

            for child in children:
                dfs(child, path + [node_id], visited)

        dfs(start_id, [], set())
        return results

    def build_chains(self) -> List[Dict[str, Any]]:
        chains: List[Dict[str, Any]] = []
        for r in self._roots():
            for path in self.trace_downstream_paths(r):
                chains.append(
                    {
                        "root_id": r,
                        "chain": path,
                        "length": len(path),
                        "created_at_range": self._created_at_range(path),
                    }
                )
        return chains

    def _created_at_range(self, chain: List[str]) -> Dict[str, Any]:
        vals = [self._get_created_at(n) for n in chain if self._get_created_at(n) is not None]
        if not vals:
            return {"min": None, "max": None}
        return {"min": min(vals), "max": max(vals)}

    def extract_type_sequence(self, chain: List[str]) -> List[str]:
        seq: List[str] = []
        seen: Set[str] = set()
        for node_id in chain:
            types = self._get_types(node_id)
            for t in types:
                if t not in seen:
                    seq.append(t)
                    seen.add(t)
        return seq

    # TODO: support n-gram causality patterns (len >= 3)
    def detect_events(self, chain: List[str]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        prev_types: Set[str] = set()

        for node_id in chain:
            cur_types = set(self._get_types(node_id))
            if prev_types:
                for p in self.patterns:
                    if len(p.sequence) != 2:
                        continue
                    a, b = p.sequence
                    if a in prev_types and b in cur_types:
                        events.append(
                            {
                                "type": p.name,
                                "from": a,
                                "to": b,
                                "node_id": node_id,
                                "meaning": p.meaning,
                            }
                        )
            prev_types = cur_types

        return events

    def chain_signals(self, events: List[Dict[str, Any]]) -> List[str]:
        return sorted(list({e["type"] for e in events}))

    def build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for it in self.items:
            node_id = it.get("id")
            if node_id is None:
                continue
            G.add_node(
                node_id,
                created_at=it.get("created_at"),
                sentiment=it.get("sentiment"),
                types=list(it.get("types") or []),
            )
        for it in self.items:
            node_id = it.get("id")
            ref = it.get("referenced_id")
            if node_id is None or ref is None:
                continue
            if ref in G and node_id in G:
                G.add_edge(ref, node_id)
        return G

    def analyze_graph(self, G: nx.DiGraph) -> Dict[str, Any]:
        if G.number_of_nodes() == 0:
            return {
                "central_nodes": [],
                "components_count": 0,
                "metrics": {"degree_top": [], "betweenness_top": []},
                "signals": [],
            }

        deg = nx.degree_centrality(G)
        btw = nx.betweenness_centrality(G, normalized=True)

        top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        top_btw = sorted(btw.items(), key=lambda x: x[1], reverse=True)[: self.top_k]

        central_nodes = []
        picked: Set[str] = set()
        for node_id, _ in top_btw + top_deg:
            if node_id in picked:
                continue
            picked.add(node_id)
            central_nodes.append(
                {
                    "id": node_id,
                    "types": list(G.nodes[node_id].get("types") or []),
                    "sentiment": G.nodes[node_id].get("sentiment"),
                    "degree": float(deg.get(node_id, 0.0)),
                    "betweenness": float(btw.get(node_id, 0.0)),
                }
            )
            if len(central_nodes) >= self.top_k:
                break

        comps = list(nx.weakly_connected_components(G))
        components_count = len(comps)

        signals = self._graph_signals(central_nodes, components_count)

        return {
            "central_nodes": central_nodes,
            "components_count": components_count,
            "metrics": {
                "degree_top": [{"id": i, "score": float(s)} for i, s in top_deg],
                "betweenness_top": [{"id": i, "score": float(s)} for i, s in top_btw],
            },
            "signals": signals,
        }  # optional structural insight

    def _graph_signals(self, central_nodes: List[Dict[str, Any]], components_count: int) -> List[str]:
        signals: List[str] = []
        if components_count >= 2:
            signals.append("multi_component_flow")

        type_counter = Counter()
        for n in central_nodes:
            for t in n.get("types", []):
                type_counter[t] += 1

        if type_counter.get("meme_negative", 0) >= max(1, len(central_nodes) // 3):
            signals.append("negative_meme_hub")

        if type_counter.get("boycott_action", 0) > 0:
            signals.append("boycott_centrality_present")

        if type_counter.get("fanwar_action", 0) > 0 or type_counter.get("fanwar_target", 0) > 0:
            signals.append("fanwar_centrality_present")

        return signals

    def turning_points(
        self,
        chain: List[str],
        events: List[Dict[str, Any]],
        graph_betweenness: Optional[Dict[str, float]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        ev_by_node = defaultdict(list)
        for e in events:
            ev_by_node[e["node_id"]].append(e)

        scores = []
        for node_id in chain:
            ev_score = len(ev_by_node.get(node_id, []))
            btw_score = 0.0
            if graph_betweenness is not None:
                btw_score = float(graph_betweenness.get(node_id, 0.0))
            score = (ev_score * 10.0) + btw_score
            if score > 0:
                scores.append((node_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        out = []
        for node_id, score in scores[:top_k]:
            out.append(
                {
                    "node_id": node_id,
                    "score": float(score),
                    "events": ev_by_node.get(node_id, []),
                    "types": self._get_types(node_id),
                    "sentiment": self._get_sentiment(node_id),
                }
            )
        return out

    def run(self) -> Dict[str, Any]:
        chains = self.build_chains()

        G = None
        graph_analysis: Dict[str, Any] = {}
        btw_map: Optional[Dict[str, float]] = None
        if self.enable_graph:
            G = self.build_graph()
            graph_analysis = self.analyze_graph(G)
            if G.number_of_nodes() > 0:
                btw_map = nx.betweenness_centrality(G, normalized=True)

        enriched = []
        for c in chains:
            chain = c["chain"]
            type_seq = self.extract_type_sequence(chain)
            events = self.detect_events(chain)
            tps = self.turning_points(chain, events, btw_map, top_k=3)
            sigs = self.chain_signals(events)

            enriched.append(
                {
                    **c,
                    "type_sequence": type_seq,
                    "events": events,
                    "turning_points": tps,
                    "chain_signals": sigs,
                }
            )

        return {
            "chains": enriched,
            "graph_analysis": graph_analysis,
            "patterns": [{"name": p.name, "sequence": list(p.sequence), "meaning": p.meaning} for p in self.patterns],
        }


def run_causality_agent(state: dict) -> dict:
    issue_id = state.get("issue_id")
    items = state.get("items", [])

    agent = CausalityAgent(
        items=items,
        patterns=DEFAULT_PATTERNS,
        enable_graph=bool(state.get("enable_graph", True)),
        top_k=int(state.get("top_k", 10)),
    )
    result = agent.run()

    return {
        "issue_id": issue_id,
        "causality": result,
    }


if __name__ == "__main__":
    items = [
        {"id": "A", "referenced_id": None, "types": ["support_action"]},
        {"id": "B", "referenced_id": "A", "types": ["meme_positive"]},
        {"id": "C", "referenced_id": "B", "types": ["meme_negative"]},
    ]

    agent = CausalityAgent(items)
    out = agent.run()
    print(out["graph_analysis"])

    from pprint import pprint
    pprint(out["chains"])
