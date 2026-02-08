"""
CausalityAgent v0.4 (final, pipeline-aligned)

역할
- spike_event 기반 네트워크 구조 분석
- type 흐름, 인과 패턴, 확산 구조적 signal 추출
- 판단 / 분기 / 대응은 수행하지 않음

설계 원칙
- spike_event 단독으로 동작 가능
- sentiment_context는 Optional (설명 보조용)
- Router / Playbook으로 전달되는 출력은 CausalityAnalysisResult 스키마만 포함
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timezone

import networkx as nx


# =====================================================
# Pattern definition
# =====================================================

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


# =====================================================
# Core CausalityAgent (items 기반, spike_event 모름)
# =====================================================

class CausalityAgent:
    def __init__(
        self,
        items: List[Dict[str, Any]],
        patterns: Optional[List[CausalityPattern]] = None,
        enable_graph: bool = True,
        top_k: int = 10,
        max_paths: int = 5,
    ):
        self.items = items or []
        self.patterns = patterns if patterns is not None else DEFAULT_PATTERNS
        self.enable_graph = enable_graph
        self.top_k = top_k
        self.max_paths = max_paths

        self.id_map = {it["id"]: it for it in self.items if it.get("id") is not None}
        self.children = self._build_children()
        self.parents = self._build_parents()

    # -----------------------------
    # graph helpers
    # -----------------------------

    def _build_children(self) -> Dict[str, List[str]]:
        m = defaultdict(list)
        for it in self.items:
            if it.get("id") and it.get("referenced_id"):
                m[it["referenced_id"]].append(it["id"])
        return dict(m)

    def _build_parents(self) -> Dict[str, List[str]]:
        m = defaultdict(list)
        for it in self.items:
            if it.get("id") and it.get("referenced_id"):
                m[it["id"]].append(it["referenced_id"])
        return dict(m)

    def _graph_assumption(self) -> str:
        for parents in self.parents.values():
            if len(parents) > 1:
                return "dag"
        return "tree"

    def _roots(self) -> List[str]:
        return [nid for nid in self.id_map if nid not in self.parents]

    # -----------------------------
    # DAG-safe path tracing
    # -----------------------------

    def trace_downstream_paths(self, start_id: str) -> List[List[str]]:
        results: List[List[str]] = []

        def dfs(node_id: str, path: List[str]):
            if node_id in path:
                return
            children = self.children.get(node_id, [])
            if not children:
                results.append(path + [node_id])
                return
            for c in children:
                dfs(c, path + [node_id])

        dfs(start_id, [])
        return results

    # -----------------------------
    # type / event extraction
    # -----------------------------

    def _get_types(self, node_id: str) -> List[str]:
        return list(self.id_map.get(node_id, {}).get("types") or [])

    def extract_sequences(self, chain: List[str]) -> Dict[str, Any]:
        raw_seq, uniq_seq, seen = [], [], set()
        for nid in chain:
            for t in self._get_types(nid):
                raw_seq.append(t)
                if t not in seen:
                    uniq_seq.append(t)
                    seen.add(t)
        return {
            "type_sequence": uniq_seq,
            "raw_type_sequence": raw_seq,
        }

    def detect_events(self, chain: List[str]) -> List[Dict[str, Any]]:
        events = []
        prev_types: Set[str] = set()

        for nid in chain:
            cur_types = set(self._get_types(nid))
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
                                "node_id": nid,
                                "meaning": p.meaning,
                            }
                        )
            prev_types = cur_types

        return events

    # -----------------------------
    # network analysis
    # -----------------------------

    def build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for it in self.items:
            nid = it.get("id")
            if nid:
                G.add_node(nid)
        for it in self.items:
            if it.get("id") and it.get("referenced_id"):
                G.add_edge(it["referenced_id"], it["id"])
        return G

    def analyze_graph(self, G: nx.DiGraph) -> Dict[str, Any]:
        if G.number_of_nodes() == 0:
            return {
                "centralization": 0.0,
                "avg_degree": 0.0,
                "central_nodes": [],
            }

        deg = nx.degree_centrality(G)
        btw = nx.betweenness_centrality(G, normalized=True)

        central_nodes = sorted(
            [{"id": n, "degree": deg[n], "betweenness": btw[n]} for n in G.nodes],
            key=lambda x: max(x["degree"], x["betweenness"]),
            reverse=True,
        )[: self.top_k]

        avg_degree = G.number_of_edges() / max(1, G.number_of_nodes())
        centralization = (
            max(n["betweenness"] for n in central_nodes)
            if central_nodes else 0.0
        )

        return {
            "central_nodes": central_nodes,
            "centralization": float(min(1.0, centralization)),
            "avg_degree": float(avg_degree),
        }

    # -----------------------------
    # main run
    # -----------------------------

    def run(self) -> Dict[str, Any]:
        all_paths: List[List[str]] = []
        for r in self._roots():
            all_paths.extend(self.trace_downstream_paths(r))

        graph_info = {
            "centralization": 0.0,
            "avg_degree": 0.0,
            "central_nodes": [],
        }

        if self.enable_graph:
            G = self.build_graph()
            graph_info = self.analyze_graph(G)

        cascade_pattern = (
            "coordinated" if graph_info["centralization"] > 0.65
            else "viral" if graph_info["centralization"] > 0.35
            else "echo_chamber"
        )

        key_paths = [
            " → ".join(path) for path in all_paths[: self.max_paths]
        ]

        # -----------------------------
        # Router / Playbook 전달용 결과
        # -----------------------------
        result: Dict[str, Any] = {
            "trigger_source": "organic",
            "hub_accounts": [],
            "retweet_network_metrics": {
                "centralization": round(graph_info["centralization"], 2),
                "avg_degree": round(graph_info["avg_degree"], 2),
            },
            "cascade_pattern": cascade_pattern,
            "estimated_origin_time": None,
            "key_propagation_paths": key_paths,
        }

        # -----------------------------
        # debug / explanation only
        # -----------------------------
        result["debug"] = {
            "graph_assumption": self._graph_assumption(),

            "trigger_source_context": {
                "value": result["trigger_source"],
                "inferred_by": "causality_network",
                "confidence": "low",
            },

            "chains": [
                {
                    "chain": p,
                    **self.extract_sequences(p),
                    "events": self.detect_events(p),
                }
                for p in all_paths
            ],

            "graph_analysis": graph_info,

            "patterns": [
                {
                    "name": p.name,
                    "sequence": list(p.sequence),
                    "meaning": p.meaning,
                }
                for p in self.patterns
            ],
        }

        return result


# =====================================================
# Pipeline entrypoint (LangGraph용)
# =====================================================

def analyze_network(
    spike_event: Dict[str, Any],
    sentiment_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    spike_event → items 변환 책임은 여기서 수행
    sentiment_context는 Optional (보조 컨텍스트)
    """

    items: List[Dict[str, Any]] = []
    prev_id = None

    for idx, msg in enumerate(spike_event.get("messages", [])):
        item = {
            "id": f"n{idx}",
            "referenced_id": prev_id,
            "types": msg.get("types", []),
        }
        items.append(item)
        prev_id = item["id"]

    agent = CausalityAgent(items)
    return agent.run()
