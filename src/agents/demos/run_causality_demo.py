"""
Demo script for running SentimentAgentAdapter → CausalityAgent v0.4
Shows how sentiment-derived types form causality chains.
Not used in Dolphin production pipeline.
"""

from sentiment_agent import build_agent
from sentiment_adapter import SentimentAgentAdapter
from causality_agent_v0_4 import run_causality_agent


def to_router_schema(result: dict) -> dict:
    return {
        "trigger_source": result["trigger_source"],
        "hub_accounts": result["hub_accounts"],
        "retweet_network_metrics": result["retweet_network_metrics"],
        "cascade_pattern": result["cascade_pattern"],
        "estimated_origin_time": result["estimated_origin_time"],
        "key_propagation_paths": result["key_propagation_paths"],
    }


if __name__ == "__main__":

    # 1. 테스트용 텍스트 시퀀스 (시간 흐름 가정)
    texts = [
        "우리 애들 오늘 무대 미쳤다 진짜 최고야 ㅋㅋ",
        "아 네네 또 레전드라구요 ㅋㅋ",
        "기대 많이 했는데 솔직히 좀 실망스럽다",
        "이번 활동은 불매한다. 이제 안 본다",
    ]

    # 2. SentimentAgent + Adapter 초기화
    sentiment_agent = build_agent(
        model_path="./sentiment_ft",
        lexicon_path="lexicon_master_final.csv",
        device="cpu",
    )
    adapter = SentimentAgentAdapter(sentiment_agent)

    # 3. Sentiment 결과를 Causality 입력 items로 변환
    items = []
    prev_id = None

    for idx, text in enumerate(texts):
        out = adapter.run(text)

        item = {
            "id": f"n{idx}",
            "referenced_id": prev_id,
            "types": out.get("types", []),
            "sentiment": out["sentiment"].get("dominant_sentiment"),
        }

        items.append(item)
        prev_id = item["id"]

    # 4. CausalityAgent 실행
    state = {
        "issue_id": "demo_issue_001",
        "items": items,
        "enable_graph": True,
        "top_k": 5,
        "include_analysis": True,
        "include_raw_types": True,
    }

    result = run_causality_agent(state)

    causality_full = result["causality"]
    causality_for_router = to_router_schema(causality_full)

    # 5-1. Router / Playbook으로 전달될 최종 스키마
    print("\n=== Router Input (CausalityAnalysisResult) ===")
    from pprint import pprint
    pprint(causality_for_router)

    # 5-2. (디버그용) Causality 내부 분석 결과
    print("\n=== Debug: Causality Chains ===")
    for c in causality_full.get("debug", {}).get("chains", []):
        pprint(c)

    print("\n=== Debug: Graph Analysis ===")
    pprint(causality_full.get("debug", {}).get("graph_analysis"))
