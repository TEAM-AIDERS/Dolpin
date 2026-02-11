"""
Mock 테스트 스크립트
Stub 구현으로 전체 워크플로우를 테스트합니다.

실행 방법:
    python tests/test_mcp_mock.py
    또는
    pytest tests/test_mcp_mock.py -v
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dolpin_langgraph.graph import compile_workflow
from src.dolpin_langgraph.nodes import router1_node
from src.dolpin_langgraph.state import AnalysisState, SpikeEvent, Message


def create_mock_spike_event() -> SpikeEvent:
    """Mock SpikeEvent 생성"""
    return {
        "keyword": "뉴진스",
        "spike_rate": 3.5,
        "baseline": 100,
        "current_volume": 350,
        "detected_at": "2026-01-10T10:00:00Z",
        "time_window": "1h",
        "messages": [
            {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "source_message_id": "187654321",
                "text": "뉴진스 컴백 대박!",
                "timestamp": "2026-01-10T10:00:00Z",
                "source": "twitter",
                "author_id": "user_123",
                "metrics": {"likes": 100, "retweets": 50},
                "is_anonymized": False,
                "detected_language": "ko"
            },
            {
                "id": "550e8400-e29b-41d4-a716-446655440002",
                "source_message_id": "187654322",
                "text": "뉴진스 최고야 ㅠㅠ",
                "timestamp": "2026-01-10T10:05:00Z",
                "source": "twitter",
                "author_id": "user_456",
                "metrics": {"likes": 200, "retweets": 80},
                "is_anonymized": False,
                "detected_language": "ko"
            }
        ],
        "raw_kafka_message_ids": [
            "550e8400-e29b-41d4-a716-446655440000"
        ]
    }


def create_initial_state(spike_event: SpikeEvent, trace_id: str = "test-trace-001") -> AnalysisState:
    """초기 State 생성"""
    return {
        # 입력
        "spike_event": spike_event,
        
        # 메타데이터
        "trace_id": trace_id,
        "workflow_start_time": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        
        # Router 결정 (초기값 None)
        "route1_decision": None,
        "route2_decision": None,
        "route3_decision": None,
        "positive_viral_detected": None,
        
        # 각 에이전트 결과 (초기값 None)
        "spike_analysis": None,
        "sentiment_result": None,
        "causality_result": None,
        "legal_risk": None,
        "amplification_summary": None,
        "playbook": None,
        
        # 노드 인사이트
        "node_insights": {},
        
        # 최종 출력
        "executive_brief": None,
        
        # 에러 로그
        "error_logs": [],
        
        # Skip 처리
        "skipped": False,
        "skip_reason": None
    }


def print_state_summary(state: AnalysisState):
    """State 요약 출력"""
    print("\n" + "="*80)
    print("📊 DOLPIN 분석 워크플로우 실행 결과")
    print("="*80)
    
    # 메타데이터
    print(f"\n🔍 Trace ID: {state['trace_id']}")
    print(f"⏱️  시작 시간: {state['workflow_start_time']}")
    
    # Router 결정
    print(f"\n🔀 Router 결정:")
    print(f"   Router 1차: {state.get('route1_decision')}")
    print(f"   Router 2차: {state.get('route2_decision')}")
    print(f"   Router 3차: {state.get('route3_decision')}")
    print(f"   긍정 바이럴: {state.get('positive_viral_detected')}")
    
    # Skip 여부
    if state.get("skipped"):
        print(f"\n⏭️  Skip: {state['skip_reason']}")
        return
    
    # 각 노드 인사이트
    if state.get("node_insights"):
        print(f"\n💡 노드 인사이트:")
        for node, insight in state["node_insights"].items():
            print(f"   • {node}: {insight}")
    
    # 최종 브리핑
    if state.get("executive_brief"):
        brief = state["executive_brief"]
        print(f"\n📋 임원 브리핑:")
        print(f"   요약: {brief['summary']}")
        print(f"   심각도: {brief['severity_score']}/10")
        print(f"   트렌드: {brief['trend_direction']}")
        print(f"   극성: {brief['issue_polarity']}")
        print(f"   소요 시간: {brief['analysis_duration_seconds']}초")
        
        if brief.get("spike_summary"):
            print(f"\n   📈 Spike: {brief['spike_summary']}")
        if brief.get("sentiment_summary"):
            print(f"   😊 Sentiment: {brief['sentiment_summary']}")
        if brief.get("legal_summary"):
            print(f"   ⚖️  Legal: {brief['legal_summary']}")
        if brief.get("action_summary"):
            print(f"   🎯 Action: {brief['action_summary']}")
        if brief.get("opportunity_summary"):
            print(f"   🌟 Opportunity: {brief['opportunity_summary']}")
        
        print(f"\n   분석 상태:")
        for agent, status in brief["analysis_status"].items():
            emoji = "✅" if status == "success" else "⏭️" if status == "skipped" else "❌"
            print(f"      {emoji} {agent}: {status}")
        
        if brief.get("user_message"):
            print(f"\n   ⚠️  사용자 메시지: {brief['user_message']}")
    
    # 에러 로그
    if state.get("error_logs"):
        print(f"\n❌ 에러 로그 ({len(state['error_logs'])}건):")
        for error in state["error_logs"]:
            print(f"   • [{error['stage']}] {error['error_type']}: {error['message']}")
    
    print("\n" + "="*80 + "\n")


def save_result_to_file(state: AnalysisState, filename: str = "mock_result.json"):
    """결과를 JSON 파일로 저장"""
    # tests/outputs 디렉토리 생성
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 경로
    filepath = output_dir / filename
    
    # datetime 객체를 문자열로 변환
    def convert_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat().replace('+00:00', 'Z')
        return obj
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=convert_datetime)
    
    print(f"💾 결과가 {filepath}에 저장되었습니다.")


def run_test_case_1_positive_viral():
    """
    테스트 케이스 1: 긍정 바이럴 (Amplification 경로)
    
    조건:
    - spike_rate: 3.5 (high)
    - spike_nature: "positive"
    - support: 0.6 (>= 0.5)
    - actionability_score: 0.7 (>= 0.6)
    
    예상 경로:
    SpikeAnalyzer → Router1 (analyze) → Sentiment → Router2 (full_analysis) 
    → Causality → Router3 (amplification) → Amplification → Playbook → ExecBrief
    
    TODO: 실제 Agent 구현 시 Stub 데이터를 실제 분석 결과로 교체
    """
    print("\n🧪 테스트 케이스 1: 긍정 바이럴 (Amplification 경로)")
    print("-" * 80)
    
    spike_event = create_mock_spike_event()
    initial_state = create_initial_state(spike_event, trace_id="test-trace-001")
    
    # 워크플로우 컴파일 및 실행
    app = compile_workflow()
    final_state = app.invoke(initial_state)
    
    # 결과 출력
    print_state_summary(final_state)
    
    # 검증
    assert final_state["route1_decision"] == "analyze", "Router 1차가 analyze여야 함"
    assert final_state["route2_decision"] == "full_analysis", "Router 2차가 full_analysis여야 함"
    assert final_state["route3_decision"] == "amplification", "Router 3차가 amplification이어야 함"
    assert final_state["positive_viral_detected"] == True, "긍정 바이럴 감지되어야 함"
    
    print("✅ 테스트 케이스 1 통과!")
    
    return final_state


def run_test_case_2_skip():
    """
    테스트 케이스 2: Skip 경로
    
    조건:
    - is_significant: False (급등 미달)
    
    예상 경로:
    SpikeAnalyzer → Router1 (skip) → 종료
    
    TODO: 실제 SpikeAnalyzer 구현 시 spike_rate 기반으로 is_significant 판단
    """
    print("\n🧪 테스트 케이스 2: Skip 경로")
    print("-" * 80)
    
    spike_event = create_mock_spike_event()
    spike_event["spike_rate"] = 1.1  # 낮은 급등률
    
    initial_state = create_initial_state(spike_event, trace_id="test-trace-002")
    
    # TODO: 현재 Stub은 항상 is_significant=True
    # 실제 구현 시 spike_rate 기반 판단 필요
    print("⚠️  현재 Stub은 항상 is_significant=True로 설정되어 있습니다.")
    print("   실제 SpikeAnalyzer 구현 시 spike_rate 기반으로 판단할 예정입니다.")
    
    # 수동으로 skip 상태 설정 (테스트용)
    initial_state["spike_analysis"] = {
        "is_significant": False,
        "spike_rate": 1.1,
        "spike_type": "organic",
        "spike_nature": "neutral",
        "peak_timestamp": "2026-01-10T10:30:00Z",
        "duration_minutes": 30,
        "confidence": 0.9,
        "actionability_score": 0.1,
        "data_completeness": "confirmed",
        "partial_data_warning": None,
        "viral_indicators": {
            "is_trending": False,
            "has_breakout": False,
            "max_rise_rate": "+10%",
            "breakout_queries": [],
            "cross_platform": ["twitter"],
            "international_reach": 0.0
        }
    }
    
    # Router1만 실행
    from src.dolpin_langgraph.nodes import router1_node
    final_state = router1_node(initial_state)
    
    print_state_summary(final_state)
    
    assert final_state["route1_decision"] == "skip", "Router 1차가 skip이어야 함"
    assert final_state["skipped"] == True, "skipped 플래그가 True여야 함"
    
    print("✅ 테스트 케이스 2 통과!")
    
    return final_state


def run_test_case_3_sentiment_only():
    """
    테스트 케이스 3: Sentiment Only 경로 (낮은 actionability)
    
    조건:
    - actionability_score: 0.2 (< 0.3)
    - 단순 팬 반응, 위기/기회 신호 없음
    
    예상 경로:
    SpikeAnalyzer → Router1 (analyze) → Sentiment → Router2 (sentiment_only) 
    → Playbook → ExecBrief
    
    ✅ nodes.py는 수정하지 않고,
    테스트 코드에서 state를 직접 세팅하여 분기 조건을 만든다.
    """
    print("\n🧪 테스트 케이스 3: Sentiment Only 경로 (낮은 actionability)")
    print("-" * 80)
    
    spike_event = create_mock_spike_event()
    spike_event["keyword"] = "에스파 예쁘다"  # 단순 팬 반응
    initial_state = create_initial_state(spike_event, trace_id="test-trace-003")
    
    # TODO: nodes.py의 spike_analyzer_node Stub 수정 필요
    # actionability_score를 0.2로 설정
    initial_state["spike_analysis"] = {
        "is_significant": True,
        "spike_rate": 2.0,
        "spike_type": "organic",
        "spike_nature": "positive",
        "peak_timestamp": "2026-01-10T10:30:00Z",
        "duration_minutes": 30,
        "confidence": 0.9,
        "actionability_score": 0.2,  # 핵심: 낮은 actionability
        "data_completeness": "confirmed",
        "partial_data_warning": None,
        "viral_indicators": {
            "is_trending": False,
            "has_breakout": False,
            "max_rise_rate": "+20%",
            "breakout_queries": [],
            "cross_platform": ["twitter"],
            "international_reach": 0.1
        }
    }

    initial_state["sentiment_result"] = {
        "dominant_sentiment": "support",
        "sentiment_distribution": {
            "support": 0.8,
            "neutral": 0.2
        },
        "confidence": 0.85,
        "sentiment_shift": "stable",
        "analyzed_count": 10,
        "representative_messages": {
            "support": ["에스파 진짜 예쁘다 ㅠㅠ"]
        }
    }

    app = compile_workflow()
    final_state = app.invoke(initial_state)

    print_state_summary(final_state)
    
    assert final_state["route1_decision"] == "analyze", \
        "Router 1차는 analyze여야 함"

    assert final_state["route2_decision"] == "sentiment_only", \
        "Router 2차는 sentiment_only여야 함"

    assert final_state["route3_decision"] is None, \
        "Router 3차는 수행되지 않아야 함"

    print("✅ 테스트 케이스 3 통과!")

    return final_state

def run_test_case_4_legal_crisis():
    """
    테스트 케이스 4: Legal 경로 (위기 신호 - 보이콧)
    
    조건:
    - actionability_score: 0.5 (중간)
    - boycott: 0.3 (>= 0.2) → 위기 신호
    - spike_nature: "negative"
    
    예상 경로:
    SpikeAnalyzer → Router1 (analyze) → Sentiment → Router2 (full_analysis)
    → Causality → Router3 (legal) → Legal RAG → Playbook → ExecBrief
    
    ⚠️ nodes.py는 수정하지 않음
    테스트 코드에서 spike_analysis / sentiment_result를 override하여
    Legal 경로를 강제로 유도함
    """
    print("\n🧪 테스트 케이스 4: Legal 경로 (위기 신호 - 보이콧)")
    print("-" * 80)
    
    spike_event = create_mock_spike_event()
    spike_event["keyword"] = "XX 불매"
    initial_state = create_initial_state(spike_event, trace_id="test-trace-004")
    
    # TODO: nodes.py Stub 수정 필요
    # spike_analyzer_node: spike_nature="negative", actionability_score=0.5
    # sentiment_node: boycott=0.3
    initial_state["spike_analysis"] = {
        "is_significant": True,
        "spike_rate": 2.5,
        "spike_type": "organic",
        "spike_nature": "negative",   # 위기
        "peak_timestamp": "2026-01-10T10:30:00Z",
        "duration_minutes": 60,
        "confidence": 0.9,
        "actionability_score": 0.5,   # 중간 구간
        "data_completeness": "confirmed",
        "partial_data_warning": None,
        "viral_indicators": {
            "is_trending": True,
            "has_breakout": False,
            "max_rise_rate": "+150%",
            "breakout_queries": [],
            "cross_platform": ["twitter"],
            "international_reach": 0.2
        }
    }

    initial_state["sentiment_result"] = {
        "dominant_sentiment": "boycott",
        "confidence": 0.8,
        "sentiment_distribution": {
            "boycott": 0.3,
            "support": 0.2,
            "neutral": 0.3,
            "disappointment": 0.2
        },
        "analyzed_count": 50,
        "sentiment_shift": "worsening",
        "has_mixed_sentiment": True,
        "representative_messages": {
            "boycott": ["이번 활동은 불매한다"]
        }
    }

    # Router2 실행 (full_analysis 유도)
    from src.dolpin_langgraph.nodes import router2_node
    state_after_r2 = router2_node(initial_state)

    # Causality stub 실행
    from src.dolpin_langgraph.nodes import causality_node
    state_after_causality = causality_node(state_after_r2)

    # Router3 실행 (Legal 유도)
    # positive_viral_detected = False 상태
    from src.dolpin_langgraph.nodes import router3_node
    final_state = router3_node(state_after_causality)

    print_state_summary(final_state)

    assert final_state["route2_decision"] == "full_analysis", \
        "Router 2차가 full_analysis여야 함"

    assert final_state["route3_decision"] == "legal", \
        "Router 3차가 legal이어야 함"

    print("✅ 테스트 케이스 4 통과!")

    return final_state

def run_test_case_5_legal_keyword():
    """
    테스트 케이스 5: Legal 경로 (법적 키워드 감지)
    
    조건:
    - dominant_sentiment: "meme_negative"
    - 명예훼손 키워드 포함
    - legal_risk: high_risk
    
    예상 경로:
    SpikeAnalyzer → Router1 (analyze) → Sentiment → Router2 (full_analysis)
    → Causality → Router3 (legal) → Legal RAG (high_risk) → Playbook → ExecBrief
    
    ⚠️ nodes.py 수정 없이 테스트 코드에서 override
    """
    print("\n🧪 테스트 케이스 5: Legal 경로 (법적 키워드)")
    print("-" * 80)
    
    spike_event = create_mock_spike_event()
    spike_event["keyword"] = "XX 명예훼손"
    initial_state = create_initial_state(spike_event, trace_id="test-trace-005")
    
    # sentiment_node: dominant_sentiment="meme_negative"
    # legal_rag_node: clearance_status="high_risk"

    # SpikeAnalyzer 결과 강제 세팅
    initial_state["spike_analysis"] = {
        "is_significant": True,
        "spike_rate": 2.8,
        "spike_type": "organic",
        "spike_nature": "negative",
        "peak_timestamp": "2026-01-10T10:30:00Z",
        "duration_minutes": 45,
        "confidence": 0.9,
        "actionability_score": 0.6,  # full_analysis 유도
        "data_completeness": "confirmed",
        "partial_data_warning": None,
        "viral_indicators": {
            "is_trending": True,
            "has_breakout": False,
            "max_rise_rate": "+180%",
            "breakout_queries": [],
            "cross_platform": ["twitter"],
            "international_reach": 0.1
        }
    }

    # Sentiment 결과 강제 세팅
    # meme_negative → 위기 신호
    initial_state["sentiment_result"] = {
        "dominant_sentiment": "meme_negative",
        "confidence": 0.75,
        "sentiment_distribution": {
            "meme_negative": 0.4,
            "support": 0.2,
            "neutral": 0.3,
            "disappointment": 0.1
        },
        "analyzed_count": 40,
        "sentiment_shift": "worsening",
        "has_mixed_sentiment": True,
        "representative_messages": {
            "meme_negative": ["이건 진짜 명예훼손이다"]
        }
    }

    # Router2 실행 → full_analysis
    from src.dolpin_langgraph.nodes import router2_node
    state_after_r2 = router2_node(initial_state)

    # Causality 실행 (stub)
    from src.dolpin_langgraph.nodes import causality_node
    state_after_causality = causality_node(state_after_r2)

    # Router3 실행 → legal
    from src.dolpin_langgraph.nodes import router3_node
    state_after_r3 = router3_node(state_after_causality)

    # Legal RAG 결과 강제 high_risk 설정
    state_after_r3["legal_risk"] = {
        "overall_risk_level": "high",
        "clearance_status": "high_risk",
        "confidence": 0.95,
        "rag_required": True,
        "rag_performed": True,
        "rag_confidence": 0.9,
        "risk_assessment": "명예훼손 소지 있음",
        "recommended_action": ["법무팀 검토 요청"],
        "referenced_documents": ["형법 제307조"],
        "signals": {
            "legal_keywords_detected": True,
            "matched_keywords": ["명예훼손"],
            "reason": "법적 키워드 감지"
        }
    }

    final_state = state_after_r3

    print_state_summary(final_state)

    assert final_state["route2_decision"] == "full_analysis", \
        "Router 2차가 full_analysis여야 함"

    assert final_state["route3_decision"] == "legal", \
        "Router 3차가 legal이어야 함"

    assert final_state["legal_risk"]["clearance_status"] == "high_risk", \
        "Legal RAG 결과가 high_risk여야 함"

    print("✅ 테스트 케이스 5 통과!")

    return final_state


def main():
    """메인 테스트 실행"""
    print("\n" + "="*80)
    print("🚀 DOLPIN Mock 워크플로우 테스트 시작")
    print("="*80)
    print("\n💡 현재 실행 가능한 테스트: 1, 2")
    print("   TODO: 테스트 3, 4, 5는 nodes.py Stub 수정 후 실행 가능\n")
    
    # 테스트 케이스 1: 긍정 바이럴
    result1 = run_test_case_1_positive_viral()
    if result1:
        save_result_to_file(result1, "mock_result_case1_positive_viral.json")
    
    # 테스트 케이스 2: Skip
    result2 = run_test_case_2_skip()
    if result2:
        save_result_to_file(result2, "mock_result_case2_skip.json")
    
    # 테스트 케이스 3: Sentiment Only
    run_test_case_3_sentiment_only()
    
    # 테스트 케이스 4: Legal - Crisis
    run_test_case_4_legal_crisis()
    
    # 테스트 케이스 5: Legal - Keyword
    run_test_case_5_legal_keyword()
    
    print("\n" + "="*80)
    print("🎉 모든 Mock 테스트 실행 완료! (5/5)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
