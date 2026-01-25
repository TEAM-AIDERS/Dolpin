"""
MCP Client 사용 테스트

MCP Client를 통해 Lexicon Server에 접근하는 기능 테스트입니다.
실제 node 함수들은 nodes.py에 정의되어 있습니다.

실행 방법:
    python tests/test_workflow_example.py
    pytest tests/test_workflow_example.py -v

버전: v1.0 (260125)
"""

import sys
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.server.mcp_client import MCPClient


def test_mcp_client_initialization():
    """MCP Client 초기화 테스트"""
    mcp = MCPClient("custom_lexicon.csv")
    assert mcp is not None
    assert hasattr(mcp, 'lexicon_analyze')
    print("✅ MCP Client 초기화 성공")


def test_lexicon_analyze():
    """Lexicon 분석 테스트"""
    mcp = MCPClient("custom_lexicon.csv")
    
    text = "이번 활동은 불매할게 진짜 탈빠하겠어"
    result = mcp.lexicon_analyze(text)
    
    assert result is not None
    assert 'matches' in result
    assert 'aggregated_signals' in result
    assert len(result['matches']) > 0  # 매칭된 용어가 있어야 함
    
    print(f"✅ Lexicon 분석 성공")
    print(f"   매칭 용어: {[m['term'] for m in result['matches']]}")
    print(f"   위험도: {result['aggregated_signals'].get('risk_flags', [])}")


def test_sentiment_context():
    """Sentiment 컨텍스트 생성 테스트"""
    mcp = MCPClient("custom_lexicon.csv")
    
    text = "이번 활동은 불매할게 진짜 탈빠하겠어"
    sentiment_context = mcp.prepare_sentiment_context(text)
    
    assert sentiment_context is not None
    assert 'sentiment_signals' in sentiment_context
    
    print(f"✅ Sentiment 컨텍스트 생성 성공")
    print(f"   신호 개수: {len(sentiment_context.get('sentiment_signals', []))}")


def test_routing_context():
    """Routing 컨텍스트 생성 테스트"""
    mcp = MCPClient("custom_lexicon.csv")
    
    text = "이번 활동은 불매할게 진짜 탈빠하겠어"
    routing_context = mcp.prepare_routing_context(text)
    
    assert routing_context is not None
    assert 'routing_signals' in routing_context
    
    print(f"✅ Routing 컨텍스트 생성 성공")
    print(f"   신호 개수: {len(routing_context.get('routing_signals', []))}")


def test_causality_context():
    """Causality 컨텍스트 생성 테스트"""
    mcp = MCPClient("custom_lexicon.csv")
    
    text = "이번 활동은 불매할게 진짜 탈빠하겠어"
    causality_context = mcp.prepare_causality_context(text)
    
    assert causality_context is not None
    assert 'causality_signals' in causality_context
    
    print(f"✅ Causality 컨텍스트 생성 성공")
    print(f"   신호 개수: {len(causality_context.get('causality_signals', []))}")


def test_playbook_context():
    """Playbook 컨텍스트 생성 테스트"""
    mcp = MCPClient("custom_lexicon.csv")
    
    text = "이번 활동은 불매할게 진짜 탈빠하겠어"
    playbook_context = mcp.prepare_playbook_context(text)
    
    assert playbook_context is not None
    assert 'action_signals' in playbook_context
    
    print(f"✅ Playbook 컨텍스트 생성 성공")
    print(f"   신호 개수: {len(playbook_context.get('action_signals', []))}")


def demo_mcp_client_full_workflow():
    """MCP Client 전체 워크플로우 데모"""
    print("\n" + "=" * 70)
    print("MCP Client 사용 예제")
    print("=" * 70)
    
    mcp = MCPClient("custom_lexicon.csv")
    
    text = "이번 활동은 불매할게 진짜 탈빠하겠어"
    state = {
        "text": text,
        "source": "twitter",
    }
    
    print(f"\n입력 텍스트: {state['text']}\n")
    
    # 모든 메서드 호출
    print("=" * 70)
    print("MCP Client 메서드 호출 결과")
    print("=" * 70)
    
    lexicon_result = mcp.lexicon_analyze(state['text'])
    print(f"\n[1] lexicon_analyze()")
    print(f"    매칭 용어: {[m['term'] for m in lexicon_result.get('matches', [])]}")
    print(f"    위험도: {lexicon_result['aggregated_signals'].get('risk_flags', [])}")
    
    sentiment_context = mcp.prepare_sentiment_context(state['text'])
    print(f"\n[2] prepare_sentiment_context()")
    print(f"    신호 개수: {len(sentiment_context.get('sentiment_signals', []))}")
    
    routing_context = mcp.prepare_routing_context(state['text'])
    print(f"\n[3] prepare_routing_context()")
    print(f"    신호 개수: {len(routing_context.get('routing_signals', []))}")
    
    causality_context = mcp.prepare_causality_context(state['text'])
    print(f"\n[4] prepare_causality_context()")
    print(f"    신호 개수: {len(causality_context.get('causality_signals', []))}")
    
    playbook_context = mcp.prepare_playbook_context(state['text'])
    print(f"\n[5] prepare_playbook_context()")
    print(f"    신호 개수: {len(playbook_context.get('action_signals', []))}")
    
    print("\n" + "=" * 70)
    print("✅ MCP Client 예제 완료")
    print("=" * 70)
    print("""
주요 포인트:
  - MCPClient는 lexicon_server.py의 LexiconServer와 통신합니다
  - 각 prepare_*_context() 메서드는 에이전트별 신호 컨텍스트를 반환합니다
  - 실제 워크플로우에서는 nodes.py의 노드 함수들이 이를 호출합니다
  
참고:
  - nodes.py: 실제 LangGraph 노드 구현 (spike_analyzer_node 등)
  - lexicon_server.py: CSV 기반 어휘 분석
  - state.py: DolpinState 정의 (워크플로우 상태 구조)
    """)


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    # pytest 없이 직접 실행할 때
    test_mcp_client_initialization()
    test_lexicon_analyze()
    test_sentiment_context()
    test_routing_context()
    test_causality_context()
    test_playbook_context()
    demo_mcp_client_full_workflow()
