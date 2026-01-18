"""
LangGraph 워크플로우 정의
DOLPIN 분석 파이프라인의 전체 그래프를 구성합니다.

버전: v1.1 (260114)
"""

from langgraph.graph import StateGraph, END
from .state import AnalysisState
from .nodes import (
    spike_analyzer_node,
    router1_node,
    sentiment_node,
    router2_node,
    causality_node,
    router3_node,
    legal_rag_node,
    amplification_node,
    playbook_node,
    exec_brief_node
)
from .edges import (
    should_continue_after_router1,
    should_continue_after_router2,
    should_continue_after_router3
)


def create_workflow() -> StateGraph:
    """
    DOLPIN 분석 워크플로우 생성
    
    Returns:
        StateGraph: 컴파일 가능한 LangGraph 워크플로우
    """
    # StateGraph 초기화
    workflow = StateGraph(AnalysisState)
    
    # ============================================================
    # 노드 추가
    # ============================================================
    workflow.add_node("spike_analyzer", spike_analyzer_node)
    workflow.add_node("router1", router1_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("router2", router2_node)
    workflow.add_node("causality", causality_node)
    workflow.add_node("router3", router3_node)
    workflow.add_node("legal_rag", legal_rag_node)
    workflow.add_node("amplification", amplification_node)
    workflow.add_node("playbook", playbook_node)
    workflow.add_node("exec_brief", exec_brief_node)
    
    # ============================================================
    # 엣지 추가 (워크플로우 정의)
    # ============================================================
    
    # 시작: SpikeAnalyzer
    workflow.set_entry_point("spike_analyzer")
    
    # SpikeAnalyzer → Router1
    workflow.add_edge("spike_analyzer", "router1")
    
    # Router1 → (조건부) Sentiment or END
    workflow.add_conditional_edges(
        "router1",
        should_continue_after_router1,
        {
            "sentiment": "sentiment",
            "end": END
        }
    )
    
    # Sentiment → Router2
    workflow.add_edge("sentiment", "router2")
    
    # Router2 → (조건부) Playbook or Causality
    workflow.add_conditional_edges(
        "router2",
        should_continue_after_router2,
        {
            "playbook": "playbook",
            "causality": "causality"
        }
    )
    
    # Causality → Router3
    workflow.add_edge("causality", "router3")
    
    # Router3 → (조건부) Legal RAG or Amplification
    workflow.add_conditional_edges(
        "router3",
        should_continue_after_router3,
        {
            "legal_rag": "legal_rag",
            "amplification": "amplification"
        }
    )
    
    # Legal RAG → Playbook
    workflow.add_edge("legal_rag", "playbook")
    
    # Amplification → Playbook
    workflow.add_edge("amplification", "playbook")
    
    # Playbook → ExecBrief
    workflow.add_edge("playbook", "exec_brief")
    
    # ExecBrief → END
    workflow.add_edge("exec_brief", END)
    
    return workflow


def compile_workflow():
    """
    워크플로우 컴파일
    
    Returns:
        CompiledGraph: 실행 가능한 그래프
    """
    workflow = create_workflow()
    return workflow.compile()