"""
MCP Client - 모든 MCP 서버 호출을 관리하는 단일 진입점

역할:
- Custom Lexicon MCP 호출 및 결과 전달
- Legal MCP 호출 (향후 확장)
- LangGraph 워크플로우 전반에서 통일된 MCP 인터페이스 제공

구조:
graph.py → nodes.py → mcp_client.py → 각 MCP 서버 (Lexicon, Legal 등)

Singleton 관리 방식:
- MCPClient 인스턴스는 모듈 전역 get_mcp_client()를 통해 단일 인스턴스로 관리됨
- 같은 프로세스 내에서는 항상 동일한 MCPClient 인스턴스를 반환
- Lexicon CSV는 최초 초기화 시 1회만 로딩되며 이후 캐시된 서버 인스턴스를 재사용
- 프로세스 종료 시에만 인스턴스가 소멸되며, 재시작 시 새로 초기화됨

설계 의도:
- MCP 서버 초기화 비용(파일 로딩, 인덱싱, 정규식 컴파일)을 최소화
- Agent 노드에서는 MCPClient의 메서드만 호출하도록 하여 의존성을 단순화
- 향후 Legal MCP, 기타 MCP 서버 확장 시 인터페이스 변경 최소화

버전: v2 (260203) - 전역 get_mcp_client 기반 Singleton 정리
"""


import logging, threading
from typing import Dict, Any, Optional, List

from src.server.lexicon_server import LexiconServer

logger = logging.getLogger(__name__)


# ============================================================
# MCP Client - Singleton 패턴
# ============================================================

class MCPClient:
    def __init__(self, lexicon_csv_path: str = "custom_lexicon.csv"):
        self._lexicon_csv_path = lexicon_csv_path
        self._lexicon_server = LexiconServer(lexicon_csv_path)
        self.legal_server = None
        logger.info(f"MCPClient initialized with {lexicon_csv_path}")

    @property
    def lexicon_server(self) -> LexiconServer:
        return self._lexicon_server

    
    # ========== Custom Lexicon MCP 도구 ==========
    
    def lexicon_lookup(self, term: str) -> Optional[Dict[str, Any]]:
        """
        팬덤 표현 단일 조회
        
        사용: state.matched_entry = mcp.lexicon_lookup(term)
        """
        return self.lexicon_server.lookup_term(term)
    
    def lexicon_analyze(self, text: str) -> Dict[str, Any]:
        """
        텍스트 분석 및 팬덤 표현 추출
        """
        result = self.lexicon_server.execute_tool("analyze_text", {"text": text})
        return result
    
    # ========== Agent용 컨텍스트 준비 ==========
    
    def prepare_sentiment_context(self, text: str) -> Dict[str, Any]:
        """
        SentimentAgent용 컨텍스트 준비
        
        사용: sentiment_ctx = mcp.prepare_sentiment_context(text)
              sentiment_result = sentiment_agent.process(sentiment_ctx)
        """
        return self.lexicon_server.execute_tool("get_sentiment_context", {"text": text})
    
    def prepare_routing_context(self, text: str) -> Dict[str, Any]:
        """Router 컨텍스트 준비"""
        return self.lexicon_server.execute_tool("get_routing_context", {"text": text})
    
    def prepare_causality_context(self, text: str) -> Dict[str, Any]:
        """CausalityAgent용 컨텍스트 준비"""
        return self.lexicon_server.execute_tool("get_causality_context", {"text": text})
    
    def prepare_playbook_context(self, text: str) -> Dict[str, Any]:
        """PlaybookAgent용 컨텍스트 준비"""
        return self.lexicon_server.execute_tool("get_playbook_context", {"text": text})
    
    # ========== Legal MCP 도구 (향후) ==========
    
    def legal_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legal RAG 체크
        
        사용: legal_result = mcp.legal_check(state)
              if legal_result['action_required']:
                  # Legal 대응 필요
        """
        if self.legal_server is None:
            return {"error": "Legal server not initialized", "action_required": False}
        
        return self.legal_server.check(context)
    
    # ========== 유틸리티 ==========
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """사용 가능한 도구 목록"""
        return {
            "lexicon": [
                "lexicon_lookup",
                "lexicon_analyze",
                "prepare_sentiment_context",
                "prepare_routing_context",
                "prepare_causality_context",
                "prepare_playbook_context",
            ],
            "legal": [
                "legal_check"
            ]
        }


# ============================================================
# 글로벌 인스턴스 (워크플로우에서 사용)
# ============================================================

# 싱글톤으로 사용할 수 있도록
_mcp_client: Optional[MCPClient] = None
_mcp_lock = threading.Lock()
_mcp_csv_path: Optional[str] = None 

def get_mcp_client(lexicon_csv_path: str = "custom_lexicon.csv") -> MCPClient:
    global _mcp_client, _mcp_csv_path
    with _mcp_lock:
        if _mcp_client is None:
            _mcp_client = MCPClient(lexicon_csv_path)
            _mcp_csv_path = lexicon_csv_path
        else:
            if _mcp_csv_path != lexicon_csv_path:
                logger.warning(
                    f"get_mcp_client called with different csv_path "
                    f"({lexicon_csv_path}) but singleton already initialized with "
                    f"({_mcp_csv_path}). Using existing instance."
                    "새로운 csv 적용을 위해서는 프로세스를 재시작해야 합니다."
                )
        return _mcp_client

def reset_mcp_client():
    global _mcp_client, _mcp_csv_path
    with _mcp_lock:
        _mcp_client = None
        _mcp_csv_path = None
        logger.info("MCPClient singleton reset")