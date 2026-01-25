"""
MCP Client - 모든 MCP 서버 호출을 관리하는 단일 진입점

역할:
- Custom Lexicon MCP 호출
- Legal MCP 호출 (향후)
- 워크플로우에서 통일된 인터페이스 제공

구조:
graph.py → nodes.py → mcp_client.py → 각 MCP 서버

버전: v1.0 (260125)
"""

import logging
from typing import Dict, Any, Optional, List

from src.server.lexicon_server import LexiconServer

logger = logging.getLogger(__name__)


# ============================================================
# MCP Client - 단일 진입점
# ============================================================

class MCPClient:
    """모든 MCP 서버 호출을 관리하는 클라이언트"""
    
    def __init__(self, lexicon_csv_path: str = "custom_lexicon.csv"):
        """
        MCP 서버들 초기화
        
        Args:
            lexicon_csv_path: Lexicon CSV 경로
        """
        # Custom Lexicon MCP 초기화
        self.lexicon_server = LexiconServer(lexicon_csv_path)
        logger.info(f"MCPClient initialized with {lexicon_csv_path}")
        
        # Legal MCP 초기화 (향후 구현)
        self.legal_server = None  # self.LegalServer() 추가 예정
    
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
        
        상태 저장 예:
        state.lexicon_matches = mcp.lexicon_analyze(text)
        state.matched_terms = [m.term for m in state.lexicon_matches['matches']]
        state.sentiment_signals = state.lexicon_matches['sentiment_signals']
        state.risk_flags = state.lexicon_matches['risk_flags']
        
        Returns:
        {
            "text": "...",
            "matches": [
                {
                    "term": "불매",
                    "normalized": "불매",
                    "type": "boycott_action",
                    "context": "...주변 텍스트..."
                },
                ...
            ],
            "aggregated_signals": {
                "total_matches": 2,
                "matched_terms": ["불매", "탈빠"],
                "sentiment_mix": {...},
                "action_triggers": [...],
                "risk_flags": ["alert", "alert"],
                "target_entities": ["agency", "artist"]
            },
            "sentiment_signals": [
                {"term": "불매", "polarity": "negative", "intensity": "high", ...},
                ...
            ],
            "routing_signals": [...],
            "action_signals": [...]
        }
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
        """RouterAgent용 컨텍스트 준비"""
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


def get_mcp_client(lexicon_csv_path: str = "custom_lexicon.csv") -> MCPClient:
    """
    MCPClient 싱글톤 인스턴스 반환
    
    워크플로우 초기화 시:
    mcp = get_mcp_client()
    """
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient(lexicon_csv_path)
    return _mcp_client


