"""
DOLPIN MCP Server Package
팬덤 특화 표현 분석을 위한 Lexicon Server

Created: 2026-01-25

구조:
  - lexicon_server.py: CSV 기반 어휘 분석 서버
  - mcp_client.py: MCPClient - 모든 MCP 서버 호출의 단일 진입점
  - embedder.py: 임베딩
  - vector_server.py: 벡터 검색 서버
"""

from .lexicon_server import LexiconServer, create_mcp_tools
from .mcp_client import MCPClient

__all__ = [
    "LexiconServer",
    "create_mcp_tools",
    "MCPClient",
]

__version__ = "2.0"
__author__ = "DOLPIN Team"
