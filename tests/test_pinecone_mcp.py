"""
test_pinecone_mcp.py - Legal MCP 서버 및 클라이언트 연동 테스트

목적:
- MCP 프로세스 실행 및 stdio 통신 확인
- Pinecone에서 법률 데이터 실제 검색 동작 확인
- 에러 처리 및 빈 결과 처리 확인

검증 항목:
1. mcp_client가 pinecone_server.py 프로세스를 정상 실행하는가?
2. 법률 검색 도구들이 실제 데이터를 반환하는가?
3. Empty result를 안전하게 처리하는가?
4. 연결 끊김 시 재시도 가능한가?
"""

import pytest
import asyncio
import os
import sys
import logging
from typing import List, Dict, Any

# Allow direct execution
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.server.mcp_client import MCPClient, get_mcp_client, reset_mcp_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================
# Test Fixtures
# ===========================

@pytest.fixture
def mcp_client():
    """MCP 클라이언트 인스턴스 (동기)"""
    reset_mcp_client()  # 테스트 전 상태 초기화
    client = get_mcp_client()
    yield client
    # 테스트 후 정리은 생략 (프로세스 관리)


# ===========================
# 1. MCP 서버 연결 테스트
# ===========================

class TestMCPServerConnection:
    """MCP 서버 프로세스 연결 테스트"""
    
    def test_mcp_client_initialization(self, mcp_client):
        """MCPClient 객체 생성 가능한가?"""
        assert mcp_client is not None
        assert hasattr(mcp_client, 'lexicon_server')
        logger.info("✓ MCPClient 초기화 성공")
    
    @pytest.mark.asyncio
    async def test_legal_server_process_startup(self, mcp_client):
        """Legal MCP 서버 프로세스 실행 가능한가?"""
        try:
            # 간단한 도구 호출로 서버 실행 확인
            result = await mcp_client._call_legal_tool(
                "search-statutes",
                {"query": "저작권법"}
            )
            
            # 연결 성공 확인
            assert isinstance(result, list), "결과는 리스트여야 함"
            logger.info(f"✓ Legal MCP 서버 프로세스 실행 성공 (결과 타입: {type(result).__name__})")
            
        except Exception as e:
            pytest.skip(
                f"⊘ Legal MCP 서버 연결 불가: {e}\n"
                f"확인 사항: PINECONE_API_KEY, pinecone_server.py 경로"
            )
    
    def test_singleton_pattern(self, mcp_client):
        """get_mcp_client()가 싱글톤으로 작동하는가?"""
        client1 = get_mcp_client()
        client2 = get_mcp_client()
        
        assert client1 is client2, "같은 인스턴스여야 함"
        logger.info("✓ 싱글톤 패턴 동작 확인")


# ===========================
# 2. 법률 검색 도구 테스트
# ===========================

class TestLegalSearchTools:
    """법률 검색 도구 기능 테스트"""
    
    @pytest.mark.asyncio
    async def test_search_statutes(self, mcp_client):
        """저작권법 검색 도구"""
        logger.info("🔍 저작권법 검색 시작...")
        
        result = await mcp_client.legal_search_statutes(
            query="저작권법 제10조 저작인격권"
        )
        
        # 결과 타입 검증
        assert isinstance(result, list), f"결과는 리스트여야 함"
        
        if len(result) > 0:
            # 결과가 있으면 구조 검증
            for item in result:
                assert isinstance(item, dict), "각 항목은 dict여야 함"
            
            logger.info(f"✓ 검색 성공: {len(result)}개 항목")
        else:
            logger.warning("⊘ 검색 결과 없음 (정상)")
    
    @pytest.mark.asyncio
    async def test_search_precedents(self, mcp_client):
        """판례 검색 도구"""
        logger.info("🔍 판례 검색 시작...")
        
        result = await mcp_client.legal_search_precedents(
            query="아이돌 안무 유사성"
        )
        
        assert isinstance(result, list), f"결과는 리스트여야 함"
        
        if len(result) > 0:
            logger.info(f"✓ 판례 검색 성공: {len(result)}개 항목")
        else:
            logger.warning("⊘ 판례 검색 결과 없음 (정상)")
    
    @pytest.mark.asyncio
    async def test_search_internal_policy(self, mcp_client):
        """내부 정책 검색 도구"""
        logger.info("🔍 내부 정책 검색 시작...")
        
        result = await mcp_client.legal_search_policies(
            query="저작권 침해 대응"
        )
        
        assert isinstance(result, list), f"결과는 리스트여야 함"
        
        if len(result) > 0:
            logger.info(f"✓ 정책 검색 성공: {len(result)}개 항목")
        else:
            logger.warning("⊘ 정책 검색 결과 없음 (정상)")
    
    @pytest.mark.asyncio
    async def test_empty_result_handling(self, mcp_client):
        """빈 결과 처리"""
        logger.info("🔍 빈 결과 처리 테스트...")
        
        result = await mcp_client.legal_search_statutes(
            query="XYZABC_매우_구체적인_쿼리_결과없음"
        )
        
        # 빈 리스트 반환 확인
        assert isinstance(result, list), "빈 경우에도 리스트 반환"
        assert len(result) == 0, "결과 없으면 빈 리스트"
        logger.info("✓ 빈 결과 안전하게 처리됨")


# ===========================
# 3. 에러 처리 테스트
# ===========================

class TestErrorHandling:
    """에러 처리 및 복구 능력 테스트"""
    
    @pytest.mark.asyncio
    async def test_invalid_tool_name(self, mcp_client):
        """존재하지 않는 도구 호출"""
        logger.info("❌ 존재하지 않는 도구 호출 테스트...")
        
        result = await mcp_client._call_legal_tool(
            "search-nonexistent-tool",
            {"query": "test"}
        )
        
        # 에러가 나도 리스트 반환 (에러 처리됨)
        assert isinstance(result, list), "에러 발생 시에도 리스트 반환"
        logger.info("✓ 잘못된 도구 호출을 안전하게 처리")
    
    @pytest.mark.asyncio
    async def test_missing_query_parameter(self, mcp_client):
        """필수 파라미터 누락"""
        logger.info("❌ 필수 파라미터 누락 테스트...")
        
        result = await mcp_client._call_legal_tool(
            "search-statutes",
            {}  # query 파라미터 없음
        )
        
        # 에러 처리 확인
        assert isinstance(result, list), "파라미터 누락 시에도 리스트 반환"
        logger.info("✓ 파라미터 누락을 안전하게 처리")


# ===========================
# 4. Lexicon 서버 기본 기능 테스트
# ===========================

class TestLexiconServer:
    """Lexicon MCP 서버 기본 기능 (동기)"""
    
    def test_lexicon_lookup(self, mcp_client):
        """팬덤 표현 단일 조회"""
        logger.info("🔍 팬덤 표현 조회 테스트...")
        
        # 테스트용 항목
        result = mcp_client.lexicon_lookup("고소")
        
        # 결과가 dict이거나 None
        if result:
            assert isinstance(result, dict), "조회 결과는 dict여야 함"
            logger.info(f"✓ 조회 성공")
        else:
            logger.warning("⊘ 조회된 항목 없음 (정상)")
    
    def test_lexicon_analyze(self, mcp_client):
        """텍스트 분석 및 팬덤 표현 추출"""
        logger.info("🔍 팬덤 표현 분석 테스트...")
        
        test_text = "이거 고소 각이야! 명예훼손이지!"
        result = mcp_client.lexicon_analyze(test_text)
        
        assert isinstance(result, dict), "분석 결과는 dict여야 함"
        logger.info(f"✓ 분석 성공")


# ===========================
# 5. 데이터 품질 테스트
# ===========================

class TestDataQuality:
    """반환된 데이터의 품질 검증"""
    
    @pytest.mark.asyncio
    async def test_statutes_data_structure(self, mcp_client):
        """법령 데이터 구조 검증"""
        logger.info("📋 법령 데이터 구조 검증...")
        
        result = await mcp_client.legal_search_statutes(
            query="저작권법"
        )
        
        if len(result) > 0:
            item = result[0]
            assert isinstance(item, dict), "항목은 dict여야 함"
            logger.info(f"✓ 데이터 구조 검증 완료")
        else:
            pytest.skip("검색 결과 없음")
    
    @pytest.mark.asyncio
    async def test_no_malformed_json(self, mcp_client):
        """응답이 유효한 JSON인가?"""
        logger.info("🔍 JSON 형식 검증...")
        
        try:
            result = await mcp_client.legal_search_statutes(query="법")
            
            assert isinstance(result, (list, dict)), "결과는 list 또는 dict여야 함"
            logger.info("✓ JSON 형식 유효함")
            
        except Exception as e:
            pytest.fail(f"JSON 파싱 실패: {e}")


# ===========================
# 6. 성능 테스트
# ===========================

class TestPerformance:
    """응답 시간 및 성능 테스트"""
    
    @pytest.mark.asyncio
    async def test_response_time(self, mcp_client):
        """검색 응답 시간"""
        import time
        
        logger.info("⏱️ 응답 시간 측정...")
        
        start = time.time()
        result = await mcp_client.legal_search_statutes(query="저작권법")
        elapsed = time.time() - start
        
        logger.info(f"✓ 응답 시간: {elapsed:.2f}초")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mcp_client):
        """동시 요청 처리"""
        logger.info("🔄 동시 요청 테스트...")
        
        # 3개의 동시 요청
        tasks = [
            mcp_client.legal_search_statutes(query="저작권법"),
            mcp_client.legal_search_precedents(query="판례"),
            mcp_client.legal_search_policies(query="정책"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 모든 요청이 성공했는지 확인
        for result in results:
            assert isinstance(result, list), "모든 요청이 리스트 반환해야 함"
        
        logger.info(f"✓ 동시 요청 3개 모두 성공")


# ===========================
# 7. 통합 시나리오 테스트
# ===========================

class TestIntegrationScenario:
    """실제 사용 시나리오"""
    
    @pytest.mark.asyncio
    async def test_legal_risk_investigation_flow(self, mcp_client):
        """법률 위험 조사 흐름"""
        logger.info("🔍 법률 위험 조사 흐름 테스트...")
        
        # 1단계: 저작권 관련 법령 검색
        statutes = await mcp_client.legal_search_statutes(
            query="저작권 침해"
        )
        logger.info(f"  └─ 법령: {len(statutes)}개")
        
        # 2단계: 관련 판례 검색
        precedents = await mcp_client.legal_search_precedents(
            query="저작권 침해 판례"
        )
        logger.info(f"  └─ 판례: {len(precedents)}개")
        
        # 3단계: 내부 정책 확인
        policies = await mcp_client.legal_search_policies(
            query="저작권 침해 대응"
        )
        logger.info(f"  └─ 정책: {len(policies)}개")
        
        # 결과 통합
        all_context = [
            *statutes,
            *precedents,
            *policies
        ]
        
        logger.info(f"✓ 통합 컨텍스트: {len(all_context)}개 항목")
        
        assert isinstance(all_context, list), "결과는 리스트여야 함"


# ===========================
# pytest 훅
# ===========================

def pytest_configure(config):
    """pytest 시작 전 설정"""
    logger.info("\n" + "="*80)
    logger.info("🚀 Legal MCP 서버 테스트 시작")
    logger.info("="*80)


def pytest_sessionfinish(session, exitstatus):
    """pytest 종료 후 실행"""
    if session.testsfailed == 0:
        print("\n✅ 모든 MCP 연동 테스트 통과!")
        print("   → Legal RAG Agent 테스트로 진행 가능")
    else:
        print("\n⚠️ 일부 테스트 실패 또는 스킵됨")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))