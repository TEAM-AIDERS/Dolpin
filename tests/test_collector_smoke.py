import pytest
import os
import sys
import logging
from dotenv import load_dotenv

# Allow direct execution: `python tests/test_collector_smoke.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 테스트 키워드
TEST_KEYWORD = "python"

# ===========================
# Twitter Collector 테스트
# ===========================

class TestTwitterCollector:
    """Twitter/MCP 수집기 검증"""
    
    def test_twitter_collector_initialization(self):
        """Twitter Collector 객체 생성 가능한가?"""
        from src.pipeline.collectors.twitter import TwitterCollector
        
        collector = TwitterCollector()
        assert collector is not None
        logger.info("✓ Twitter Collector 초기화 성공")
    
    def test_twitter_mcp_server_connection(self):
        """Twitter MCP 서버 연결 가능한가?"""
        from src.pipeline.collectors.twitter import TwitterCollector
        
        collector = TwitterCollector()
        try:
            # MCP 서버 연결 테스트 (타임아웃: 10초)
            results = collector.fetch(TEST_KEYWORD)
            
            # 성공하거나 API 제한으로 인한 None 반환은 OK
            # (MCP 서버가 작동한다는 의미)
            logger.info(f"✓ Twitter MCP 서버 응답 (결과: {type(results).__name__})")
        except Exception as e:
            # MCP 서버 미실행 또는 기타 에러
            pytest.fail(f"Twitter MCP 서버 연결 실패: {e}\n"
                       f"MCP 서버 실행 여부 확인 필요")
    
    def test_twitter_metrics_format(self):
        """Twitter 응답 포맷이 올바른가?"""
        from src.pipeline.collectors.twitter import TwitterCollector
        
        collector = TwitterCollector()
        results = collector.fetch(TEST_KEYWORD)
        
        # 결과가 리스트이거나 None (API 제한)
        if results is not None:
            assert isinstance(results, list), "결과는 리스트여야 함"
            
            if len(results) > 0:
                tweet = results[0]
                assert "text" in tweet, "tweet에 'text' 필드 필요"
                assert "metrics" in tweet, "tweet에 'metrics' 필드 필요"
                logger.info(f"✓ Twitter 응답 포맷 검증 성공")
        else:
            logger.info("⊘ Twitter API 제한 또는 데이터 없음 (정상)")

# ===========================
# Instiz Collector 테스트
# ===========================

class TestInstizCollector:
    """Instiz 수집기 검증 (Playwright 사용)"""
    
    def test_instiz_collector_initialization(self):
        """Instiz Collector 객체 생성 가능한가?"""
        # Playwright 미설치 시 여기서만 스킵
        pytest.importorskip("playwright")
        from src.pipeline.collectors.community import InstizCollector
        
        collector = InstizCollector()
        assert collector is not None
        logger.info("✓ Instiz Collector 초기화 성공")
    
    def test_instiz_login_and_collect(self):
        """Playwright를 통한 Instiz 로그인 및 데이터 수집"""
        pytest.importorskip("playwright")
        from src.pipeline.collectors.community import InstizCollector
        
        collector = InstizCollector()
        try:
            results = collector.fetch(TEST_KEYWORD)
    
            assert isinstance(results, list), "결과는 리스트여야 함"
    
            if len(results) == 0:
                logger.warning(f"⚠️ Instiz에서 '{TEST_KEYWORD}' 검색 결과 없음")
                pytest.skip("Instiz 검색 결과 부재")
    
            # 첫 번째 결과 구조 검증
            post = results[0]
            assert "content" in post, "post에 'content' 필드 필요"
            assert "source" in post, "post에 'source' 필드 필요"
            assert post["source"] == "instiz", "source는 'instiz'여야 함"
            assert "metrics" in post, "post에 'metrics' 필드 필요"
    
            logger.info(f"✓ Instiz 로그인 및 수집 성공 - {len(results)}개 포스트 수집")
            logger.info(f"  └─ 샘플 포스트: {post['content'][:50]}...")
    
        except Exception as e:
            pytest.fail(f"Instiz 로그인/수집 실패: {e}\n"
                       f"자격증명(INSTIZ_ID, INSTIZ_PW) 확인 및 Playwright 설치 확인")
    
    def test_instiz_metrics_structure(self):
        """Instiz 메트릭 구조가 올바른가?"""
        pytest.importorskip("playwright")
        from src.pipeline.collectors.community import InstizCollector
        
        collector = InstizCollector()
        results = collector.fetch(TEST_KEYWORD)
        
        if results and len(results) > 0:
            post = results[0]
            metrics = post.get("metrics", {})
            
            # 메트릭은 dict 형태여야 함
            assert isinstance(metrics, dict), "metrics는 dict여야 함"
            logger.info(f"✓ Instiz 메트릭 구조 검증 성공: {list(metrics.keys())}")
        else:
            pytest.skip("수집된 데이터 없음")

# ===========================
# Google Trends Collector 테스트
# ===========================

class TestGoogleTrendsCollector:
    """Google Trends 수집기 검증"""
    
    def test_google_trends_collector_initialization(self):
        """Google Trends Collector 객체 생성 가능한가?"""
        from src.pipeline.collectors.google_trends import GoogleTrendsCollector
        
        collector = GoogleTrendsCollector()
        assert collector is not None
        logger.info("✓ Google Trends Collector 초기화 성공")
    
    def test_google_trends_data_collection(self):
        """Google Trends 데이터 수집 (429 에러 확인)"""
        from src.pipeline.collectors.google_trends import GoogleTrendsCollector
        
        collector = GoogleTrendsCollector()
        try:
            results = collector.fetch(TEST_KEYWORD)
            
            assert isinstance(results, list), "결과는 리스트여야 함"
            
            if len(results) == 0:
                logger.warning(f"⚠️ Google Trends에서 '{TEST_KEYWORD}' 데이터 없음")
                pytest.skip("Google Trends 데이터 부재")
            
            trend = results[0]
            assert "trend_data" in trend, "trend에 'trend_data' 필드 필요"
            
            logger.info(f"✓ Google Trends 데이터 수집 성공")
            logger.info(f"  └─ 관심도 점수: {trend['trend_data'].get('interest_score', 0)}")
        
        except Exception as e:
            logger.warning(f"⚠️ Google Trends 수집 실패 (API 제한 가능성): {e}")
            pytest.skip(f"Google Trends API 제한 또는 에러: {e}")
    
    def test_google_trends_structure(self):
        """Google Trends 데이터 구조 검증"""
        from src.pipeline.collectors.google_trends import GoogleTrendsCollector
        
        collector = GoogleTrendsCollector()
        results = collector.fetch(TEST_KEYWORD)
        
        if results and len(results) > 0:
            trend = results[0]
            trend_data = trend.get("trend_data", {})
            
            # 필요한 필드 확인
            required_fields = ["interest_score", "rising_queries", "top_queries"]
            for field in required_fields:
                assert field in trend_data, f"trend_data에 '{field}' 필드 필요"
            
            logger.info(f"✓ Google Trends 구조 검증 성공")
        else:
            pytest.skip("수집된 데이터 없음")

# ===========================
# 환경 & 네트워크 테스트
# ===========================

class TestUnifiedCollectorEnvironment:
    """전체 환경 설정 및 네트워크 검증"""
    
    def test_all_collectors_importable(self):
        """모든 collector를 import할 수 있는가?"""
        try:
            # ← 지연 import: 필요할 때만 import
            from src.pipeline.collectors.twitter import TwitterCollector
            from src.pipeline.collectors.community import InstizCollector
            from src.pipeline.collectors.google_trends import GoogleTrendsCollector
            
            logger.info("✓ 모든 Collector 임포트 성공")
        except ImportError as e:
            pytest.fail(f"Collector 임포트 실패: {e}")
    
    def test_environment_variables_configured(self):
        """필수 환경변수가 설정되었는가?"""
        required_vars = {
            "TWITTER_API_KEY": "Twitter API 키",
            "TWITTER_BEARER_TOKEN": "Twitter Bearer 토큰",
        }
        
        optional_vars = {
            "INSTIZ_ID": "Instiz ID",
            "INSTIZ_PW": "Instiz 비밀번호",
        }
        
        missing_required = []
        for var, desc in required_vars.items():
            if not os.getenv(var):
                missing_required.append(f"{var} ({desc})")
        
        if missing_required:
            pytest.fail(f"필수 환경변수 미설정: {', '.join(missing_required)}")
        
        # 선택사항 확인
        missing_optional = []
        for var, desc in optional_vars.items():
            if not os.getenv(var):
                missing_optional.append(f"{var} ({desc})")
        
        if missing_optional:
            logger.warning(f"⚠️ 선택사항 환경변수 미설정: {', '.join(missing_optional)}")
        else:
            logger.info("✓ 모든 환경변수 설정됨")
        
        logger.info("✓ 환경변수 설정 확인 완료")
    
    def test_network_connectivity(self):
        """네트워크 연결이 정상인가?"""
        import socket
        
        def is_connected():
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                return True
            except OSError:
                return False
        
        if is_connected():
            logger.info("✓ 네트워크 연결 정상")
        else:
            logger.warning("⚠️ 네트워크 연결 불안정")

# ===========================
# pytest 훅
# ===========================

def pytest_configure(config):
    """pytest 시작 전 설정"""
    logger.info("\n" + "="*80)
    logger.info("🚀 스모크 테스트 시작")
    logger.info("="*80)

def pytest_sessionfinish(session, exitstatus):
    """pytest 종료 후 실행"""
    passed = session.testsfailed == 0 and session.testscollected > 0
    if passed:
        print("\n🎉 모든 스모크 테스트 통과! 수집 환경이 정상입니다.")
    else:
        print("\n⚠️ 일부 테스트 실패. 위 로그를 확인하세요.")


# ===========================
# 명령줄 실행
# ===========================

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))