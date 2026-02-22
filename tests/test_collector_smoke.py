
import pytest
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Allow direct execution: `python tests/test_collector_smoke.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 프로젝트 imports
from src.pipeline.collectors.twitter import TwitterCollector
from src.pipeline.collectors.community import InstizCollector
from src.pipeline.collectors.google_trends import GoogleTrendsCollector

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================
# 테스트 설정
# ===========================

TEST_KEYWORD = "NCT WISH"  # 모든 플랫폼에서 충분한 데이터가 있는 중립적인 키워드

# 환경변수 검증
def check_env_variables():
    """필수 환경변수 확인"""
    required_vars = {
        "twitter": [
            "TWITTER_API_KEY",
            "TWITTER_API_KEY_SECRET",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_TOKEN_SECRET",
            "TWITTER_BEARER_TOKEN",
        ],
        "instiz": ["INSTIZ_ID", "INSTIZ_PW"],
    }
    
    missing_vars = {}
    for service, vars_list in required_vars.items():
        missing = [var for var in vars_list if not os.getenv(var)]
        if missing:
            missing_vars[service] = missing
    
    return missing_vars


# ===========================
# Twitter (MCP) 테스트
# ===========================

class TestTwitterCollector:
    """Twitter MCP 서버 의존성 테스트"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """테스트 전 환경 확인"""
        missing = check_env_variables().get("twitter", [])
        if missing:
            pytest.skip(f"Twitter 환경변수 누락: {missing}")
        yield
    
    def test_twitter_collector_initialization(self):
        """Twitter Collector 초기화 성공 여부"""
        try:
            collector = TwitterCollector()
            assert collector.server_params is not None
            logger.info("✓ Twitter Collector 초기화 성공")
        except Exception as e:
            pytest.fail(f"Twitter Collector 초기화 실패: {e}")
    
    def test_twitter_mcp_server_connection(self):
        """MCP 서버 연결 여부 (실제 API 호출)"""
        collector = TwitterCollector()
        try:
            results = collector.fetch(TEST_KEYWORD)
            
            # 결과 검증
            assert isinstance(results, list), "결과는 리스트여야 함"
            
            if len(results) == 0:
                logger.warning(f"⚠️ Twitter에서 '{TEST_KEYWORD}' 검색 결과 없음 (API 제한 가능)")
                pytest.skip("Twitter API 제한 또는 검색 결과 부재")
            
            # 첫 번째 결과 구조 검증
            tweet = results[0]
            assert "text" in tweet, "tweet에 'text' 필드 필요"
            assert "author_id" in tweet, "tweet에 'author_id' 필드 필요"
            assert "metrics" in tweet, "tweet에 'metrics' 필드 필요"
            assert isinstance(tweet["metrics"], dict), "metrics는 딕셔너리여야 함"
            
            logger.info(f"✓ Twitter MCP 연결 성공 - {len(results)}개 트윗 수집")
            logger.info(f"  └─ 샘플 트윗: {tweet['text'][:50]}...")
            
        except Exception as e:
            pytest.fail(f"Twitter MCP 서버 연결 실패: {e}\n"
                       f"MCP 서버(X-v2-server)가 실행 중인지 확인하세요.")
    
    def test_twitter_metrics_format(self):
        """Twitter 메트릭 포맷 검증"""
        collector = TwitterCollector()
        results = collector.fetch(TEST_KEYWORD)
        
        if not results:
            pytest.skip("수집된 트윗이 없음")
        
        for tweet in results[:3]:  # 최대 3개 항목만 검증
            metrics = tweet.get("metrics", {})
            assert "likes" in metrics, "metrics에 'likes' 필드 필요"
            assert "retweets" in metrics, "metrics에 'retweets' 필드 필요"
            assert "replies" in metrics, "metrics에 'replies' 필드 필요"
            
            # 숫자 타입 검증
            assert isinstance(metrics["likes"], int), "likes는 정수여야 함"
            assert isinstance(metrics["retweets"], int), "retweets는 정수여야 함"
            assert isinstance(metrics["replies"], int), "replies는 정수여야 함"
        
        logger.info("✓ Twitter 메트릭 포맷 검증 통과")


# ===========================
# Instiz (커뮤니티) 테스트
# ===========================

class TestInstizCollector:
    """Instiz 플레이라이트 의존성 테스트"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """테스트 전 환경 확인"""
        missing = check_env_variables().get("instiz", [])
        if missing:
            pytest.skip(f"Instiz 환경변수 누락: {missing}")
        yield
    
    def test_instiz_collector_initialization(self):
        """Instiz Collector 초기화 성공 여부"""
        try:
            collector = InstizCollector()
            assert collector.base_url == "https://www.instiz.net"
            assert collector.source == "instiz"
            assert collector.user_id is not None
            assert collector.user_pw is not None
            logger.info("✓ Instiz Collector 초기화 성공")
        except Exception as e:
            pytest.fail(f"Instiz Collector 초기화 실패: {e}")
    
    def test_instiz_login_and_collect(self):
        """Playwright를 통한 Instiz 로그인 및 데이터 수집"""
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
        """Instiz 메트릭 구조 검증"""
        collector = InstizCollector()
        results = collector.fetch(TEST_KEYWORD)
        
        if not results:
            pytest.skip("수집된 포스트가 없음")
        
        for post in results[:3]:
            metrics = post.get("metrics", {})
            # views는 필수 메트릭
            assert "views" in metrics, "metrics에 'views' 필드 필요"
            assert isinstance(metrics["views"], int), "views는 정수여야 함"
        
        logger.info("✓ Instiz 메트릭 구조 검증 통과")


# ===========================
# Google Trends 테스트
# ===========================

class TestGoogleTrendsCollector:
    """Google Trends API 의존성 테스트"""
    
    def test_google_trends_collector_initialization(self):
        """Google Trends Collector 초기화"""
        try:
            collector = GoogleTrendsCollector()
            assert collector.pytrends is not None
            logger.info("✓ Google Trends Collector 초기화 성공")
        except Exception as e:
            pytest.fail(f"Google Trends Collector 초기화 실패: {e}")
    
    def test_google_trends_data_collection(self):
        """Google Trends 데이터 수집 (429 에러 확인)"""
        collector = GoogleTrendsCollector()
        try:
            results = collector.fetch(TEST_KEYWORD)
            
            assert isinstance(results, list), "결과는 리스트여야 함"
            
            if len(results) == 0:
                logger.warning(f"⚠️ Google Trends에서 '{TEST_KEYWORD}' 데이터 없음")
                pytest.skip("Google Trends 데이터 부재")
            
            trend = results[0]
            assert "type" in trend, "trend에 'type' 필드 필요"
            assert trend["type"] == "trend", "type은 'trend'여야 함"
            assert "source" in trend, "trend에 'source' 필드 필요"
            assert trend["source"] == "google_trends", "source는 'google_trends'여야 함"
            
            logger.info(f"✓ Google Trends 데이터 수집 성공")
            logger.info(f"  └─ 키워드: {trend.get('keyword')}")
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg:
                pytest.fail(f"❌ Google Trends 429 에러 (요청 제한)\n{e}")
            else:
                pytest.fail(f"Google Trends 수집 실패: {e}")
    
    def test_google_trends_structure(self):
        """Google Trends 응답 구조 검증"""
        collector = GoogleTrendsCollector()
        results = collector.fetch(TEST_KEYWORD)
        
        if not results:
            pytest.skip("수집된 데이터가 없음")
        
        trend = results[0]
        trend_data = trend.get("trend_data", {})
        
        # 필수 필드 검증
        assert "interest_score" in trend_data, "trend_data에 'interest_score' 필요"
        assert "is_partial" in trend_data, "trend_data에 'is_partial' 필요"
        assert "rising_queries" in trend_data, "trend_data에 'rising_queries' 필요"
        assert "top_queries" in trend_data, "trend_data에 'top_queries' 필요"
        
        # 타입 검증
        assert isinstance(trend_data["interest_score"], int), "interest_score는 정수"
        assert isinstance(trend_data["is_partial"], bool), "is_partial은 불린"
        assert isinstance(trend_data["rising_queries"], list), "rising_queries는 리스트"
        assert isinstance(trend_data["top_queries"], list), "top_queries는 리스트"
        
        logger.info("✓ Google Trends 구조 검증 통과")
        logger.info(f"  └─ 관심도: {trend_data['interest_score']}")
        if trend_data.get("rising_queries"):
            logger.info(f"  └─ 급상승 검색어: {trend_data['rising_queries'][0]['query']}")


# ===========================
# 통합 테스트
# ===========================

class TestUnifiedCollectorEnvironment:
    """전체 수집 환경 체크"""
    
    def test_all_collectors_importable(self):
        """모든 Collector가 import 가능한지 확인"""
        try:
            from src.pipeline.collectors.twitter import TwitterCollector
            from src.pipeline.collectors.community import InstizCollector
            from src.pipeline.collectors.google_trends import GoogleTrendsCollector
            from src.pipeline.collector import UnifiedCollector
            
            logger.info("✓ 모든 Collector import 성공")
        except ImportError as e:
            pytest.fail(f"Collector import 실패: {e}")
    
    def test_environment_variables_configured(self):
        """필수 환경변수 설정 여부"""
        missing = check_env_variables()
        
        if missing:
            msg = "누락된 환경변수:\n"
            for service, vars_list in missing.items():
                msg += f"  {service}: {', '.join(vars_list)}\n"
            logger.warning(f"⚠️ {msg}")
            pytest.skip(f"환경변수 설정 필요: {missing}")
        
        logger.info("✓ 모든 필수 환경변수 설정됨")
    
    def test_network_connectivity(self):
        """기본 네트워크 연결 테스트"""
        import socket
        
        hosts = [
            ("google.com", 443, "Google"),
            ("twitter.com", 443, "Twitter"),
            ("instiz.net", 443, "Instiz"),
        ]
        
        failed = []
        for host, port, name in hosts:
            try:
                socket.create_connection((host, port), timeout=5)
                logger.info(f"✓ {name} 연결 성공")
            except Exception as e:
                failed.append((name, str(e)))
        
        if failed:
            msg = "네트워크 연결 실패:\n"
            for name, error in failed:
                msg += f"  {name}: {error}\n"
            pytest.fail(msg)


# ===========================
# 리포트 생성
# ===========================

class SmokeTestReporter:
    """스모크 테스트 결과 리포트"""
    
    @staticmethod
    def print_header():
        print("\n" + "="*70)
        print("🔥 SMOKE TEST: 수집 환경 점검")
        print("="*70)
        print(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"테스트 키워드: {TEST_KEYWORD}\n")
    
    @staticmethod
    def print_summary(passed, failed, skipped):
        print("\n" + "="*70)
        print("📊 테스트 결과 요약")
        print("="*70)
        print(f"✓ 통과: {passed}")
        print(f"✗ 실패: {failed}")
        print(f"⊘ 스킵: {skipped}")
        print("="*70 + "\n")


# ===========================
# pytest 훅
# ===========================

def pytest_configure(config):
    """pytest 시작 전 실행"""
    SmokeTestReporter.print_header()


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