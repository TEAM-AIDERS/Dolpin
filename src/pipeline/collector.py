import uuid
import re
import logging
from datetime import datetime
from typing import List

from src.pipeline.collectors.community import InstizCollector
from src.pipeline.collectors.twitter import TwitterCollector
from src.pipeline.collectors.google_trends import GoogleTrendsCollector
from src.schemas.kafka_schema import KafkaMessage, ContentData, TrendData
from src.pipeline.kafka_producer import KafkaProducer

# 로깅 설정
logger = logging.getLogger(__name__)

class UnifiedCollector: 
    def __init__(self):
        # 수집기 인스턴스화
        self.twitter_client = TwitterCollector()
        self.community_client = InstizCollector()
        self.trends_client = GoogleTrendsCollector()
        
        self.producer = KafkaProducer()
        
        # 중복 데이터 방지 캐시
        self.seen_contents = set()
        
    # HTML 태그 제거 및 불필요한 공백 정리
    def _clean_text(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # 개인정보 보호를 위한 사용자 아이디 익명화
    def _anonymize_user(self, user_id: str) -> str:
        if not user_id or user_id == "unknown" or user_id == "anonymous":
            return "user_unknown"
        return f"{user_id[:3]}_***"
    
    # 중복 데이터 체크 
    def _is_duplicate(self, content: str) -> bool:
        if content in self.seen_contents:
            return True
        self.seen_contents.add(content)
        if len(self.seen_contents) > 1000:
            self.seen_contents.clear()
        return False
    
    def run(self, keyword: str):
        try:
            self.collect_and_standardize(keyword)
        except Exception as e:
            logger.error(f"❌ 수집 중 에러 발생: {e}")
        finally:
            # 잔여 메시지 전송
            self.producer.flush()
            
    def collect_and_standardize(self, keyword: str):
        """각 플랫폼별 데이터를 수집하고 KafkaMessage 규격으로 변환하여 전송"""
        
        # 1. 트위터 수집 및 전송
        try:
            raw_tweets = self.twitter_client.fetch(keyword)
            if isinstance(raw_tweets, list):
                for tweet in raw_tweets:
                    text = tweet.get('text')
                    if not text or self._is_duplicate(text):
                        continue
                    
                    try:
                        # Raw 데이터를 바로 사용하지 않고 규격화된 객체 생성
                        msg = KafkaMessage(
                            type="post",
                            source="twitter",
                            keyword=keyword,
                            collected_at=datetime.utcnow(),
                            content_data=ContentData(
                                text=self._clean_text(text),
                                author_id=self._anonymize_user(str(tweet.get('author_id', ''))), 
                                metrics=tweet.get('metrics', {})
                            )
                        )
                        self.producer.send(msg)
                    except Exception as ve:
                        logger.error(f"❌ Twitter Data Validation Error: {ve}")
        except Exception as e:
            logger.warning(f"⚠️ 트위터 수집 실패 (API 제한 가능성): {e}")

        # 2. 커뮤니티(인스티즈) 수집 및 전송
        try:
            raw_posts = self.community_client.fetch(keyword)
            if isinstance(raw_posts, list):
                for post in raw_posts:
                    content = post.get('content')
                    if not content or self._is_duplicate(content):
                        continue
                    
                    try:
                        msg = KafkaMessage(
                            type="post",
                            source=post.get('source', 'instiz'), 
                            keyword=keyword,
                            collected_at=datetime.utcnow(),
                            content_data=ContentData(
                                text=self._clean_text(content),
                                author_id=self._anonymize_user(post.get('writer', 'unknown')),
                                metrics=post.get('metrics', {})
                            )
                        )
                        self.producer.send(msg)
                    except Exception as ve:
                        logger.error(f"❌ Instiz Data Validation Error: {ve}")
        except Exception as e:
            logger.warning(f"⚠️ 커뮤니티 수집 실패: {e}")
        
        # 3. 구글트렌드 수집 및 전송
        try:
            raw_trends = self.trends_client.fetch(keyword)
            if isinstance(raw_trends, list):
                for trend in raw_trends:
                    try:
                        # trend_data 내부 필드 검증 및 맵핑
                        t_data = trend.get('trend_data', {})
                        msg = KafkaMessage(
                            type="trend",
                            source="google_trends",
                            keyword=keyword,
                            collected_at=datetime.utcnow(),
                            trend_data=TrendData(
                                interest_score=t_data.get('interest_score', 0),
                                is_partial=t_data.get('is_partial', False),
                                rising_queries=t_data.get('rising_queries', []),
                                top_queries=t_data.get('top_queries', []),
                                region_stats=t_data.get('region_stats', [])
                            )
                        )
                        self.producer.send(msg)
                    except Exception as ve:
                        logger.error(f"❌ Google Trends Validation Error: {ve}")
        except Exception as e:
            logger.warning(f"⚠️ 구글 트렌드 수집 실패: {e}")

        self.producer.flush()
        print(f"[{keyword}] 모든 수집 및 전송 프로세스 완료")

if __name__ == "__main__":
    collector = UnifiedCollector()
    collector.run("엔시티 위시")