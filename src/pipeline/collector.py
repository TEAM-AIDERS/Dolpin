import uuid
import re
from datetime import datetime
from typing import List, Dict, Any

from src.pipeline.collectors.community import InstizCollector
from src.pipeline.collectors.twitter import TwitterCollector
from src.pipeline.collectors.google_trends import GoogleTrendsCollector
from src.schemas.kafka_schema import KafkaMessage, ContentData, TrendData
from src.pipeline.kafka_producer import KafkaProducer

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
   
    # 개인정보 보호 위해 아이디 변환 
    def _anonymize_user(self, user_id: str) -> str:
        if not user_id or user_id == "anonymous":
            return "user_unknown"
        return f"{user_id[:3]}_***"
    
    # 중복 방지 메서드 
    def _is_duplicate(self, content: str) -> bool:
        if content in self.seen_contents:
            return True
        self.seen_contents.add(content)
        # 캐시 너무 커지면 리셋.. 
        if len(self.seen_contents) > 1000:
            self.seen_contents.clear()
        return False
    
    # 외부 호출 진입점 
    def run(self, keyword: str):
        try:
            self.collect_and_standardize(keyword)
        except Exception as e:
            print(f"❌ 수집 중 에러 발생: {e}")
        finally:
            # 에러가 나더라도 남아있는 배치는 전송 시도.. 
            self.producer.flush()
            
    # 수집 후 스키마 통합 
    def collect_and_standardize(self, keyword: str) -> List[KafkaMessage]:
        try:
            # 1. 트위터
            raw_tweets = self.twitter_client.fetch(keyword)
            for tweet in raw_tweets:
                text = tweet.get('text')
                author_id = tweet.get('author_id')
                if not text or not author_id:
                    continue
                
                cleaned_text = self._clean_text(tweet['text'])
                safe_user = self._anonymize_user(tweet['author_id'])
                
                if not self._is_duplicate(tweet['text']):
                    msg = KafkaMessage(
                        type="post",
                        source="twitter",
                        keyword=keyword,
                        collected_at=datetime.utcnow(),
                        content_data=ContentData(
                            text=cleaned_text,
                            author_id=safe_user, 
                            metrics=tweet.get('metrics', {})
                        )
                    )
                    self.producer.send(msg)
        
        except Exception as e:
            print(f"⚠️ 트위터 수집 건너뜀 (API 제한 혹은 에러): {e}")
        # 2. 커뮤니티 
        raw_posts = self.community_client.fetch(keyword)
        for post in raw_posts:
            cleaned_text = self._clean_text(post['content'])
            safe_user = self._anonymize_user(post['writer'])
            
            if not self._is_duplicate(post['content']):
                msg = KafkaMessage(
                    type="post",
                    source=post['source'], 
                    keyword=keyword,
                    collected_at=datetime.utcnow(),
                    content_data=ContentData(
                        text=cleaned_text,
                        author_id=safe_user,
                        metrics={"replies": post.get('comment_count', 0)}
                    )
                )
                self.producer.send(msg)
        
        # 3. 구글트렌드
        raw_trends = self.trends_client.fetch(keyword)
        if raw_trends:
            for trend in raw_trends:
                msg = KafkaMessage(
                    type="trend",
                    source="google_trends",
                    keyword=keyword,
                    collected_at=datetime.utcnow(),
                    trend_data=TrendData(
                        interest_score=trend['score'],
                        is_partial=trend['is_partial'],
                        rising_queries=trend['rising_queries'],
                        top_queries=trend['top_queries'],
                        region_stats=trend['region_stats']
                    )
                )
                self.producer.send(msg)

        self.producer.flush()
        print(f"[{keyword}] 전송 완료!")

# 최종 실행 메서드 
if __name__ == "__main__":
    # 테스트 실행
    collector = UnifiedCollector()
    collector.run("엔시티 위시")