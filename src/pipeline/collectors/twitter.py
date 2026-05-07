import asyncio
import os
import uuid
import datetime
import json
import logging
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

logger = logging.getLogger(__name__)

class TwitterCollector:
    def __init__(self):
        # MCP 서버 경로 설정 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        server_path = os.path.normpath(os.path.join(current_dir, "../../../src/server/x-v2-server/dist/index.js"))
        # os.environ을 기반으로 시스템 환경변수(PATH 등)를 상속하고 Twitter 키를 덮어씀
        # - os.environ 없이 키만 넘기면 자식 프로세스(node)가 PATH를 못 찾아 실행 실패
        # - os.getenv()가 None이면 str(None)="None"이 전달되므로 None 값은 필터링
        twitter_env = {
            k: v for k, v in {
                "TWITTER_API_KEY": os.getenv("TWITTER_API_KEY"),
                "TWITTER_API_KEY_SECRET": os.getenv("TWITTER_API_KEY_SECRET"),
                "TWITTER_ACCESS_TOKEN": os.getenv("TWITTER_ACCESS_TOKEN"),
                "TWITTER_ACCESS_TOKEN_SECRET": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
                "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN"),
            }.items() if v is not None
        }
        self.server_params = StdioServerParameters(
            command="node",
            args=[server_path],
            env={**os.environ, **twitter_env},
        )
    def fetch(self, keyword: str) -> list:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.collect(keyword))
        except Exception as e:
            logger.error(f"Twitter fetch error: {e}")
            return []
        finally:
            loop.close()
        
    async def collect(self, keyword: str, max_results: int = 10):
        results = []
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # 최신 트윗 검색 도구 호출
                    response = await session.call_tool(
                        "search_tweets", 
                        arguments={
                            "query": f"{keyword} -is:retweet",
                            "maxResults": max_results
                        }
                    )
                    
                    # 응답 데이터 파싱
                    raw_data = response.content[0].text if response.content else "[]"
                    tweets = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                    
                    for tweet in tweets:
                        results.append({
                            "text": tweet.get('text', ''),      
                            "author_id": str(tweet.get('author_id', 'unknown')),
                            "metrics": self._format_metrics(tweet.get('public_metrics', {}))
                        })
                    return results
        except Exception as e:
            logger.error(f"Twitter 수집 중 오류 (keyword: {keyword}): {e}")
            return []
        
    # 메트릭 세개만 추출 
    def _format_metrics(self, raw_metrics: dict) -> dict:
        return {
            "likes": raw_metrics.get('like_count', 0),
            "retweets": raw_metrics.get('retweet_count', 0),
            "replies": raw_metrics.get('reply_count', 0)
        }