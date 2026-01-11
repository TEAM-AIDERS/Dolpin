import asyncio
import os
import uuid
import datetime
import json
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

class TwitterCollector:
    def __init__(self):
        # MCP 서버 경로 설정 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        server_path = os.path.normpath(os.path.join(current_dir, "../../../src/server/x-v2-server/dist/index.js"))
        
       
        self.server_params = StdioServerParameters(
            command="node",
            args=[server_path],
            env={
                "TWITTER_API_KEY": os.getenv("TWITTER_API_KEY"),
                "TWITTER_API_KEY_SECRET": os.getenv("TWITTER_API_KEY_SECRET"),
                "TWITTER_ACCESS_TOKEN": os.getenv("TWITTER_ACCESS_TOKEN"),
                "TWITTER_ACCESS_TOKEN_SECRET": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
                "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN"),
            }
        )

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
                            "max_results": max_results
                        }
                    )
                    
                    # 응답 데이터 파싱
                    raw_data = response.content[0].text if response.content else "[]"
                    tweets = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

                    for tweet in tweets:
                        # 원본 데이터를 표준 스키마 필드에 연결
                        standardized_data = {
                            "message_id": str(uuid.uuid4()),
                            "type": "post",
                            "source": "twitter",
                            "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
                            "keyword": keyword,
                            "content_data": {
                                "text": tweet.get('text', ''),
                                "author_id": str(tweet.get('author_id', 'unknown')), 
                                "metrics": {
                                    "likes": tweet.get('public_metrics', {}).get('like_count', 0),
                                    "retweets": tweet.get('public_metrics', {}).get('retweet_count', 0)
                                }
                            }
                        }
                        results.append(standardized_data)
            
            return results

        except Exception as e:
            print(f"❌ Twitter 수집기 매핑 중 에러: {e}")
            return []
