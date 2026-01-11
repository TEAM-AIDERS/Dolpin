import datetime
import uuid
from pytrends.request import TrendReq

class GoogleTrendsCollector:
    def __init__(self, hl='ko-KR', tz=540):
        self.pytrends = TrendReq(hl=hl, tz=tz)

    def collect(self, keyword: str, geo: str = 'KR'):
        try:
            # 1. 시간대별 관심도 수집 (최근 1시간)
            self.pytrends.build_payload([keyword], timeframe='now 1-H', geo=geo)
            df_over_time = self.pytrends.interest_over_time()

            if df_over_time.empty:
                return None
            # 가장 마지막 시간의 관심도 점수 추출 
            latest_row = df_over_time.iloc[-1]
            interest_score = int(latest_row[keyword])
            is_partial = bool(latest_row['is_partial'])

            # 2. 관련 급상승 검색어 수집
            related_queries = self.pytrends.related_queries()
            rising_list = []
            
            if keyword in related_queries and related_queries[keyword]['rising'] is not None:
                rising_df = related_queries[keyword]['rising']
                for _, row in rising_df.head(5).iterrows():
                    val = row['value'] # 검색량 증가율 
                    rising_list.append({
                        "query": row['query'],
                        "value": "Breakout" if str(val) == 'None' or val >= 5000 else f"+{val}%"
                    })

            # 3. 인기 검색어 수집
            top_list = []
            if keyword in related_queries and related_queries[keyword]['top'] is not None:
                top_df = related_queries[keyword]['top']
                top_list = top_df.head(5)['query'].tolist()

            # 4. 결과 조립
            result = {
                "message_id": str(uuid.uuid4()),
                "type": "trend",
                "source": "google_trends",
                "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
                "keyword": keyword,
                "trend_data": {
                    "interest_score": interest_score,
                    "is_partial": is_partial,
                    "rising_queries": rising_list,
                    "top_queries": top_list,
                    "region_stats": [
                        {"geo": geo, "value": interest_score}
                    ]
                }
            }
            return result

        except Exception as e:
            print(f"Error collecting Google Trends for {keyword}: {e}")
            return None