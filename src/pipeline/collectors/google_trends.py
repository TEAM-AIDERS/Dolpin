import datetime
import uuid
import logging
import pandas as pd
from pytrends.request import TrendReq

logger = logging.getLogger(__name__)

class GoogleTrendsCollector:
    def __init__(self, hl='ko-KR', tz=540):
        self.pytrends = TrendReq(
            hl=hl, 
            tz=tz, 
            timeout=(10, 25), 
            retries=3, 
            backoff_factor=0.5
        )
    
    def fetch(self, keyword: str, geo: str = 'KR') -> list:
        result = self.collect(keyword, geo)
        return [result] if result else []
    
    def collect(self, keyword: str, geo: str = 'KR'):
        try:
            # 1. 시간대별 관심도 수집 
            self.pytrends.build_payload([keyword], timeframe='today 1-m', geo=geo)
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
                    val = row['value']  # 검색량 증가율 
                    formatted_value = self._format_rising_value(val)
                    rising_list.append({
                        "query": row['query'],
                        "value": formatted_value
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
            logger.error(f"Google Trends 수집 중 오류 (keyword: {keyword}): {e}")
            return None
    
    def _format_rising_value(self, val) -> str:
        # NaN 체크
        if pd.isna(val):
            return "Breakout"
        
        # 문자열 타입 체크
        if isinstance(val, str):
            if val.lower() == 'breakout':
                return "Breakout"
            try:
                numeric_val = int(val)
                return f"+{numeric_val}%"
            except ValueError:
                logger.warning(f"예상치 못한 문자열 value: {val}")
                return "Breakout"
        
        # 숫자 타입 체크
        if isinstance(val, (int, float)):
            numeric_val = int(val)
            if numeric_val >= 5000:
                return "Breakout"
            else:
                return f"+{numeric_val}%"
        
        # 예상치 못한 타입
        logger.warning(f"예상치 못한 value 타입: {type(val)}, 값: {val}")
        return "Breakout"
    