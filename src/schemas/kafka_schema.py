from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Union
from datetime import datetime
import uuid
from typing import Optional

# 커뮤니치 및 SNS 포스트 데이터 
class ContentData(BaseModel):
    text: str
    author_id: str
    metrics: Dict[str, int] = Field(
        default_factory=lambda: {"likes": 0, "retweets": 0, "replies": 0},
    )

# 급상승 검색어
class TrendQuery(BaseModel):
    query: str
    value: str  

# 지역별 트렌드 통계 
class RegionStat(BaseModel):
    geo: str
    value: int
    
# 트렌드 수치 
class TrendData(BaseModel):
    interest_score: int = Field(ge=0, le=100)
    is_partial: bool
    rising_queries: List[TrendQuery]
    top_queries: List[str]
    region_stats: List[RegionStat]

    
# 통합 Kafka 메시지 메인 모델 
class KafkaMessage(BaseModel):
    # 호환성을 위해 alias를 사용하더라도 원래 필드명을 기반으로 값을 설정할 수 있게 함...
    model_config = ConfigDict(populate_by_name=True)
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  
    source: str 
    collected_at: datetime = Field(default_factory=datetime.now)
    keyword: str  

    content_data: Optional[ContentData] = None 
    trend_data: Optional[TrendData] = None
    
    # 유효성 검사 메서드 
    def validate_payload(self):
        if self.type == "post" and self.content_data is None:
            raise ValueError("type이 'post'일 경우 content_data가 반드시 필요합니다.")
        if self.type == "trend" and self.trend_data is None:
            raise ValueError("type이 'trend'일 경우 trend_data가 반드시 필요합니다.")
