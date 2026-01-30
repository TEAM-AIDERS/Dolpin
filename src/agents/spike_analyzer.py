import os
import numpy as np
import logging
import time
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from ..dolpin_langgraph.state import SpikeEvent, SpikeAnalysisResult
from google.cloud import monitoring_v3


load_dotenv()
logger = logging.getLogger(__name__)

class SpikeAnalyzerAgent:
    def __init__(self):
        # 통계 베이스라인 (최근 24시간)
        self.baseline_history = deque(maxlen=288)
        
        # GCP 환경 설정 
        self.project_id = os.getenv("GCP_PROJECT_ID")
        if monitoring_v3 and self.project_id:
            try:
                self.client = monitoring_v3.MetricServiceClient()
                self.project_path = f"projects/{self.project_id}"
                logger.info(f"GCP Monitoring Client Initialized: {self.project_id}")
            except Exception as e:
                logger.error(f"❌ GCP Client 초기화 에러: {e}")
                self.client = None
        else:
            self.client = None
    
    # 통계 지표 계산 
    def _calculate_metrics(self, current_vol: float, messages: List[Dict]) -> Dict[str, float]:
        # Z-Score
        if len(self.baseline_history) < 2:
            z_score = 0.0                   # 과거 데이터 없으면 0 
        else:
            arr = np.array(list(self.baseline_history)) # 최근 24시간 동안 5분 단위 언급량들 
            # (current_vol - 평균) / 표준편차
            z_score = (current_vol - np.mean(arr)) / (np.std(arr) + 1e-6)
            
        # Acceleration (EPS 비교)
        now = datetime.now(timezone.utc)
        ts_list = []
        for m in messages:  # 시간 정보 뽑아서 리스트로 정리 
            try:
                ts = datetime.fromisoformat(m.get('timestamp', '').replace("Z", "+00:00"))
                ts_list.append(ts)
            except:
                continue
        # 최근 1분/5분간 1초당 이벤트수    
        eps_1m = sum(1 for t in ts_list if t >= now - timedelta(minutes=1)) / 60.0
        eps_5m = len(ts_list) / 300.0
        accel = (eps_1m - eps_5m) / (eps_5m + 1e-6) if eps_5m > 0 else 0.0
        
        return {
            "z_score": round(float(z_score), 2),
            "acceleration": round(float(accel), 3),
            "eps_1m": round(float(eps_1m), 4)
        }
    async def _calculate_keyword_risk(self, messages: List[Dict]) -> float:
       if not messages: return 0.0
       # 분석할 텍스트 추출 
       combined_text = " ".join([m.get('text', '') for m in messages])
       
       # Custom Lexicon MCP 호출 (MCP 구현 완료 후 수정 예정)
    
    # 신뢰 가능한 급증인지 
    def _calculate_confidence(self, spike_rate: float, data_completeness: str) -> float:
        base = 0.9
        comp_map = {"confirmed": 1.0, "mixed": 0.8, "partial": 0.6}
        comp_f = comp_map.get(data_completeness, 0.6)
        clarity_f = 1.0 if spike_rate >= 5.0 else (0.9 if spike_rate >= 3.0 else 0.7)
        baseline_f = 1.0 if len(self.baseline_history) >= 100 else 0.7
        
        return round(base * comp_f * clarity_f * baseline_f, 2)
    
    # 분석 결과 메트릭 GCP로 전송 
    def _export_to_gcp(self, metrics: Dict[str, float], keyword: str, action_score: float):
        if not self.client: return
        
        # GCP용 포맷으로 현재 시각 준비 
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        
        all_metrics = {**metrics, "actionability_score": action_score}
        series_list = []
        # 각 메트릭 딕셔너리 형태로 구성 
        for m_name, m_val in metrics.items():
            series = {
                "metric": {
                    "type": f"custom.googleapis.com/dolpin/analysis/{m_name}",
                    "labels": {
                        "keyword": keyword
                    },
                },
                "resource": {
                    "type": "global",
                    "labels": {"project_id": self.project_id},
                },
                "points": [{
                    "interval": {"end_time": {"seconds": seconds, "nanos": nanos}},
                    "value": {"double_value": float(m_val)},
                }],
            }
            series_list.append(series)
            
            try:
                # 메트릭 전송
                self.client.create_time_series(
                    request={
                        "name": self.project_path,
                        "time_series": series_list,
                    }
                )
            except Exception as e:
                logger.error(f"❌ GCP Metric Export Failed: {e}")
    
    # 메인 실행 함수 
    def analyze(self, event: SpikeEvent) -> SpikeAnalysisResult:
        # 통계 지표 산출 
        m = self._calculate_metrics(event['current_volume'], event.get('messages', []))
        
        # actionability_score 산출
        spike_intensity = min((event['spike_rate'] / 5.0) * 0.8 + (max(m['acceleration'], 0) * 0.2), 1.0)
        keyword_weight = self._calculate_keyword_risk(event.get("messages", []))
        action_score = round(spike_intensity * 0.5 + keyword_weight * 0.5, 2)
        
        data_comp = "confirmed" if event.get('messages') else "partial"
        
        # 결과 생성
        result: SpikeAnalysisResult = {
            "is_significant": m['z_score'] >= 3.0 or event['spike_rate'] >= 3.0,
            "spike_rate": event['spike_rate'],
            "spike_type": "coordinated" if m['acceleration'] > 1.0 and m['z_score'] > 5.0 else "organic",
            "spike_nature": "neutral",
            "peak_timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_minutes": 5,
            "confidence": self._calculate_confidence(event['spike_rate'], data_comp),
            "actionability_score": action_score,
            "data_completeness": data_comp,
            "partial_data_warning": None if data_comp == "confirmed" else "⚠️ Payload missing",
            "viral_indicators": {
                "is_trending": m['z_score'] >= 3.0,
                "has_breakout": m['z_score'] >= 5.0,
                "max_rise_rate": f"{event['spike_rate']:.1f}x",
                "breakout_queries": [event['keyword']] if m['z_score'] >= 5.0 else [],
                "cross_platform": list(set(msg.get('source', 'unknown') for msg in event.get('messages', []))),
                "international_reach": 0.0
            }
        }
        
        # 업데이트 및 메트릭 전송
        self._export_to_gcp(m, event['keyword'], action_score)
        self.baseline_history.append(event['current_volume'])

        logger.info(f"Analysis Result [{event['keyword']}]: Score {action_score} | Conf {result['confidence']}")
        return result