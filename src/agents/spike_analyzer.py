import os
import numpy as np
import logging
import time
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Any, Dict, List

from dotenv import load_dotenv

from ..dolpin_langgraph.state import SpikeAnalysisResult, SpikeEvent
from ..server.mcp_client import get_mcp_client

try:
    from google.cloud import monitoring_v3
except Exception:  # pragma: no cover - optional dependency
    monitoring_v3 = None

load_dotenv()
logger = logging.getLogger(__name__)


class SpikeAnalyzerAgent:
    def __init__(
        self,
        significant_spike_threshold: float = 3.0,
        breakout_spike_threshold: float = 5.0,
    ):
        # 통계 베이스라인 (최근 24시간, 5분 단위 288개)
        self.baseline_history = deque(maxlen=288)
        self.significant_spike_threshold = significant_spike_threshold
        self.breakout_spike_threshold = breakout_spike_threshold

        # MCP Client - 실패해도 fallback 0.0으로 처리
        try:
            self.mcp = get_mcp_client()
        except Exception as e:
            logger.warning("MCP client init failed (keyword risk will be 0.0): %s", e)
            self.mcp = None

        # GCP Monitoring Client - 없으면 None으로 처리
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.client = None
        self.project_path = None
        if monitoring_v3 and self.project_id:
            try:
                self.client = monitoring_v3.MetricServiceClient()
                self.project_path = f"projects/{self.project_id}"
                logger.info("GCP Monitoring Client initialized: %s", self.project_id)
            except Exception as e:
                logger.warning("GCP client 초기화 실패 (metrics export disabled): %s", e)

    # ──────────────────────────────────────────────
    # 통계 지표 계산 (Z-Score, Acceleration, EPS)
    # ──────────────────────────────────────────────
    def _calculate_metrics(self, current_vol: float, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        # Z-Score: 과거 데이터가 2개 이상일 때만 의미있음
        if len(self.baseline_history) < 2:
            z_score = 0.0
        else:
            arr = np.array(list(self.baseline_history), dtype=float) # 최근 24시간 동안 5분 단위 언급량들
            # (current_vol - 평균) / 표준편차
            z_score = (current_vol - float(np.mean(arr))) / (float(np.std(arr)) + 1e-6)

        # Acceleration: 최근 1분 EPS vs 5분 EPS 비교
        now = datetime.now(timezone.utc)
        ts_list = []
        for msg in messages:
            try:
                ts_str = str(msg.get("timestamp", "")).replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts_list.append(ts)
            except Exception:
                continue

        eps_1m = sum(1 for ts in ts_list if ts >= now - timedelta(minutes=1)) / 60.0
        eps_5m = len(ts_list) / 300.0
        accel = (eps_1m - eps_5m) / (eps_5m + 1e-6) if eps_5m > 0 else 0.0

        return {
            "z_score": round(float(z_score), 2),
            "acceleration": round(float(accel), 3),
            "eps_1m": round(float(eps_1m), 4),
        }

    # ──────────────────────────────────────────────
    # MCP 렉시콘 기반 키워드 리스크 산출
    # ──────────────────────────────────────────────
    def _calculate_keyword_risk(self, messages: List[Dict[str, Any]]) -> float:
        if not messages or not self.mcp:
            return 0.0

        combined_text = " ".join(str(m.get("text", "")) for m in messages).strip()
        if not combined_text:
            return 0.0

        try:
            analysis_result = self.mcp.lexicon_analyze(combined_text) or {}
            # matched_terms (구버전) / matches (신버전) 둘 다 지원
            matched_terms = analysis_result.get("matched_terms") or analysis_result.get("matches") or []
            risk_score = 0.0

            for entry in matched_terms:
                strength = entry.get("action_strength", "none")
                if strength == "collective":
                    risk_score += 0.5
                elif strength == "declaration":
                    risk_score += 0.3

                flag = entry.get("risk_flag", "none")
                if flag == "alert":
                    risk_score += 0.5
                elif flag == "watch":
                    risk_score += 0.2

            return round(min(1.0, max(0.0, risk_score)), 2)
        except Exception as e:
            logger.warning("MCP lexicon analyze failed (fallback 0.0): %s", e)
            return 0.0

    # ──────────────────────────────────────────────
    # 신뢰도 계산
    # ──────────────────────────────────────────────
    def _calculate_confidence(self, spike_rate: float, data_completeness: str) -> float:
        base = 0.9
        comp_map = {"confirmed": 1.0, "mixed": 0.8, "partial": 0.6}
        comp_f = comp_map.get(data_completeness, 0.6)
        clarity_f = 1.0 if spike_rate >= 5.0 else (0.9 if spike_rate >= 3.0 else 0.7)
        # baseline_history가 충분히 쌓였을 때 신뢰도 상승
        baseline_f = 1.0 if len(self.baseline_history) >= 100 else 0.7
        return round(base * comp_f * clarity_f * baseline_f, 2)

    # ──────────────────────────────────────────────
    # 메시지 텍스트 기반 spike 성격 추정
    # ──────────────────────────────────────────────
    def _estimate_spike_nature(self, messages: List[Dict[str, Any]]) -> str:
        positive_cues = ("좋", "최고", "감사", "응원", "축하", "사랑", "행복", "대박")
        negative_cues = ("불매", "보이콧", "실망", "화나", "논란", "문제", "싫", "최악", "탈덕")
        positive_hits = 0
        negative_hits = 0

        for msg in messages:
            text = str(msg.get("text", "")).lower()
            if any(c in text for c in positive_cues):
                positive_hits += 1
            if any(c in text for c in negative_cues):
                negative_hits += 1

        if positive_hits > 0 and negative_hits > 0:
            return "mixed"
        if negative_hits > positive_hits:
            return "negative"
        if positive_hits > 0:
            return "positive"
        return "neutral"

    # ──────────────────────────────────────────────
    # GCP Monitoring으로 메트릭 전송 (없으면 skip)
    # ──────────────────────────────────────────────
    def _export_to_gcp(self, metrics: Dict[str, float], keyword: str, action_score: float):
        if not self.client or not self.project_path:
            return

        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        all_metrics = {**metrics, "actionability_score": action_score}
        series_list = []

        for metric_name, metric_value in all_metrics.items():
            series_list.append(
                {
                    "metric": {
                        "type": f"custom.googleapis.com/dolpin/analysis/{metric_name}",
                        "labels": {"keyword": keyword},
                    },
                    "resource": {
                        "type": "global",
                        "labels": {"project_id": self.project_id},
                    },
                    "points": [
                        {
                            "interval": {"end_time": {"seconds": seconds, "nanos": nanos}},
                            "value": {"double_value": float(metric_value)},
                        }
                    ],
                }
            )

        try:
            self.client.create_time_series(
                request={"name": self.project_path, "time_series": series_list}
            )
            logger.debug("GCP metrics exported: keyword=%s", keyword)
        except Exception as e:
            logger.warning("GCP metric export failed (non-critical): %s", e)

    # ──────────────────────────────────────────────
    # 메인 분석 함수
    # ──────────────────────────────────────────────
    def analyze(self, event: SpikeEvent) -> SpikeAnalysisResult:
        # ── 기본값 정제 ──────────────────────────
        baseline = int(event.get("baseline", 0) or 0)
        current_volume = int(event.get("current_volume", 0) or 0)
        raw_spike_rate = float(event.get("spike_rate", 0.0) or 0.0)
        if raw_spike_rate <= 0 and baseline > 0 and current_volume > 0:
            raw_spike_rate = current_volume / baseline
        spike_rate = round(float(raw_spike_rate), 2)

        keyword = str(event.get("keyword", "unknown"))
        messages = list(event.get("messages", []) or [])

        # ── 통계 지표 산출 ────────────────────────
        metrics = self._calculate_metrics(current_volume, messages)

        # ── actionability_score ───────────────────
        spike_intensity = min((spike_rate / 5.0) * 0.8 + (max(metrics["acceleration"], 0) * 0.2), 1.0)
        keyword_weight = self._calculate_keyword_risk(messages)
        action_score = round(spike_intensity * 0.5 + keyword_weight * 0.5, 2)

        # ── 데이터 완전성 판단 ─────────────────────
        data_completeness = "confirmed" if (messages and baseline > 0 and current_volume > 0) else "partial"

        # ── is_significant / has_breakout ─────────
        is_significant = bool(
            metrics["z_score"] >= 3.0
            or spike_rate >= self.significant_spike_threshold
        )
        has_breakout = bool(
            metrics["z_score"] >= 5.0
            or spike_rate >= self.breakout_spike_threshold
        )

        # ── peak_timestamp / duration ─────────────
        timestamps = [str(m.get("timestamp")) for m in messages if m.get("timestamp")]
        detected_at = event.get("detected_at")
        if detected_at:
            timestamps.append(str(detected_at))
        peak_timestamp = max(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()

        duration_minutes = 0
        if len(timestamps) >= 2:
            parsed = []
            for ts in timestamps:
                try:
                    parsed.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except Exception:
                    continue
            if len(parsed) >= 2:
                duration_minutes = int((max(parsed) - min(parsed)).total_seconds() / 60)

        # ── viral_indicators ──────────────────────
        sources = sorted({str(m.get("source", "unknown")) for m in messages if m.get("source")})
        non_ko_count = sum(
            1 for m in messages
            if str(m.get("detected_language", "")).lower().strip() not in ("", "ko", "kr")
        )
        international_reach = round(non_ko_count / max(1, len(messages)), 2)

        # ── 최종 결과 구성 ────────────────────────
        result: SpikeAnalysisResult = {
            "is_significant": is_significant,
            "spike_rate": spike_rate,
            "spike_type": "coordinated" if metrics["acceleration"] > 1.0 and has_breakout else "organic",
            "spike_nature": self._estimate_spike_nature(messages),
            "peak_timestamp": peak_timestamp,
            "duration_minutes": duration_minutes,
            "confidence": self._calculate_confidence(spike_rate, data_completeness),
            "actionability_score": action_score,
            "data_completeness": data_completeness,
            "partial_data_warning": None if data_completeness == "confirmed" else "incomplete spike_event fields",
            "viral_indicators": {
                "is_trending": is_significant,
                "has_breakout": has_breakout,
                "max_rise_rate": "Breakout" if has_breakout else f"+{max(0.0, (spike_rate - 1.0) * 100):.0f}%",
                "breakout_queries": [keyword] if has_breakout and keyword != "unknown" else [],
                "cross_platform": sources,
                "international_reach": international_reach,
            },
        }

        # ── 업데이트 및 메트릭 전송─────────────────────────
        self._export_to_gcp(metrics, keyword, action_score)
        self.baseline_history.append(current_volume)

        logger.info(
            "Spike analysis [%s]: rate=%.2f, is_significant=%s, score=%.2f, conf=%.2f",
            keyword, spike_rate, result["is_significant"], action_score, result["confidence"]
        )
        return result