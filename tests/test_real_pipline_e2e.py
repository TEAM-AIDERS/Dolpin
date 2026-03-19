"""
실제 데이터 파이프라인 E2E 테스트

Mock이 아닌 실제 데이터 수집부터 분석 결과까지 검증합니다:
  1. Kafka에서 실제 메시지 수집
  2. SpikeAnalyzer 실행
  3. Router 분기
  4. 각 경로별 분석 실행
  5. 최종 executive_brief 생성

선택사항:
  - REAL_PIPELINE_TEST_KEYWORD (기본: "테스트키워드")
  - REAL_PIPELINE_TIMEOUT (기본: 180초)
  - REAL_PIPELINE_MIN_MESSAGES (기본: 3)
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================================
# Kafka 메시지 수집
# ============================================================================

class KafkaMessageCollector:
    """Kafka에서 실제 메시지 수집"""
    
    def __init__(self, timeout: int = 180, min_messages: int = 3):
        self.timeout = timeout
        self.min_messages = min_messages
        self.collected_messages: List[Dict[str, Any]] = []
        self.kafka_ready = self._check_kafka_config()
    
    def _check_kafka_config(self) -> bool:
        """Kafka 환경 변수 확인"""
        required = [
            "KAFKA_BOOTSTRAP_SERVERS",
            "KAFKA_API_KEY",
            "KAFKA_API_SECRET",
            "KAFKA_TOPIC",
        ]
        missing = [k for k in required if not os.getenv(k)]
        
        if missing:
            logger.warning(f"Missing Kafka config: {missing}")
            return False
        
        logger.info("✓ Kafka config available")
        return True
    
    def collect_messages(self, keyword: str, timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Kafka에서 특정 키워드의 메시지 수집
        
        Args:
            keyword: 검색 키워드
            timeout: 타임아웃 (초)
        
        Returns:
            수집된 메시지 리스트
        
        Raises:
            RuntimeError: Kafka 설정 없음
            TimeoutError: 타임아웃 초과
        """
        if not self.kafka_ready:
            raise RuntimeError("Kafka config not available")
        
        timeout = timeout or self.timeout
        
        try:
            from confluent_kafka import Consumer
        except ImportError:
            raise RuntimeError("confluent_kafka not installed")

        topic = os.getenv("KAFKA_TOPIC")
        mode = os.getenv("MODE")
      
        logger.info(f"Collecting messages for keyword: {keyword}")
        logger.info(f"Topic: {topic}, Mode: {mode}")
        logger.info(f"Timeout: {timeout}s, Min messages: {self.min_messages}")
        
        consumer_conf = {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": os.getenv("KAFKA_API_KEY"),
            "sasl.password": os.getenv("KAFKA_API_SECRET"),
            "group.id": f"dolpin-real-e2e-{int(time.time())}",
            "auto.offset.reset": "earliest" if os.getenv("MODE") == "REPLAY" else "latest",
            "enable.auto.commit": False,
        }
        
        consumer = Consumer(consumer_conf)
        
        try:
            logger.info(f"Subscribing to topic: {topic}")
            consumer.subscribe([topic])
            
            # 파티션 할당 대기
            logger.info("Waiting for partition assignment...")
            assign_deadline = time.time() + 30
            while time.time() < assign_deadline:
                consumer.poll(0.2)
                if consumer.assignment():
                    logger.info("✓ Assigned partitions: %d partition(s)", len(consumer.assignment()))
                    break
            
            if not consumer.assignment():
                raise RuntimeError("Partition assignment timeout")
            
            # 메시지 수집
            messages = []
            deadline = time.time() + timeout
            
            logger.info(f"Collecting messages until {self.min_messages} found or timeout...")
            
            while time.time() < deadline:
                msg = consumer.poll(1.0)
                
                if msg is None or msg.error():
                    continue

                raw_value = msg.value()
                logger.info(f"Raw Kafka payload bytes: {raw_value}")
                
                try:
                    data = json.loads(msg.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning(f"Failed to decode message: {msg.value()}")
                    continue
                
                # 키워드 필터링
                if data.get("keyword") != keyword:
                   logger.debug(f"Skipping non-matching keyword: {data.get('keyword')}")
                   continue                 
                  
                messages.append(data)
                logger.info(f"✓ Message {len(messages)}: {data.get('type')} from {data.get('source')}")
                
                if len(messages) >= self.min_messages:
                    logger.info(f"✓ Collected {len(messages)} messages")
                    break
            
            if len(messages) < self.min_messages:
                raise TimeoutError(
                    f"Only {len(messages)}/{self.min_messages} messages collected after {timeout}s"
                )
            
            self.collected_messages = messages
            return messages
        
        finally:
            consumer.close()


# ============================================================================
# Kafka 메시지 → Spike Event 변환
# ============================================================================

def kafka_messages_to_spike_event(
    messages: List[Dict[str, Any]],
    keyword: str
) -> Dict[str, Any]:
    """
    Kafka 메시지 → SpikeEvent 변환

    스파이크 분석을 위한 입력 데이터 구성
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # 메시지 정규화
    normalized_messages = []
    for kafka_msg in messages:
        if kafka_msg.get("type") == "post" and kafka_msg.get("content_data"):
            content = kafka_msg["content_data"]
            normalized_messages.append({
                "id": content.get("id", f"msg_{len(normalized_messages)}"),
                "source_message_id": content.get("id", f"src_{len(normalized_messages)}"),
                "text": content.get("text", ""),
                "timestamp": kafka_msg.get("collected_at", now),
                "source": kafka_msg.get("source", "unknown"),
                "author_id": content.get("author_id", "unknown"),
                "metrics": content.get("metrics", {"likes": 0, "retweets": 0}),
                "is_anonymized": False,
                "detected_language": "ko",
            })
        elif kafka_msg.get("type") == "trend" and kafka_msg.get("trend_data"):
            trend = kafka_msg["trend_data"]
            for i, query in enumerate(trend.get("rising_queries", [])):
                normalized_messages.append({
                    "id": f"trend_{len(normalized_messages)}",
                    "source_message_id": f"src_trend_{len(normalized_messages)}",
                    "text": query.get("query", ""),
                    "timestamp": kafka_msg.get("collected_at", now),
                    "source": kafka_msg.get("source", "unknown"),
                    "detected_language": "ko",
                })

    # 스파이크 분석 입력 생성
    # TODO: spike_rate는 실제 수집기에서 계산되어야 함 (현재 하드코딩)
    # - SpikeAnalyzerAgent는 spike_rate=0이면 current_volume/baseline으로 자동 계산 가능
    # - 하지만 significant_spike_threshold=3.0이므로 current_volume이 baseline의 3배 이상이어야 분석 진행
    # - 실제 E2E에서는 Google Trends interest_score나 수집된 post 수 기반으로 계산 필요
    return {
        "keyword": keyword,
        "spike_rate": 2.5,  # TODO: 하드코딩 — 실제 수집기에서 계산한 값으로 교체 필요
        "baseline": 100,
        "current_volume": int(100 * 2.5),  # 250
        "detected_at": now,
        "time_window": "1h",
        "messages": normalized_messages,
        "raw_kafka_message_ids": [m.get("id", f"msg_{i}") for i, m in enumerate(messages)],
    }


# ============================================================================
# 워크플로우 실행
# ============================================================================

def run_workflow(spike_event: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """
    실제 워크플로우 실행
    
    Args:
        spike_event: SpikeEvent 객체
        trace_id: 추적 ID
    
    Returns:
        최종 state (executive_brief 포함)
    """
    from src.dolpin_langgraph.graph import compile_workflow
    
    logger.info(f"[{trace_id}] Compiling workflow...")
    workflow = compile_workflow()
    
    # 초기 State 생성
    state = {
        "trace_id": trace_id,
        "spike_event": spike_event,
        "workflow_start_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "route1_decision": None,
        "route2_decision": None,
        "route3_decision": None,
        "positive_viral_detected": None,
        "spike_analysis": None,
        "sentiment_result": None,
        "causality_result": None,
        "legal_risk": None,
        "amplification_summary": None,
        "playbook": None,
        "node_insights": {},
        "executive_brief": None,
        "error_logs": [],
        "skipped": False,
        "skip_reason": None,
    }
    
    logger.info(f"[{trace_id}] Starting workflow with {len(spike_event['messages'])} messages...")
    
    start_time = time.time()
    result = workflow.invoke(state)
    elapsed = time.time() - start_time
    
    logger.info(f"[{trace_id}] Workflow completed in {elapsed:.2f}s")
    
    return result


# ============================================================================
# 데이터 충분성 검증
# ============================================================================

def validate_data_sufficiency(messages: List[Dict[str, Any]], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    데이터가 분석에 충분한지 검증
    
    Returns:
        검증 결과 (dict)
    """
    validation = {
        "total_messages": len(messages),
        "has_text_content": False,
        "sentiment_result_quality": "unknown",
        "network_analysis_quality": "unknown",
        "warnings": [],
        "recommendations": [],
    }
    
    # 1. 메시지 개수
    if len(messages) < 3:
        validation["warnings"].append(f"Very few messages ({len(messages)}), analysis may be unreliable")
        validation["recommendations"].append("Collect at least 10-20 messages for reliable analysis")
    
    # 2. 메시지 내용
    text_messages = [m for m in messages if m.get("text")]
    validation["has_text_content"] = len(text_messages) > 0
    
    if not validation["has_text_content"]:
        validation["warnings"].append("No text content in messages")
        validation["recommendations"].append("Ensure messages have text field")
    
    avg_text_length = sum(len(m.get("text", "")) for m in text_messages) / max(1, len(text_messages))
    if avg_text_length < 5:
        validation["warnings"].append(f"Very short messages (avg {avg_text_length:.1f} chars)")
        validation["recommendations"].append("Ensure messages have sufficient text content")
    
    # 3. Sentiment 결과
    sentiment_result = result.get("sentiment_result")
    if sentiment_result:
        validation["sentiment_result_quality"] = "good"
        analyzed_count = sentiment_result.get("analyzed_count", 0)
        if analyzed_count < len(text_messages) * 0.8:
            validation["sentiment_result_quality"] = "partial"
            validation["warnings"].append(f"Only {analyzed_count}/{len(text_messages)} messages analyzed")
    else:
        validation["sentiment_result_quality"] = "missing"
        if not result.get("skipped"):
            validation["warnings"].append("Sentiment result missing (expected for non-skip paths)")
    
    # 4. Causality 결과
    causality_result = result.get("causality_result")
    if result.get("route2_decision") == "full_analysis":
        if causality_result:
            validation["network_analysis_quality"] = "good"
            network_size = causality_result.get("network_size", 0)
            if network_size < 3:
                validation["network_analysis_quality"] = "limited"
                validation["warnings"].append(f"Small network ({network_size} nodes)")
        else:
            validation["network_analysis_quality"] = "missing"
            validation["warnings"].append("Causality result missing (expected for full_analysis path)")
    
    return validation


# ============================================================================
# 테스트 케이스
# ============================================================================

@pytest.mark.real_pipeline
class TestRealPipelineE2E:
    """실제 데이터 파이프라인 검증"""
    
    @pytest.fixture
    def real_pipeline_enabled(self):
        """실제 파이프라인 테스트 활성화 확인"""
        if os.getenv("RUN_REAL_PIPELINE", "").lower() not in {"1", "true", "yes", "on"}:
            pytest.skip("Set RUN_REAL_PIPELINE=1 to run real pipeline tests")
    
    @pytest.fixture
    def collector(self):
        """Kafka 메시지 수집기"""
        collector = KafkaMessageCollector(
            timeout=int(os.getenv("REAL_PIPELINE_TIMEOUT", "180")),
            min_messages=int(os.getenv("REAL_PIPELINE_MIN_MESSAGES", "3")),
        )
        
        if not collector.kafka_ready:
            pytest.skip("Kafka config not available")
        
        return collector
    
    def test_full_pipeline_kafka_to_executive_brief(self, real_pipeline_enabled, collector):
        """
        완전한 파이프라인: Kafka 수집 → 분석 → 보고서 생성
        
        검증:
          1. Kafka에서 메시지 수집 가능
          2. SpikeAnalyzer 실행 가능
          3. 라우팅 결정 발생
          4. 최종 executive_brief 생성
          5. 데이터 충분성 검증
        """
        keyword = os.getenv("REAL_PIPELINE_TEST_KEYWORD", "테스트키워드")
        
        # 1단계: Kafka에서 메시지 수집
        logger.info(f"\n{'='*70}")
        logger.info("Step 1: Collecting messages from Kafka")
        logger.info(f"{'='*70}")
        
        messages = collector.collect_messages(keyword)
        assert len(messages) >= 3, f"Expected at least 3 messages, got {len(messages)}"
        logger.info(f"✓ Collected {len(messages)} messages")
        
        # 2단계: SpikeEvent 생성
        logger.info(f"\n{'='*70}")
        logger.info("Step 2: Building SpikeEvent from Kafka messages")
        logger.info(f"{'='*70}")
        
        spike_event = kafka_messages_to_spike_event(messages, keyword)
        assert len(spike_event["messages"]) > 0, "No normalized messages"
        logger.info(f"✓ Built SpikeEvent with {len(spike_event['messages'])} messages")
        
        # 3단계: 워크플로우 실행
        logger.info(f"\n{'='*70}")
        logger.info("Step 3: Running workflow")
        logger.info(f"{'='*70}")
        
        trace_id = f"real-e2e-{int(time.time())}"
        result = run_workflow(spike_event, trace_id)
        
        # 4단계: 결과 검증
        logger.info(f"\n{'='*70}")
        logger.info("Step 4: Validating results")
        logger.info(f"{'='*70}")
        
        # 라우팅 결정
        route1 = result.get("route1_decision")
        route2 = result.get("route2_decision")
        route3 = result.get("route3_decision")
        
        logger.info(f"✓ route1_decision: {route1}")
        logger.info(f"✓ route2_decision: {route2}")
        logger.info(f"✓ route3_decision: {route3}")
        
        assert route1 is not None, "route1_decision should not be None"
        assert route1 in ["skip", "analyze"], f"Invalid route1: {route1}"
        
        if route1 == "analyze":
            assert route2 is not None, "route2_decision should not be None for non-skip path"
            assert route2 in ["sentiment_only", "full_analysis"], f"Invalid route2: {route2}"
        
        # executive_brief 검증
        executive_brief = result.get("executive_brief")
        
        if route1 == "skip":
            assert executive_brief is None, "executive_brief should be None on skip path"
            logger.info("✓ Skip path: executive_brief is None (expected)")
        else:
            assert executive_brief is not None, "executive_brief should be generated for non-skip path"
            logger.info("✓ Executive brief generated")
            logger.info(f"  - Summary: {executive_brief.get('summary', '')[:100]}...")
        
        # 에러 로그 확인
        error_logs = result.get("error_logs", [])
        if error_logs:
            logger.warning(f"⚠️  {len(error_logs)} error(s) occurred:")
            for error in error_logs:
                logger.warning(f"   - [{error.get('node')}] {error.get('message')}")
        
        # 5단계: 데이터 충분성 검증
        logger.info(f"\n{'='*70}")
        logger.info("Step 5: Validating data sufficiency")
        logger.info(f"{'='*70}")
        
        validation = validate_data_sufficiency(spike_event["messages"], result)
        
        logger.info(f"✓ Total messages: {validation['total_messages']}")
        logger.info(f"✓ Has text content: {validation['has_text_content']}")
        logger.info(f"✓ Sentiment quality: {validation['sentiment_result_quality']}")
        logger.info(f"✓ Network analysis quality: {validation['network_analysis_quality']}")
        
        if validation["warnings"]:
            logger.warning("⚠️  Warnings:")
            for warning in validation["warnings"]:
                logger.warning(f"   - {warning}")
        
        if validation["recommendations"]:
            logger.info("💡 Recommendations:")
            for rec in validation["recommendations"]:
                logger.info(f"   - {rec}")
        
        # 최종 출력
        logger.info(f"\n{'='*70}")
        logger.info("✅ Full pipeline completed successfully!")
        logger.info(f"{'='*70}\n")
    
    def test_pipeline_data_availability(self, real_pipeline_enabled, collector):
        """
        데이터 가용성 검증
        
        확인:
          - Kafka에서 메시지 수집 가능한가?
          - 다양한 소스에서 메시지를 받는가?
          - 메시지 구조가 올바른가?
        """
        keyword = os.getenv("REAL_PIPELINE_TEST_KEYWORD", "테스트키워드")
        
        logger.info(f"Checking data availability for keyword: {keyword}")
        
        messages = collector.collect_messages(keyword)
        
        # 메시지 유형 분석
        types = {}
        sources = {}
        
        for msg in messages:
            msg_type = msg.get("type", "unknown")
            source = msg.get("source", "unknown")
            
            types[msg_type] = types.get(msg_type, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        logger.info(f"\nMessage types: {types}")
        logger.info(f"Message sources: {sources}")
        
        # 검증
        assert len(types) > 0, "No message types found"
        assert len(sources) > 0, "No message sources found"
        
        # 다양성 확인
        if len(sources) == 1:
            logger.warning("⚠️  Only one source, consider collecting from multiple sources")
        else:
            logger.info(f"✓ Data from {len(sources)} different sources")
        
        logger.info(f"\n✅ Data availability check passed!")
    
    def test_pipeline_message_content_quality(self, real_pipeline_enabled, collector):
        """
        메시지 콘텐츠 품질 검증
        
        확인:
          - 텍스트 길이가 충분한가?
          - 의미 있는 내용이 있는가?
          - 메트릭 정보가 있는가?
        """
        keyword = os.getenv("REAL_PIPELINE_TEST_KEYWORD", "테스트키워드")
        
        logger.info(f"Checking message content quality for keyword: {keyword}")
        
        messages = collector.collect_messages(keyword)
        
        quality_metrics = {
            "total": len(messages),
            "with_text": 0,
            "with_metrics": 0,
            "with_author": 0,
            "avg_text_length": 0,
            "avg_engagement": 0,
        }
        
        total_length = 0
        total_engagement = 0
        
        for msg in messages:
            content = msg.get("content_data", {})
            trend = msg.get("trend_data", {})
            
            # 텍스트
            text = content.get("text", "") or trend.get("rising_queries", [{}])[0].get("query", "")
            if text:
                quality_metrics["with_text"] += 1
                total_length += len(text)
            
            # 메트릭
            if content.get("metrics") or trend.get("interest_score"):
                quality_metrics["with_metrics"] += 1
            
            # 작성자
            if content.get("author_id"):
                quality_metrics["with_author"] += 1
            
            # 참여도
            metrics = content.get("metrics", {})
            engagement = metrics.get("likes", 0) + metrics.get("retweets", 0)
            total_engagement += engagement
        
        quality_metrics["avg_text_length"] = total_length / max(1, quality_metrics["with_text"])
        quality_metrics["avg_engagement"] = total_engagement / max(1, quality_metrics["total"])
        
        logger.info(f"\nQuality metrics:")
        logger.info(f"  - With text: {quality_metrics['with_text']}/{quality_metrics['total']}")
        logger.info(f"  - With metrics: {quality_metrics['with_metrics']}/{quality_metrics['total']}")
        logger.info(f"  - With author: {quality_metrics['with_author']}/{quality_metrics['total']}")
        logger.info(f"  - Avg text length: {quality_metrics['avg_text_length']:.1f} chars")
        logger.info(f"  - Avg engagement: {quality_metrics['avg_engagement']:.1f}")
        
        # 최소 기준
        assert quality_metrics["with_text"] >= quality_metrics["total"] * 0.8, \
            f"Not enough messages with text ({quality_metrics['with_text']}/{quality_metrics['total']})"
        
        assert quality_metrics["avg_text_length"] >= 5, \
            f"Text too short (avg {quality_metrics['avg_text_length']:.1f} chars)"
        
        logger.info(f"\n✅ Content quality check passed!")


# ============================================================================
# 메인
# ============================================================================

if __name__ == "__main__":
    exit_code = pytest.main([
        __file__,
        "-v",
        "-m", "real_pipeline",
        "-s",  # stdout 출력 활성화
    ])
    
    sys.exit(exit_code)
