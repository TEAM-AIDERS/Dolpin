import asyncio
import concurrent.futures
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List

from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

from src.schemas.kafka_schema import KafkaMessage

load_dotenv()
logger = logging.getLogger(__name__)

# 같은 키워드의 메시지를 묶어서 처리하기 위한 버퍼 윈도우 (초)
BUFFER_WINDOW_SECONDS = float(os.getenv("BUFFER_WINDOW_SECONDS", "15"))


class KafkaConsumer:
    def __init__(self, group_id: str = "dolpin-analyzer-group"):
        self.conf = {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            'sasl.mechanism': 'PLAIN',
            'security.protocol': 'SASL_SSL',
            'sasl.username': os.getenv('KAFKA_API_KEY'),
            'sasl.password': os.getenv('KAFKA_API_SECRET'),
            'group.id': group_id,
            # replay 모드면 과거 메시지까지 모두 읽고, 실시간이면 최신부터 읽음
            'auto.offset.reset': 'earliest' if os.getenv('MODE') == 'REPLAY' else 'latest',
            'enable.auto.commit': True,
        }
        self.consumer = Consumer(self.conf)
        self.topic = os.getenv('KAFKA_TOPIC')
        self.dlq_path = "failed_events_log.jsonl"
        self._graph = None  # compile_workflow lazy init

        # 키워드별 메시지 버퍼
        self._buffer: Dict[str, List[KafkaMessage]] = defaultdict(list)
        self._buffer_opened_at: Dict[str, float] = {}  # 키워드별 버퍼 최초 수신 시각

    def _get_graph(self):
        """워크플로우 그래프 싱글톤 (최초 호출 시 한 번만 컴파일)"""
        if self._graph is None:
            from src.dolpin_langgraph.graph import compile_workflow
            self._graph = compile_workflow()
            logger.info("LangGraph 워크플로우 컴파일 완료")
        return self._graph

    def run_pipeline_batch(self, msgs: List[KafkaMessage]) -> dict:
        """여러 KafkaMessage를 하나의 SpikeEvent로 묶어 파이프라인 실행"""
        from src.pipeline.transformer import kafka_messages_to_state

        state = kafka_messages_to_state(msgs)
        graph = self._get_graph()

        def _invoke():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(graph.ainvoke(state))
            finally:
                loop.close()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_invoke)
        executor.shutdown(wait=False)
        return future.result(timeout=180)

    def _flush_buffer(self, keyword: str, callback: Callable[[List[KafkaMessage]], None]):
        """키워드 버퍼를 플러시하고 콜백 호출"""
        msgs = self._buffer.pop(keyword, [])
        self._buffer_opened_at.pop(keyword, None)
        if msgs:
            logger.info(f"버퍼 플러시: [{keyword}] {len(msgs)}건 묶음 분석")
            callback(msgs)

    def _flush_expired_buffers(self, callback: Callable[[List[KafkaMessage]], None]):
        """윈도우가 만료된 키워드 버퍼 자동 플러시"""
        import time
        now = time.monotonic()
        expired = [
            kw for kw, opened in self._buffer_opened_at.items()
            if now - opened >= BUFFER_WINDOW_SECONDS
        ]
        for kw in expired:
            self._flush_buffer(kw, callback)

    # 메시지 소비하고 콜백으로 넘기는 메서드
    def consume(self, callback: Callable[[List[KafkaMessage]], None]):
        import time
        try:
            self.consumer.subscribe([self.topic])
            logger.info(f"Consumer 시작 (Mode: {os.getenv('MODE')}, 버퍼 윈도우: {BUFFER_WINDOW_SECONDS}초)")

            while True:
                # 만료된 버퍼 주기적으로 플러시
                self._flush_expired_buffers(callback)

                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())

                raw_data = msg.value().decode('utf-8')

                try:
                    data_dict = json.loads(raw_data)
                    validated_msg = KafkaMessage.model_validate(data_dict)
                    keyword = validated_msg.keyword

                    logger.info(f"메시지 수신 [{validated_msg.source}] 키워드: {keyword}")

                    # 버퍼에 추가
                    if keyword not in self._buffer_opened_at:
                        self._buffer_opened_at[keyword] = time.monotonic()
                    self._buffer[keyword].append(validated_msg)

                    # trend 메시지 수신 시 즉시 플러시 (집계 데이터라 바로 처리)
                    if validated_msg.type == "trend":
                        self._flush_buffer(keyword, callback)

                except Exception as e:
                    logger.error(f"❌ 데이터 검증 실패: {e}")
                    self._handle_failure(raw_data, str(e))
        finally:
            self.consumer.close()

    # 실패한 메시지 따로 저장
    def _handle_failure(self, raw_data: str, error_msg: str):
        failure_entry = {
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": error_msg,
            "payload": raw_data,
        }
        with open(self.dlq_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(failure_entry, ensure_ascii=False) + "\n")
        logger.warning(f"⚠️ 실패 데이터가 {self.dlq_path}에 기록되었습니다.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    consumer = KafkaConsumer()

    def process_data(msgs: List[KafkaMessage]):
        keyword = msgs[0].keyword
        logger.info(f"--- 분석 시작 --- 키워드: {keyword} ({len(msgs)}건)")
        try:
            result = consumer.run_pipeline_batch(msgs)
            skipped = result.get("skipped", False)
            if skipped:
                logger.info(f"⏭️  스킵됨: [{keyword}] reason={result.get('skip_reason')}")
            else:
                from src.pipeline.result_store import save_result

                save_result(result)

                brief = (result.get("executive_brief") or {}).get("summary", "N/A")
                logger.info(f"✅ 분석 완료: [{keyword}] {brief}")
        except Exception as e:
            logger.error(f"❌ 파이프라인 실패: [{keyword}] {e}")

    consumer.consume(callback=process_data)
