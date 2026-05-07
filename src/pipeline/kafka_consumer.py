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
            # 수동 커밋으로 전환 → 처리 완료 후에만 offset 커밋 (at-least-once 보장)
            # True일 경우 poll() 직후 offset이 커밋되어 LangGraph 처리 실패 시 메시지 유실
            'enable.auto.commit': False,
            # LangGraph 파이프라인 최대 180초 + 여유 60초
            # 기본값(300초)과 겹치지 않지만 명시해 리밸런싱 트리거 조건을 코드에서 확인 가능하게 함
            'max.poll.interval.ms': 300000,
        }
        self.consumer = Consumer(self.conf)
        self.topic = os.getenv('KAFKA_TOPIC')
        self.dlq_path = "failed_events_log.jsonl"
        self._graph = None  # compile_workflow lazy init
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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
            # legal_rag_node가 async이므로 별도 스레드에서 새 이벤트 루프로 실행
            return asyncio.run(graph.ainvoke(state))

        future = self._executor.submit(_invoke)
        try:
            return future.result(timeout=180)
        except concurrent.futures.TimeoutError:
            # 타임아웃 시 executor 교체
            # stuck worker가 self._executor의 유일한 슬롯을 점유한 상태이므로,
            # 교체하지 않으면 이후 모든 submit()이 큐 뒤에 쌓여 연쇄 타임아웃 발생
            # → consumer 전체가 재시작 전까지 마비됨
            logger.error("LangGraph 타임아웃 (180초) — executor 교체하여 다음 메시지 처리 보장")
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            raise

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

                commit_eligible = False
                try:
                    data_dict = json.loads(raw_data)
                    validated_msg = KafkaMessage.model_validate(data_dict)
                    keyword = validated_msg.keyword

                    logger.info(f"메시지 수신 [{validated_msg.source}] 키워드: {keyword}")

                    # 버퍼에 추가
                    if keyword not in self._buffer_opened_at:
                        self._buffer_opened_at[keyword] = time.monotonic()
                    self._buffer[keyword].append(validated_msg)
                    commit_eligible = True  # 버퍼에 정상 추가됨

                    # trend 메시지 수신 시 즉시 플러시 (집계 데이터라 바로 처리)
                    if validated_msg.type == "trend":
                        self._flush_buffer(keyword, callback)

                except Exception as e:
                    # 검증/파이프라인 실패 시 DLQ 보관
                    logger.error(f"❌ 처리 실패 (DLQ 저장 시도): {e}")
                    self._handle_failure(raw_data, str(e))
                    # _handle_failure가 성공해야만 True — 디스크 풀/권한 오류로 DLQ 저장이
                    # 실패하면 False를 유지해 커밋하지 않음 → 메시지가 Kafka에 남아 재처리 가능
                    commit_eligible = True  # DLQ 저장 성공

                finally:
                    if commit_eligible:
                        # 정상 처리 완료 또는 DLQ 저장 성공 시에만 커밋
                        self.consumer.commit(message=msg)
                    else:
                        # DLQ 저장도 실패한 경우 커밋하지 않음
                        # → 메시지가 Kafka에 남아 재처리 기회 보존
                        logger.warning("⚠️ DLQ 저장 실패 — 커밋 보류, 메시지 재처리 예정")
        finally:
            self._executor.shutdown(wait=True)
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
