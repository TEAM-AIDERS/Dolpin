import asyncio
import concurrent.futures
import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

from src.schemas.kafka_schema import KafkaMessage

load_dotenv()
logger = logging.getLogger(__name__)


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

    def _get_graph(self):
        """워크플로우 그래프 싱글톤 (최초 호출 시 한 번만 컴파일)"""
        if self._graph is None:
            from src.dolpin_langgraph.graph import compile_workflow
            self._graph = compile_workflow()
            logger.info("LangGraph 워크플로우 컴파일 완료")
        return self._graph

    def run_pipeline(self, msg: KafkaMessage) -> dict:
        """
        KafkaMessage → AnalysisState 변환 후 LangGraph 워크플로우 실행

        Args:
            msg: Kafka에서 수신한 검증된 메시지

        Returns:
            완료된 AnalysisState (executive_brief 포함)
        """
        from src.pipeline.transformer import kafka_message_to_state

        state = kafka_message_to_state(msg)
        graph = self._get_graph()

        def _invoke():
            # legal_rag_node가 async이므로 별도 스레드에서 새 이벤트 루프로 실행
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

    # 메시지 소비하고 콜백으로 넘기는 메서드
    def consume(self, callback: Callable[[KafkaMessage], None]):
        try:
            self.consumer.subscribe([self.topic])
            logger.info(f"Consumer 시작 (Mode: {os.getenv('MODE')})")

            while True:
                # 메시지 기다림, 없으면 루프
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
                    # 데이터 검증
                    data_dict = json.loads(raw_data)
                    validated_msg = KafkaMessage.model_validate(data_dict)

                    trace_id = getattr(validated_msg, 'message_id', 'no-id')
                    logger.info(f"메시지 수신 성공 [{validated_msg.source}] ID: {trace_id}")

                    # 외부 함수로 메시지 넘김
                    callback(validated_msg)

                except Exception as e:
                    # 검증 실패 시 DLQ 보관
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

    def process_data(msg: KafkaMessage):
        logger.info(f"--- 분석 시작 --- 키워드: {msg.keyword}")
        try:
            result = consumer.run_pipeline(msg)
            skipped = result.get("skipped", False)
            if skipped:
                logger.info(f"⏭️  스킵됨: [{msg.keyword}] reason={result.get('skip_reason')}")
                logger.info(f"spike_summary={result.get('spike_summary')}")
                logger.info(f"executive_brief={result.get('executive_brief')}")
            else:
                from src.pipeline.result_store import save_result
                from src.integrations.slack.formatter import format_to_slack
                from src.integrations.slack.sender import send_to_slack

                save_result(result)

                slack_message = format_to_slack(result)
                sent = send_to_slack(slack_message)
                
                brief = (result.get("executive_brief") or {}).get("summary", "N/A")
                logger.info(f"✅ 분석 완료: [{msg.keyword}] {brief}")
                logger.info(f"📨 Slack 전송 결과: {sent}")
        except Exception as e:
            logger.error(f"❌ 파이프라인 실패: [{msg.keyword}] {e}")

    consumer.consume(callback=process_data)
