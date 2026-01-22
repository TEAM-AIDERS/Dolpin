import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Callable

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
            'enable.auto.commit': True
        }
        self.consumer = Consumer(self.conf)
        self.topic = os.getenv('KAFKA_TOPIC')
        self.dlq_path = "failed_events_log.jsonl"
    
    # 메시지 소비하고 콜백으로 넘기는 메서드 
    def consume(self, callback: Callable[[KafkaMessage], None]):
        try:
            self.consumer.subscribe([self.topic])
            logger.info(f"Consumer 시작 (Mode: {os.getenv('MODE')})")
            
            while True:
                # 메시지 기다림, 없으면 루프 
                msg = self.consumer.poll(1.0)
                if msg is None: continue
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
                    
                    # Trace ID 확인
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
            "payload": raw_data
        }
        with open(self.dlq_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(failure_entry, ensure_ascii=False) + "\n")
        logger.warning(f"⚠️ 실패 데이터가 {self.dlq_path}에 기록되었습니다.")
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def process_data(msg: KafkaMessage):
        # 나중에 에이전트로 연결될 지점 
        print(f"--- 분석 시작 ---\n키워드: {msg.keyword}\n내용: {msg.content_data.text[:50] if msg.content_data else 'Trend Data'}")

    consumer = KafkaConsumer()
    consumer.consume(callback=process_data)