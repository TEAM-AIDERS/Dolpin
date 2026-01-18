import json
import os
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Producer
from src.schemas.kafka_schema import KafkaMessage

load_dotenv()


class KafkaProducer:
    def __init__(self):
        self.conf = {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': os.getenv('KAFKA_API_KEY'),
            'sasl.password': os.getenv('KAFKA_API_SECRET'),

            # 비용 최적화
            'linger.ms': 10000,
            'batch.size': 32768,
            'compression.type': 'gzip',

            # 신뢰성
            'retries': 5,
            'retry.backoff.ms': 500,
            'acks': 'all',
            'delivery.timeout.ms': 30000,
        }

        self.producer = Producer(self.conf)
        self.topic = os.getenv('KAFKA_TOPIC')
        self.dlq_file = "failed_events_log.jsonl"

    # 실패 메시지 기록할 DLQ
    def _write_to_dlq(self, data: str, error: str):
        with open(self.dlq_file, "a", encoding="utf-8") as f:
            log = {
                "occurred_at": datetime.now().isoformat(),
                "error": error,
                "payload": json.loads(data),
            }
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

    # 전송 결과 콜백
    def _delivery_report(self, err, msg):
        if err is not None:
            print(f"❌ Kafka 전송 실패: {err}")
            self._write_to_dlq(msg.value().decode('utf-8'), str(err))
        else:
            print(
                f"Kafka 성공: {msg.topic()} "
                f"[{msg.partition()}] offset {msg.offset()}"
            )

    def send(self, message: KafkaMessage):
        try:
            data = message.model_dump()

            # REPLAY 모드일 경우 타임아웃 업데이트 로직
            if os.getenv("MODE") == "REPLAY":
                data["collected_at"] = datetime.now().isoformat()

            # 아티스트명 + 시간 조합으로 키 생성하여 부하 분산
            current_hour = datetime.now().strftime("%Y%m%d%H")
            routing_key = f"{data['keyword']}-{current_hour}"

            self.producer.produce(
                topic=self.topic,
                key=routing_key,
                value=json.dumps(data, default=str).encode('utf-8'),
                callback=self._delivery_report,
            )
            self.producer.poll(0)

        except Exception as e:
            print(f"❌ Producer 내부 에러: {e}")
            self._write_to_dlq(
                json.dumps(message.model_dump(), default=str),
                str(e),
            )

    def flush(self):
        self.producer.flush()
