"""
Kafka Consumer Cloud Run 진입점.
Cloud Run은 HTTP 포트가 열려 있어야 하므로 헬스체크 서버를 백그라운드로 실행한다.
"""

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


def _run_health_server():
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    logger.info(f"헬스체크 서버 시작: 포트 {port}")
    server.serve_forever()


def _run_consumer():
    from src.pipeline.kafka_consumer import KafkaConsumer
    from src.schemas.kafka_schema import KafkaMessage

    consumer = KafkaConsumer()

    def process_data(msgs: List[KafkaMessage]):
        keyword = msgs[0].keyword
        logger.info(f"--- 분석 시작 --- 키워드: {keyword} ({len(msgs)}건)")
        try:
            result = consumer.run_pipeline_batch(msgs)
            if result.get("skipped"):
                logger.info(f"⏭️  스킵됨: [{keyword}] reason={result.get('skip_reason')}")
            else:
                from src.pipeline.result_store import save_result
                save_result(result)
                brief = (result.get("executive_brief") or {}).get("summary", "N/A")
                logger.info(f"✅ 분석 완료: [{keyword}] {brief}")
        except Exception as e:
            logger.error(f"❌ 파이프라인 실패: [{keyword}] {e}")

    consumer.consume(callback=process_data)


if __name__ == "__main__":
    health_thread = threading.Thread(target=_run_health_server, daemon=True)
    health_thread.start()

    _run_consumer()
