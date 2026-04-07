from src.pipeline.kafka_producer import KafkaProducer
from src.schemas.kafka_schema import KafkaMessage, ContentData

producer = KafkaProducer()

messages = [
    "이번 대응 너무 실망이다",
    "보이콧 해야 하는 거 아닌가",
    "팬들 반응 진짜 안 좋다",
    "논란이 너무 커졌다",
    "공식 입장 빨리 내야 한다",
] * 10

for i, text in enumerate(messages, start=1):
    msg = KafkaMessage(
        type="post",
        source="twitter",
        keyword="엔시티 위시",
        content_data=ContentData(
            text=text,
            author_id=f"demo_user_{i}",
            metrics={"likes": 10, "retweets": 3, "replies": 1},
        ),
    )
    producer.send(msg)

producer.flush()
print("✅ 시스템용 Kafka 메시지 전송 완료")
