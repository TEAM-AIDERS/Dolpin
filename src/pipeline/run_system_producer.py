from src.pipeline.kafka_producer import KafkaProducer
from src.schemas.kafka_schema import KafkaMessage, ContentData

producer = KafkaProducer()

messages = [
    "브랜드논란 때문에 불매하자는 반응이 늘고 있다",
    "이번 브랜드논란 해명 필요하다",
    "팬들 사이에서 boycott 얘기가 급격히 퍼진다",
    "브랜드논란 관련 부정 반응이 집중되고 있다",
    "이번 이슈는 이미지 타격이 크다",
    "브랜드논란 관련 해명문이 필요해 보인다",
    "팬덤에서 불매 해시태그가 확산 중이다",
    "논란 대응이 늦어서 더 커지는 분위기다",
    "브랜드논란 때문에 실망 반응이 많다",
    "이번 건은 위기 대응이 필요하다",
] 

for i, text in enumerate(messages, start=1):
    msg = KafkaMessage(
        type="post",
        source="twitter",
        keyword="브랜드논란",
        content_data=ContentData(
            text=text,
            author_id=f"demo_user_{i}",
            metrics={
                "likes": 50 + i,
                "retweets": 10 + (i % 7),
                "replies": 3 + (i % 4),
            },
        ),
    )
    producer.send(msg)

producer.flush()
print("✅ 시스템용 Kafka 메시지 전송 완료")
