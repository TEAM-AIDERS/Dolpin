"""
DOLPIN 실시간 모니터링 데모 실행기

사용법:
  python -m src.pipeline.run_system_producer
  python -m src.pipeline.run_system_producer 엔시티위시
"""

import sys
import time
from src.pipeline.kafka_producer import KafkaProducer
from src.schemas.kafka_schema import KafkaMessage, ContentData

# ── 데모 데이터 (키워드별 시나리오) ─────────────────────────

DEMO_DATA = {
    "엔시티 위시": {
        "source_mix": ["twitter", "twitter", "instiz", "twitter", "instiz",
                       "twitter", "twitter", "instiz", "twitter", "google_trends"],
        "messages": [
            "엔시티 위시 콘서트 너무 좋았다ㅠㅠ 팀위시처하자",
            "위시 선공개곡 진짜 미쳤다 계속 듣고있어 중독됨",
            "엔시티 위시 사랑해",
            "위시 콘서트 갔다온 후기 올림 퀄리티 진짜 기대 이상이었어",
            "엔시티 위시 선공개 뮤직비디오 비주얼 미침 계속 돌려보는 중",
            "위시 콘서트 직관한 사람 여기 있음? 나 완전 힐링하고 왔어",
            "엔시티 위시 컴백 준비 많이 했나봐 이 퀄리티가 말이 되나",
            "위시 선공개 스밍 열심히 하자",
            "엔시티 위시 콘서트 앙코르 때 진짜 울뻔 했음 감동이야",
            "위시 이번 컴백 역대급이다 선공개부터 이러면 정규는 어떨지 기대됨",
        ],
        "metrics": [
            {"likes": 1240, "retweets": 380, "replies": 92},
            {"likes": 870,  "retweets": 210, "replies": 54},
            {"likes": 2100, "retweets": 560, "replies": 130},
            {"likes": 430,  "retweets": 88,  "replies": 47},
            {"likes": 3400, "retweets": 920, "replies": 201},
            {"likes": 290,  "retweets": 61,  "replies": 33},
            {"likes": 760,  "retweets": 195, "replies": 68},
            {"likes": 510,  "retweets": 140, "replies": 29},
            {"likes": 980,  "retweets": 244, "replies": 87},
            {"likes": 1870, "retweets": 430, "replies": 114},
        ],
    },
}

_DEFAULT_KEYWORD = "엔시티 위시"

_SOURCE_LABEL = {
    "twitter":       "Twitter(X)",
    "instiz":        "인스티즈",
    "google_trends": "Google 트렌드",
}



def main():
    # ── 키워드 결정 ──────────────────────────────────────────
    if len(sys.argv) > 1:
        keyword = " ".join(sys.argv[1:]).strip()
    else:
        raw = input("\n모니터링할 키워드를 입력하세요: ").strip()
        keyword = raw if raw else _DEFAULT_KEYWORD

    data = DEMO_DATA.get(keyword, DEMO_DATA[_DEFAULT_KEYWORD])
    messages  = data["messages"]
    sources   = data["source_mix"]
    metrics   = data["metrics"]

    print(f"\n{'─'*52}")
    print(f"  🐬 DOLPIN 실시간 모니터링")
    print(f"  키워드: {keyword}")
    print(f"{'─'*52}\n")

    # ── 수집 단계 ─────────────────────────────────────────────
    print("[ 1/3 ]  데이터 수집 중...\n")
    time.sleep(0.4)

    producer = KafkaProducer()

    for i, (text, source, metric) in enumerate(zip(messages, sources, metrics), start=1):
        source_label = _SOURCE_LABEL.get(source, source)
        print(f"  [{source_label}] {text[:45]}{'...' if len(text) > 45 else ''}")
        time.sleep(0.35)

        msg = KafkaMessage(
            type="post",
            source=source,
            keyword=keyword,
            content_data=ContentData(
                text=text,
                author_id=f"user_{i:04d}",
                metrics=metric,
            ),
        )
        producer.send(msg)

    producer.flush()

    print(f"\n  총 {len(messages)}건 수집 완료\n")
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    main()
