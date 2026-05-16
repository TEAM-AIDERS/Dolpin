# tests/test_slack_send.py

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations.slack import format_to_slack, send_to_slack
from src.integrations.slack.formatter import DEFAULT_DASHBOARD_URL
from dotenv import load_dotenv
load_dotenv()

def test_send_real_slack():
    """실제 Slack 전송 테스트 (상세 분석보기 버튼 + 대시보드 URL 포함)"""

    if not os.getenv("SLACK_BOT_TOKEN"):
        print("⚠️ SLACK_BOT_TOKEN 없음, 테스트 스킵")
        return

    mock_state = {
        "trace_id": "dashtest-001",
        "dashboard_url": DEFAULT_DASHBOARD_URL,
        "spike_event": {
            "keyword": "테스트 아티스트",
            "time_window": "1h",
            "messages": [{"source": "twitter"}, {"source": "instiz"}],
        },
        "spike_analysis": {"spike_rate": 3.2, "spike_nature": "positive", "spike_type": "organic"},
        "sentiment_result": {
            "dominant_sentiment": "support",
            "sentiment_distribution": {"support": 0.72, "neutral": 0.18},
            "representative_messages": {"support": ["진짜 대박이다", "역시 믿고 보는 아티스트"]},
        },
        "playbook": {
            "situation_type": "opportunity",
            "recommended_actions": [
                {
                    "action": "amplify_viral",
                    "urgency": "medium",
                    "description": "긍정 바이럴 콘텐츠 확산",
                    "rationale": "팬덤 참여율이 높아 자연 확산 가능성 큼",
                }
            ],
        },
        "executive_brief": {
            "severity_score": 6,
            "issue_polarity": "positive",
            "spike_summary": "3.2배 급등 — 신보 발매 이후 자연 발생 버즈",
            "sentiment_summary": "응원/지지 72% 우세, 전반적 긍정 기류",
            "legal_summary": "법률 검토 미수행",
            "opportunity_summary": "팬 참여 캠페인 + 공식 채널 콘텐츠 연계 추천",
            "generated_at": "2026-05-14T09:00:00Z",
            "analysis_duration_seconds": 4.1,
        },
    }

    message = format_to_slack(mock_state)

    # 버튼 URL 확인
    actions_block = next((b for b in message["blocks"] if b.get("type") == "actions"), None)
    detail_btn = next(
        (e for e in actions_block["elements"] if e.get("action_id") == "action_detail"), None
    ) if actions_block else None
    btn_url = detail_btn.get("url") if detail_btn else None
    print(f"버튼 URL: {btn_url or '(없음)'}")

    success = send_to_slack(message)
    if success:
        print("Slack 전송 성공! [🔍 상세 분석 보기] 버튼을 눌러 대시보드를 확인하세요.")
    else:
        print("Slack 전송 실패!")

if __name__ == "__main__":
    test_send_real_slack()