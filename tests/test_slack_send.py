# tests/test_slack_send.py

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations.slack import format_to_slack, send_to_slack
from dotenv import load_dotenv
load_dotenv()

def test_send_real_slack():
    """실제 Slack 전송 테스트"""
    
    # 환경변수 확인
    if not os.getenv("SLACK_BOT_TOKEN"):
        print("⚠️ SLACK_BOT_TOKEN 없음, 테스트 스킵")
        return
    
    # 테스트 메시지
    message = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "🧪 *DOLPIN 테스트 메시지*\nSlack 연동이 정상 작동합니다!"
                }
            }
        ],
        "text": "DOLPIN 테스트"
    }
    
    # 전송
    success = send_to_slack(message)
    
    if success:
        print("Slack 전송 성공!")
    else:
        print("Slack 전송 실패!")

if __name__ == "__main__":
    test_send_real_slack()