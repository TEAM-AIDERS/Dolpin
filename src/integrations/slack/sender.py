# src/integrations/slack/sender.py

import os
import logging
from typing import Dict, Any
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


def send_to_slack(message: Dict[str, Any]) -> bool:
    try:
        # 환경변수 확인
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        channel_id = os.getenv("SLACK_CHANNEL_ID")
        
        if not bot_token or not channel_id:
            logger.warning(
                "⚠️ SLACK BOT TOKEN 또는 SLACK CHANNEL ID가 설정되지 않았습니다"
            )
            return False
        
        # Slack 클라이언트 생성
        client = WebClient(token=bot_token)
        
        # 메시지 전송
        response = client.chat_postMessage(
            channel=channel_id,
            text=message.get("text", "DOLPIN 이슈 리포트"),
            blocks=message.get("blocks", [])
        )
        
        if response["ok"]:
            logger.info("Slack 메시지 전송 성공")
            return True
        else:
            logger.error(f"Slack 메시지 전송 실패: {response}")
            return False
    
    except SlackApiError as e:
        logger.error(f"Slack API 에러: {e.response['error']}")
        return False
    
    except Exception as e:
        logger.error(f"Slack 전송 예외: {e}")
        return False