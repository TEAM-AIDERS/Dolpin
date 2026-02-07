# src/integrations/slack/formatter.py

from typing import Dict, Any, List, Optional
from src.dolpin_langgraph.state import AnalysisState


def format_to_slack(state: AnalysisState) -> Dict[str, Any]:
    """
    AnalysisState → Slack Block Kit 변환
    
    현재: 완성된 노드 데이터만 변환
    미완성: "분석 중" 또는 생략
    
    Args:
        state: 전체 분석 상태
    
    Returns:
        dict: Slack message payload (blocks + text)
    """
    
    blocks = []
    exec_brief = state.get("executive_brief", {})
    playbook = state.get("playbook", {})
    
    # ===== 1. Header =====
    situation = exec_brief.get("summary", "분석 중")
    severity_icon = _get_severity_icon(exec_brief.get("severity_score", 5))
    
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"{severity_icon} DOLPIN 이슈 리포트: {situation}"
        }
    })
    
    # ===== 2. 요약 정보 (Fields) =====
    blocks.append({
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": f"*우선순위:*\n{_format_priority(playbook.get('priority', 'unknown'))}"
            },
            {
                "type": "mrkdwn",
                "text": f"*트렌드:*\n{_format_trend(exec_brief.get('trend_direction', 'stable'))}"
            },
            {
                "type": "mrkdwn",
                "text": f"*이슈 성격:*\n{_format_polarity(exec_brief.get('issue_polarity', 'mixed'))}"
            },
            {
                "type": "mrkdwn",
                "text": f"*심각도:*\n{exec_brief.get('severity_score', 5)}/10"
            }
        ]
    })
    
    blocks.append({"type": "divider"})
    
    # ===== 3. 현재 상황 (ExecBrief) =====
    spike_summary = exec_brief.get("spike_summary")
    if spike_summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*📊 현재 상황*\n{spike_summary}"
            }
        })
    
    # ===== 4. 팬 반응 (SentimentAgent) =====
    sentiment_summary = exec_brief.get("sentiment_summary")
    if sentiment_summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*💬 팬 반응*\n{sentiment_summary}"
            }
        })
    
    # ===== 5. 권장 조치 (PlaybookAgent) =====
    if playbook and playbook.get("recommended_actions"):
        actions_text = _format_actions(playbook["recommended_actions"])
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*📋 권장 조치*\n{actions_text}"
            }
        })
    
    # ===== 6. 기회 요약 (Opportunity만) =====
    opportunity_summary = exec_brief.get("opportunity_summary")
    if opportunity_summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*✨ 확산 기회*\n{opportunity_summary}"
            }
        })
    
    # ===== 7. 법적 리스크 (LegalRAG) =====
    legal_summary = exec_brief.get("legal_summary")
    if legal_summary and legal_summary != "법률 검토 미수행":
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*⚖️ 법적 검토*\n{legal_summary}"
            }
        })
    
    # ===== 8. 인과관계 (Causality) =====
    causality = state.get("causality_result")
    if causality:
        trigger = causality.get("trigger_source", "unknown")
        cascade = causality.get("cascade_pattern", "unknown")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔗 확산 경로*\n트리거: {trigger} | 패턴: {cascade}"
            }
        })
    
    # ===== 9. 에러 메시지 (있으면) =====
    user_message = exec_brief.get("user_message")
    if user_message:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"⚠️ *알림*\n{user_message}"
            }
        })
    
    blocks.append({"type": "divider"})
    
    # ===== 10. Footer =====
    generated_at = exec_brief.get("generated_at", "")
    duration = exec_brief.get("analysis_duration_seconds", 0)
    trace_id = state.get("trace_id", "unknown")
    
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"생성: {generated_at} | 분석 시간: {duration:.1f}초 | Trace: `{trace_id[:8]}`"
            }
        ]
    })
    
    return {
        "blocks": blocks,
        "text": f"DOLPIN 이슈 리포트: {situation}"  # fallback
    }


# ============================================================
# 헬퍼 함수들
# ============================================================

def _get_severity_icon(severity_score: int) -> str:
    """Severity score → 이모지"""
    if severity_score >= 8:
        return "🔴"
    elif severity_score >= 6:
        return "🟠"
    elif severity_score >= 4:
        return "🟡"
    else:
        return "🟢"


def _format_priority(priority: str) -> str:
    """Priority → 한글 + 이모지"""
    priority_map = {
        "urgent": "🔴 긴급",
        "high": "🟠 높음",
        "medium": "🟡 보통",
        "low": "🟢 낮음"
    }
    return priority_map.get(priority, "⚪ 알 수 없음")


def _format_trend(trend: str) -> str:
    """Trend direction → 한글 + 이모지"""
    trend_map = {
        "escalating": "📈 악화",
        "declining": "📉 개선",
        "stable": "➡️ 안정"
    }
    return trend_map.get(trend, "➡️ 안정")


def _format_polarity(polarity: str) -> str:
    """Issue polarity → 한글 + 이모지"""
    polarity_map = {
        "positive": "😊 긍정",
        "negative": "😟 부정",
        "mixed": "😐 혼재"
    }
    return polarity_map.get(polarity, "😐 혼재")


def _format_actions(actions: List[Dict]) -> str:
    """Actions 포맷팅"""
    if not actions:
        return "권장 조치 없음"
    
    lines = []
    for i, action in enumerate(actions[:3], 1):  # 상위 3개만
        action_type = action.get("action", "unknown")
        description = action.get("description", "")
        urgency = action.get("urgency", "medium")
        
        urgency_icon = {
            "immediate": "🔴",
            "urgent": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }.get(urgency, "⚪")
        
        # description이 있으면 사용, 없으면 action_type 사용
        if description:
            lines.append(f"{urgency_icon} {i}. {description}")
        else:
            # action_type을 한글로 매핑
            action_text = _translate_action_type(action_type)
            lines.append(f"{urgency_icon} {i}. {action_text}")
    
    return "\n".join(lines)


def _translate_action_type(action_type: str) -> str:
    """Action type → 한글"""
    action_map = {
        "issue_statement": "공식 입장문 발표",
        "amplify_viral": "긍정 바이럴 확산",
        "legal_response": "법적 대응",
        "monitor_only": "모니터링 지속",
        "engage_influencers": "허브 계정 협력",
        "internal_review": "내부 조사",
        "prepare_communication": "커뮤니케이션 준비"
    }
    return action_map.get(action_type, action_type)