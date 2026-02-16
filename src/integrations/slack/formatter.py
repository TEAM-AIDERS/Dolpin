# src/integrations/slack/formatter.py

import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.dolpin_langgraph.state import AnalysisState

# Slack 메시지 길이 제한 (공식 권장값)
MAX_SECTION_TEXT = 900
MAX_ACTIONS = 3
DEFAULT_ALLOWED_LINK_HOSTS = {"dashboard.example.com"}


def format_to_slack(state: AnalysisState) -> Dict[str, Any]:
    """
    AnalysisState를 Slack Block Kit payload로 변환.

    - 모든 텍스트는 mrkdwn escape 처리
    - 외부 링크(dashboard/incident)는 allowlist 검증 후 clickable 링크로
    """
    blocks: List[Dict[str, Any]] = []
    exec_brief = state.get("executive_brief") or {}
    playbook = state.get("playbook") or {}

    situation_raw = exec_brief.get("summary", "분석 중")
    situation = _truncate_for_slack(str(situation_raw), 120)
    severity_icon = _get_severity_icon(int(exec_brief.get("severity_score", 5)))

    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{severity_icon} DOLPIN 이슈 리포트 | {situation}",
            },
        }
    )

    blocks.append(
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*우선순위:*\n{_format_priority(playbook.get('priority', 'unknown'))}"},
                {"type": "mrkdwn", "text": f"*트렌드:*\n{_format_trend(exec_brief.get('trend_direction', 'stable'))}"},
                {"type": "mrkdwn", "text": f"*이슈 성격:*\n{_format_polarity(exec_brief.get('issue_polarity', 'mixed'))}"},
                {"type": "mrkdwn", "text": f"*심각도:*\n{int(exec_brief.get('severity_score', 5))}/10"},
            ],
        }
    )
    blocks.append({"type": "divider"})

    _add_mrkdwn_section(blocks, "? 현재 상황", exec_brief.get("spike_summary"))
    _add_mrkdwn_section(blocks, "? 팬 반응", exec_brief.get("sentiment_summary"))

    recommended_actions = playbook.get("recommended_actions") or []
    if recommended_actions:
        actions_text = _format_actions(recommended_actions)
        _add_mrkdwn_section(blocks, "? 권장 조치", actions_text, already_safe=True)

    _add_mrkdwn_section(blocks, "? 확산 기회", exec_brief.get("opportunity_summary"))

    legal_summary = exec_brief.get("legal_summary")
    if legal_summary and legal_summary != "법률 검토 미수행":
        _add_mrkdwn_section(blocks, "?? 법적 검토", legal_summary)

    causality = state.get("causality_result") or {}
    if causality:
        trigger = _escape_mrkdwn_text(str(causality.get("trigger_source", "unknown")))
        cascade = _escape_mrkdwn_text(str(causality.get("cascade_pattern", "unknown")))
        _add_mrkdwn_section(blocks, "? 확산 경로", f"트리거: {trigger} | 패턴: {cascade}", already_safe=True)

    user_message = exec_brief.get("user_message")
    if user_message:
        _add_mrkdwn_section(blocks, "?? 알림", user_message)

    link_line = _build_link_line(exec_brief, state)
    if link_line:
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": link_line}],
            }
        )

    blocks.append({"type": "divider"})

    generated_at = _escape_mrkdwn_text(str(exec_brief.get("generated_at", "")))
    duration = float(exec_brief.get("analysis_duration_seconds", 0.0) or 0.0)
    trace_id = str(state.get("trace_id", "unknown"))[:8]

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"생성: {generated_at} | 분석 시간: {duration:.1f}초 | Trace: `{trace_id}`",
                }
            ],
        }
    )

    return {
        "blocks": blocks,
        "text": f"DOLPIN 이슈 리포트 | {situation}",
    }


def _add_mrkdwn_section(
    blocks: List[Dict[str, Any]],
    title: str,
    content: Optional[str],
    *,
    already_safe: bool = False,
) -> None:
    if not content:
        return
    body = content if already_safe else _escape_mrkdwn_text(str(content))
    body = _truncate_for_slack(body, MAX_SECTION_TEXT)
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{title}*\n{body}",
            },
        }
    )


def _escape_mrkdwn_text(text: str) -> str:
    """Slack mrkdwn에서 안전하게 표시하기 위해 특수문자 escape"""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for mention in ("@here", "@channel", "@everyone"):
        escaped = escaped.replace(mention, f"`{mention}`")
    return escaped


def _truncate_for_slack(text: str, max_len: int) -> str:
    """텍스트 길이 제한 (Slack API 제한 대응)"""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _build_link_line(exec_brief: Dict[str, Any], state: AnalysisState) -> str:
    """대시보드/인시던트 링크 라인 생성"""
    links: List[str] = []
    dashboard = _safe_link(exec_brief.get("dashboard_url"), "대시보드")
    incident = _safe_link(exec_brief.get("incident_url"), "인시던트")

    if dashboard:
        links.append(dashboard)
    if incident:
        links.append(incident)

    trace_id = str(state.get("trace_id", ""))
    if trace_id:
        links.append(f"Trace `{trace_id[:8]}`")

    return " | ".join(links)


def _safe_link(url: Optional[str], label: str) -> str:
    """URL 검증 후 안전한 Slack 링크 생성 (allowlist 기반)"""
    if not url:
        return ""

    parsed = urlparse(str(url))
    if parsed.scheme != "https" or not parsed.netloc:
        return ""

    allowlist_raw = os.getenv("SLACK_LINK_ALLOWLIST", "")
    allowed_hosts = {
        host.strip().lower()
        for host in allowlist_raw.split(",")
        if host.strip()
    }
    if not allowed_hosts:
        allowed_hosts = set(DEFAULT_ALLOWED_LINK_HOSTS)

    host = parsed.netloc.lower()
    if host not in allowed_hosts:
        return ""

    safe_label = _escape_mrkdwn_text(label)
    return f"<{url}|{safe_label}>"


def _get_severity_icon(severity_score: int) -> str:
    if severity_score >= 8:
        return ":red_circle:"
    if severity_score >= 6:
        return ":orange_circle:"
    if severity_score >= 4:
        return ":yellow_circle:"
    return ":green_circle:"


def _format_priority(priority: str) -> str:
    priority_map = {
        "urgent": ":red_circle: 긴급",
        "high": ":orange_circle: 높음",
        "medium": ":yellow_circle: 보통",
        "low": ":green_circle: 낮음",
    }
    return priority_map.get(priority, "? 알 수 없음")


def _format_trend(trend: str) -> str:
    trend_map = {
        "escalating": ":chart_with_upwards_trend: 악화",
        "declining": ":chart_with_downwards_trend: 개선",
        "stable": ":straight_ruler: 안정",
    }
    return trend_map.get(trend, ":straight_ruler: 안정")


def _format_polarity(polarity: str) -> str:
    polarity_map = {
        "positive": ":large_green_circle: 긍정",
        "negative": ":red_circle: 부정",
        "mixed": ":large_yellow_circle: 혼재",
    }
    return polarity_map.get(polarity, ":large_yellow_circle: 혼재")


def _format_actions(actions: List[Dict[str, Any]]) -> str:
    """권장 조치 목록 포맷팅"""
    if not actions:
        return "권장 조치 없음"

    lines: List[str] = []
    for i, action in enumerate(actions[:MAX_ACTIONS], 1):
        action_type = str(action.get("action", "unknown"))
        description = str(action.get("description", "")).strip()
        urgency = str(action.get("urgency", "medium"))

        urgency_icon = {
            "immediate": ":red_circle:",
            "urgent": ":red_circle:",
            "high": ":orange_circle:",
            "medium": ":yellow_circle:",
            "low": ":green_circle:",
        }.get(urgency, ":white_circle:")

        if description:
            text = _escape_mrkdwn_text(_truncate_for_slack(description, 180))
        else:
            text = _escape_mrkdwn_text(_translate_action_type(action_type))
        lines.append(f"{urgency_icon} {i}. {text}")

    return "\n".join(lines)


def _translate_action_type(action_type: str) -> str:
    """Action type을 한글로 번역"""
    action_map = {
        "issue_statement": "공식 입장문 발표",
        "amplify_viral": "긍정 바이럴 확산",
        "legal_response": "법적 대응 준비",
        "monitor_only": "모니터링 지속",
        "engage_influencers": "허브 계정 협력",
        "internal_review": "내부 조사 시작",
        "prepare_communication": "커뮤니케이션 준비",
    }
    return action_map.get(action_type, action_type)
