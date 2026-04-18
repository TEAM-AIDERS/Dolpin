# src/integrations/slack/formatter.py

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.dolpin_langgraph.state import AnalysisState

MAX_SECTION_TEXT = 2900
DEFAULT_ALLOWED_LINK_HOSTS = {"dashboard.example.com"}

_SOURCE_LABEL = {
    "twitter":       "Twitter(X)",
    "instiz":        "인스티즈",
    "google_trends": "Google 트렌드",
}
_TIME_WINDOW_LABEL = {
    "1h": "최근 1시간",
    "3h": "최근 3시간",
    "24h": "최근 24시간",
}
_SENTIMENT_LABEL = {
    "support": "응원/지지",
    "disappointment": "실망",
    "boycott": "보이콧",
    "meme": "밈/유머",
    "fanwar": "팬워",
    "neutral": "중립",
}
_SPIKE_NATURE_LABEL = {
    "positive": "긍정 이슈",
    "negative": "부정 이슈",
    "mixed": "혼재 이슈",
    "neutral": "중립 이슈",
}
_SPIKE_TYPE_LABEL = {
    "organic": "자연 발생",
    "media_driven": "미디어 주도",
    "coordinated": "조직적 확산",
}

# 상황별 커뮤니케이션 DO/DON'T
_COMM_GUIDE = {
    "crisis": {
        "do": ["감정 공감 우선", "명확하고 쉬운 표현", "빠른 초기 반응"],
        "dont": ["법적·방어적 표현", "책임 회피성 언급", "기계적 반복 답변"],
    },
    "opportunity": {
        "do": ["팬 참여 유도", "긍정 에너지 증폭", "공식 채널 활성화"],
        "dont": ["과도한 상업적 메시지", "팬 감정 무시", "타이밍 놓치기"],
    },
    "monitoring": {
        "do": ["상황 모니터링 지속", "팬 반응 수집", "준비 태세 유지"],
        "dont": ["과잉 대응", "불필요한 언급", "상황 악화 조장"],
    },
}


# ============================================================
# 메인 포맷 함수
# ============================================================

def format_to_slack(state: AnalysisState) -> Dict[str, Any]:
    """
    AnalysisState → Slack Block Kit payload 변환.
    데모 수준의 실무 레이아웃: 헤더 / 현재 상황 / 권장 전략 /
    커뮤니케이션 가이드 / 성명 초안 / 확산 분석 / 푸터
    """
    blocks: List[Dict[str, Any]] = []

    exec_brief   = state.get("executive_brief") or {}
    playbook     = state.get("playbook") or {}
    spike_event  = state.get("spike_event") or {}
    spike        = state.get("spike_analysis") or {}
    sentiment    = state.get("sentiment_result") or {}
    keyword       = str(spike_event.get("keyword", "unknown"))
    generated_at  = str(exec_brief.get("generated_at", ""))
    severity      = int(exec_brief.get("severity_score", 5))
    situation     = str(playbook.get("situation_type", "monitoring"))

    # ── 1. 헤더 ─────────────────────────────────────────────
    severity_icon = _get_severity_icon(severity)
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "emoji": True,
            "text": f"{severity_icon} DOLPIN 팬덤 이슈 대응 리포트",
        },
    })

    generated_display = _format_dt(generated_at)
    blocks.append({
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": f"*아티스트:* {_esc(keyword)}"},
            {"type": "mrkdwn", "text": f"*생성 시각:* {generated_display}"},
        ],
    })
    blocks.append({"type": "divider"})

    # ── 2. 현재 상황 요약 ────────────────────────────────────
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*:pushpin: 현재 상황 요약*"},
    })

    issue_type_text = _build_issue_type(spike, playbook)
    reaction_text   = _build_reaction(sentiment, exec_brief)
    blocks.append({
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": f"*이슈 유형*\n{issue_type_text}"},
            {"type": "mrkdwn", "text": f"*주요 반응*\n{reaction_text}"},
        ],
    })

    sources_text = _build_sources(spike_event)
    if sources_text:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f":round_pushpin: 분석 출처: {sources_text}"}],
        })

    # 상세 분석 결과 (spike_summary / sentiment_summary — nodes.py에서 풍부하게 생성)
    spike_summary = exec_brief.get("spike_summary")
    if spike_summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _trunc(str(spike_summary), MAX_SECTION_TEXT),
            },
        })

    sentiment_summary = exec_brief.get("sentiment_summary")
    if sentiment_summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _trunc(str(sentiment_summary), MAX_SECTION_TEXT),
            },
        })

    # 팬 대표 반응 메시지
    rep_messages_text = _build_representative_messages(sentiment)
    if rep_messages_text:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*:thought_balloon: 팬 반응 탐지*\n{rep_messages_text}",
            },
        })

    risk_line = _build_risk_line(severity, situation, spike)
    if risk_line:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": risk_line}],
        })

    blocks.append({"type": "divider"})

    # ── 3. 권장 대응 전략 ────────────────────────────────────
    actions = playbook.get("recommended_actions") or []
    if actions:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*:dart: 권장 대응 전략*"},
        })
        for action in actions[:3]:
            action_block = _format_action_block(action)
            if action_block:
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": action_block},
                })
        blocks.append({"type": "divider"})

    # ── 4. 팬 커뮤니케이션 가이드 ───────────────────────────
    do_text, dont_text = _build_comm_guide(situation)
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*:speech_balloon: 팬 커뮤니케이션 가이드*"},
    })
    blocks.append({
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": do_text},
            {"type": "mrkdwn", "text": dont_text},
        ],
    })
    blocks.append({"type": "divider"})

    # ── 5. 공식 성명 초안 ────────────────────────────────────
    draft = _extract_draft(actions)
    if draft:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*:memo: 공식 성명 초안*\n{_esc(_trunc(draft, 800))}",
            },
        })
        blocks.append({"type": "divider"})

    # ── 6. 확산 기회 (긍정 이슈일 때) ───────────────────────
    opportunity_summary = exec_brief.get("opportunity_summary")
    if opportunity_summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*:rocket: 확산 기회*\n{_trunc(str(opportunity_summary), MAX_SECTION_TEXT)}",
            },
        })

    # ── 8. 법적 검토 ─────────────────────────────────────────
    legal_summary = exec_brief.get("legal_summary")
    if legal_summary and legal_summary != "법률 검토 미수행":
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*:scales: 법적 검토*\n{_esc(str(legal_summary))}",
            },
        })

    # ── 9. 알림 (에러) ───────────────────────────────────────
    user_message = exec_brief.get("user_message")
    if user_message:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f":warning: {_esc(str(user_message))}"}],
        })

    # ── 10. 액션 버튼 ────────────────────────────────────────
    trace_id = str(state.get("trace_id", "unknown"))[:8]
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "emoji": True, "text": "✅ 확인 완료"},
                "style": "primary",
                "value": f"confirm_{trace_id}",
                "action_id": "action_confirm",
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "emoji": True, "text": "🔍 상세 분석 보기"},
                "value": f"detail_{trace_id}",
                "action_id": "action_detail",
            },
        ],
    })

    blocks.append({"type": "divider"})

    # ── 11. 푸터 ─────────────────────────────────────────────
    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": "본 리포트는 의사결정을 보조하기 위한 AI 기반 분석 결과이며, 최종 판단 및 책임은 담당자에게 있습니다.",
        }],
    })

    duration = float(exec_brief.get("analysis_duration_seconds", 0.0) or 0.0)
    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": (
                f":bulb: Powered by DOLPIN | "
                f"생성: {generated_display} | "
                f"분석 시간: {duration:.1f}초 | "
                f"Trace: `{trace_id}`"
            ),
        }],
    })

    return {
        "blocks": blocks,
        "text": f"DOLPIN 이슈 대응 리포트 | {keyword}",
    }


# ============================================================
# 섹션별 빌더
# ============================================================

def _build_issue_type(spike: Dict, playbook: Dict) -> str:
    """이슈 유형 필드 텍스트"""
    spike_rate   = spike.get("spike_rate", 0)
    spike_nature = spike.get("spike_nature", "neutral")
    spike_type   = spike.get("spike_type", "organic")

    nature_label = _SPIKE_NATURE_LABEL.get(spike_nature, spike_nature)
    type_label   = _SPIKE_TYPE_LABEL.get(spike_type, spike_type)

    return f"{spike_rate}배 급등 ({type_label})\n{nature_label}"


def _build_reaction(sentiment: Dict, exec_brief: Dict) -> str:
    """주요 반응 필드 텍스트 (dominant + 퍼센트)"""
    dominant = sentiment.get("dominant_sentiment", "")
    dist     = sentiment.get("sentiment_distribution", {})
    polarity = exec_brief.get("issue_polarity", "mixed")

    polarity_icon = {
        "positive": ":large_green_circle:",
        "negative": ":red_circle:",
        "mixed":    ":large_yellow_circle:",
    }.get(polarity, ":white_circle:")

    label = _SENTIMENT_LABEL.get(dominant, dominant) if dominant else "분석 중"
    pct   = dist.get(dominant, 0) * 100

    return f"{polarity_icon} {label} {pct:.0f}%"


def _build_sources(spike_event: Dict) -> str:
    """실제 messages.source 필드 기반 분석 출처 텍스트"""
    messages    = spike_event.get("messages") or []
    seen        = dict.fromkeys(
        msg.get("source", "") for msg in messages if msg.get("source")
    )
    source_labels = [_SOURCE_LABEL.get(s, s) for s in seen if s]

    if not source_labels:
        return ""

    time_window = spike_event.get("time_window", "")
    time_label  = _TIME_WINDOW_LABEL.get(time_window, time_window)

    sources_str = ", ".join(source_labels)
    return f"{sources_str} ({time_label})" if time_label else sources_str


def _build_risk_line(severity: int, situation: str, spike: Dict) -> str:
    """위험도 / 기회 한 줄 요약"""
    if situation == "crisis":
        if severity >= 8:
            return f":rotating_light: *위험도 {severity}/10* — 즉각 대응 필요"
        elif severity >= 6:
            return f":warning: *위험도 {severity}/10* — 빠른 확산 단계, 1차 공식 대응 권장"
        else:
            return f":warning: *위험도 {severity}/10* — 모니터링 강화 권장"
    elif situation == "opportunity":
        return f":rocket: *기회 점수 {severity}/10* — 긍정 바이럴 확산 중, 적극 활용 권장"
    else:
        return f":eyes: *모니터링 레벨 {severity}/10* — 지속 관찰 권장"


def _format_action_block(action: Dict[str, Any]) -> str:
    """액션 한 블록: 아이콘 + 이름(bold) + 설명"""
    urgency = str(action.get("urgency", "medium"))
    urgency_icon = {
        "immediate": "🔴",
        "urgent":    "🔴",
        "high":      "🟠",
        "medium":    "🟡",
        "low":       "🟢",
    }.get(urgency, "⚪")

    action_type  = str(action.get("action", "unknown"))
    description  = str(action.get("description", "")).strip()
    action_name  = _translate_action_type(action_type)

    lines = [f"{urgency_icon} *{_esc(action_name)}*"]
    if description and description not in ("대응 전략", ""):
        lines.append(_esc(_trunc(description, 400)))

    rationale = str(action.get("rationale", "")).strip()
    if rationale and len(rationale) > 10:
        lines.append(f"_→ {_esc(_trunc(rationale, 200))}_")

    return "\n".join(lines)


def _build_comm_guide(situation: str) -> Tuple[str, str]:
    """DO / DON'T 텍스트 튜플 반환 (Slack section fields 용)"""
    guide = _COMM_GUIDE.get(situation, _COMM_GUIDE["monitoring"])
    do_items   = "\n".join(f"• {item}" for item in guide["do"])
    dont_items = "\n".join(f"• {item}" for item in guide["dont"])
    return f"*DO*\n{do_items}", f"*DON'T*\n{dont_items}"


def _extract_draft(actions: List[Dict[str, Any]]) -> Optional[str]:
    """recommended_actions 중 draft 가 있는 첫 번째 항목 반환"""
    for action in actions:
        draft = action.get("draft")
        if draft and len(str(draft).strip()) > 20:
            return str(draft).strip()
    return None


def _build_representative_messages(sentiment: Dict) -> Optional[str]:
    """sentiment_result의 representative_messages에서 팬 반응 샘플 추출"""
    rep = sentiment.get("representative_messages") or {}
    if not rep:
        return None

    lines = []
    # dominant 감정 우선, 최대 3건
    dominant = sentiment.get("dominant_sentiment", "")
    order = [dominant] + [k for k in rep if k != dominant]

    count = 0
    for label in order:
        messages = rep.get(label) or []
        for msg in messages:
            text = str(msg).strip()
            if not text:
                continue
            lines.append(f"> {_esc(_trunc(text, 120))}")
            count += 1
            if count >= 3:
                break
        if count >= 3:
            break

    return "\n".join(lines) if lines else None



# ============================================================
# 공통 유틸
# ============================================================

def _esc(text: str) -> str:
    """Slack mrkdwn 특수문자 escape"""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for mention in ("@here", "@channel", "@everyone"):
        escaped = escaped.replace(mention, f"`{mention}`")
    return escaped


def _trunc(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_dt(iso_str: str) -> str:
    """ISO 8601 → 'YYYY-MM-DD HH:MM (KST)' 표시"""
    if not iso_str:
        return "알 수 없음"
    try:
        dt  = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        kst = dt + timedelta(hours=9)
        return kst.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(iso_str)[:16]


def _get_severity_icon(severity_score: int) -> str:
    if severity_score >= 8:
        return "🔴"
    if severity_score >= 6:
        return "🟠"
    if severity_score >= 4:
        return "🟡"
    return "🟢"


def _translate_action_type(action_type: str) -> str:
    action_map = {
        "issue_statement":       "공식 입장문 발표",
        "amplify_viral":         "긍정 바이럴 확산",
        "legal_response":        "법률팀 검토 및 대응",
        "monitor_only":          "팬 반응 모니터링",
        "engage_influencers":    "허브 계정 협력",
        "internal_review":       "내부 조사 및 재발 방지",
        "prepare_communication": "팬 소통 준비",
    }
    return action_map.get(action_type, action_type)


def _safe_link(url: Optional[str], label: str) -> str:
    """URL 검증 후 안전한 Slack 링크 생성 (allowlist 기반)"""
    if not url:
        return ""
    parsed = urlparse(str(url))
    if parsed.scheme != "https" or not parsed.netloc:
        return ""

    allowlist_raw = os.getenv("SLACK_LINK_ALLOWLIST", "")
    allowed_hosts = {h.strip().lower() for h in allowlist_raw.split(",") if h.strip()}
    if not allowed_hosts:
        allowed_hosts = set(DEFAULT_ALLOWED_LINK_HOSTS)

    if parsed.netloc.lower() not in allowed_hosts:
        return ""

    return f"<{url}|{_esc(label)}>"
