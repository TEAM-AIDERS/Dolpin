
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import random
import re
import html as _html
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="DOLPIN 분석 대시보드",
    page_icon="🐬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 커스텀 CSS
# ============================================================

st.markdown("""
<style>
    /* 전체 배경 */
    .main { background-color: #0e1117; }

    /* 심각도 배지 */
    .badge-full {
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-simple {
        background: linear-gradient(135deg, #4ECDC4, #44A1A0);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* 법적 리스크 경고 박스 */
    .legal-warning {
        background-color: rgba(255, 59, 59, 0.12);
        border-left: 4px solid #FF3B3B;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .legal-warning h4 { color: #FF6B6B; margin: 0 0 8px 0; }
    .legal-warning p { color: #FFAAAA; margin: 4px 0; font-size: 14px; }

    /* 액션 카드 */
    .action-card-immediate {
        border-left: 4px solid #FF3B3B;
        background-color: rgba(255, 59, 59, 0.08);
        padding: 10px 14px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .action-card-high {
        border-left: 4px solid #FF9500;
        background-color: rgba(255, 149, 0, 0.08);
        padding: 10px 14px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .action-card-medium {
        border-left: 4px solid #FFCC00;
        background-color: rgba(255, 204, 0, 0.08);
        padding: 10px 14px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .action-card-low {
        border-left: 4px solid #30D158;
        background-color: rgba(48, 209, 88, 0.08);
        padding: 10px 14px;
        border-radius: 4px;
        margin: 4px 0;
    }

    /* 인사이트 박스 */
    .insight-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #AAAAAA;
        margin-top: 8px;
    }
    .insight-box span.label {
        color: #7EB8F7;
        font-weight: 600;
    }

    /* 섹션 헤더 */
    .section-header {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* 구분선 */
    hr { border-color: rgba(255,255,255,0.08); }

    /* 스킵 배너 */
    .skip-banner {
        background-color: rgba(126, 184, 247, 0.08);
        border: 1px solid rgba(126, 184, 247, 0.3);
        border-radius: 10px;
        padding: 24px 28px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 헬퍼 함수들
# ============================================================

def strip_html(text: str) -> str:
    """HTML 태그를 제거하고 plain text만 반환한다."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = _html.unescape(text)
    return " ".join(text.split())


def get_urgency_label(urgency: str) -> str:
    labels = {
        "immediate": "🔴 즉시",
        "high": "🟠 높음",
        "medium": "🟡 보통",
        "low": "🟢 낮음",
    }
    return labels.get(urgency, urgency)


def get_action_label(action: str) -> str:
    labels = {
        "issue_statement": "📢 공식 성명 발표",
        "amplify_viral": "🚀 바이럴 확산",
        "legal_response": "⚖️ 법적 대응",
        "monitor_only": "👁️ 모니터링",
        "engage_influencers": "🤝 인플루언서 협업",
        "internal_review": "🔍 내부 검토",
        "prepare_communication": "✍️ 커뮤니케이션 준비",
    }
    return labels.get(action, action)


def build_volume_timeline(spike_event: dict) -> go.Figure:
    """언급량 추이 라인 차트 생성"""
    detected_at = datetime.fromisoformat(
        spike_event["detected_at"].replace("Z", "+00:00")
    ).replace(tzinfo=None)
    baseline = spike_event["baseline"]
    current_volume = spike_event["current_volume"]
    time_window = spike_event.get("time_window", "3h")
    hours = int(re.sub(r"\D", "", time_window) or "1")

    n_points = hours * 2 + 3
    times = [detected_at - timedelta(hours=hours) + timedelta(minutes=30 * i)
             for i in range(n_points)]

    random.seed(42)
    volumes = []
    spike_idx = n_points // 2
    for i in range(n_points):
        if i < spike_idx - 1:
            v = baseline + random.randint(-20, 30)
        elif i == spike_idx - 1:
            v = int(baseline * 1.8) + random.randint(0, 50)
        elif i == spike_idx:
            v = int(current_volume * 0.9) + random.randint(0, 100)
        elif i == spike_idx + 1:
            v = current_volume + random.randint(-50, 50)
        else:
            decay = (i - spike_idx - 1) * 0.15
            v = max(int(current_volume * (1 - decay)), baseline + 50)
            v += random.randint(-30, 30)
        volumes.append(max(v, 0))

    df = pd.DataFrame({"시간": times, "언급량": volumes})
    fig = go.Figure()

    fig.add_hline(
        y=baseline,
        line_dash="dot",
        line_color="rgba(100,100,100,0.5)",
        annotation_text=f"기준선 ({baseline})",
        annotation_font_color="#888",
        annotation_position="bottom right",
    )
    fig.add_trace(go.Scatter(
        x=df["시간"],
        y=df["언급량"],
        mode="lines+markers",
        name="언급량",
        line=dict(color="#7EB8F7", width=2.5),
        marker=dict(size=6, color="#7EB8F7"),
        fill="tozeroy",
        fillcolor="rgba(126,184,247,0.08)",
        hovertemplate="<b>%{x|%H:%M}</b><br>언급량: %{y:,}건<extra></extra>",
    ))

    peak_idx = volumes.index(max(volumes))
    fig.add_trace(go.Scatter(
        x=[df["시간"].iloc[peak_idx]],
        y=[df["언급량"].iloc[peak_idx]],
        mode="markers+text",
        name="피크",
        marker=dict(size=12, color="#FF6B6B", symbol="star"),
        text=[f"Peak<br>{df['언급량'].iloc[peak_idx]:,}"],
        textposition="top center",
        textfont=dict(color="#FF6B6B", size=11),
        hovertemplate="<b>피크</b><br>%{x|%H:%M}<br>언급량: %{y:,}건<extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis=dict(title=None, tickformat="%H:%M",
                   gridcolor="rgba(255,255,255,0.06)", showgrid=True),
        yaxis=dict(title="언급량 (건)",
                   gridcolor="rgba(255,255,255,0.06)", showgrid=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CCCCCC", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        height=280,
    )
    return fig


def build_sentiment_donut(dist: dict) -> go.Figure:
    """감정 분포 도넛 차트 생성"""
    label_map = {
        "support": "지지",
        "disappointment": "실망",
        "boycott": "불매/보이콧",
        "meme": "밈",
        "fanwar": "팬전쟁",
        "neutral": "중립",
    }
    color_map = {
        "support": "#30D158",
        "disappointment": "#FF9F0A",
        "boycott": "#FF453A",
        "meme": "#BF5AF2",
        "fanwar": "#FF6B6B",
        "neutral": "#636366",
    }

    labels = [label_map.get(k, k) for k in dist.keys()]
    values = [round(v * 100, 1) for v in dist.values()]
    colors = [color_map.get(k, "#888") for k in dist.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.60,
        marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
        textinfo="percent",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        sort=False,
    )])

    dominant_key = max(dist, key=lambda k: dist[k])
    dominant_pct = round(dist[dominant_key] * 100, 1)
    dominant_label = label_map.get(dominant_key, dominant_key)
    dominant_color = color_map.get(dominant_key, "#FFF")

    fig.add_annotation(
        text=f"<b>{dominant_label}</b><br><span style='font-size:20px;color:{dominant_color}'>{dominant_pct}%</span>",
        x=0.5, y=0.5,
        font=dict(size=13, color="white"),
        showarrow=False,
        xref="paper", yref="paper",
    )
    fig.update_layout(
        title=None,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5,
                    xanchor="left", x=1.02, font=dict(size=11, color="#CCCCCC")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=80, t=10, b=10),
        height=240,
    )
    return fig


def render_urgency_badge(urgency: str) -> str:
    colors = {
        "immediate": "#FF3B3B",
        "high": "#FF9500",
        "medium": "#FFCC00",
        "low": "#30D158",
    }
    c = colors.get(urgency, "#888")
    return (
        f'<span style="background:{c};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:11px;font-weight:700;">'
        f'{get_urgency_label(urgency)}</span>'
    )


_gcs_ready = bool(os.getenv("GCS_BUCKET_NAME"))


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:
    st.markdown("### 🐬 DOLPIN Dashboard")
    st.markdown("---")

    st.markdown("**파이프라인 상태**")
    if _gcs_ready:
        st.success("🟢 GCS 연결됨")
    else:
        st.info("🔌 GCS 미연결 (개발 모드)")

    st.markdown("---")
    if st.button("🔄 새로고침"):
        st.rerun()

    st.markdown("---")
    with st.expander("🧪 테스트", expanded=False):
        if st.button("🖼️ UI 미리보기", key="load_mock_ui"):
            _mock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_result.json")
            try:
                with open(_mock_path, "r", encoding="utf-8") as _f:
                    st.session_state["pipeline_result"] = json.load(_f)
                st.rerun()
            except FileNotFoundError:
                st.error("mock_result.json 파일이 없습니다.")
        st.caption("mock_result.json으로 UI 레이아웃 확인")
        if st.button("🗑️ 초기화", key="clear_mock"):
            st.session_state.pop("pipeline_result", None)
            st.rerun()

    st.markdown("---")
    st.markdown("**노드 인사이트**")
    show_insights = st.toggle("상세 인사이트 표시", value=True)

    st.markdown("---")
    st.caption("v1.0 · DOLPIN Multi-Agent")


# ============================================================
# 상태 로드
# ============================================================

# session_state 우선 (UI 미리보기), 없으면 GCS에서 로드
if "pipeline_result" not in st.session_state and _gcs_ready:
    from src.pipeline.result_store import load_result
    _gcs_result = load_result()
    if _gcs_result:
        st.session_state["pipeline_result"] = _gcs_result

state = st.session_state.get("pipeline_result")

# ── 결과 없음: 대기 화면 ──────────────────────────────────────
if state is None:
    st.markdown("### 🐬 DOLPIN 분석 대시보드")
    st.markdown("---")
    if _gcs_ready:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px;">
                <div style="font-size:64px;margin-bottom:16px;">🐬</div>
                <div style="font-size:22px;font-weight:700;color:#FFFFFF;margin-bottom:12px;">
                    분석 결과 대기 중
                </div>
                <div style="font-size:15px;color:#888;max-width:500px;margin:0 auto;line-height:1.7;">
                    파이프라인에서 스파이크 이벤트를 분석하면<br>
                    결과가 자동으로 표시됩니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px;">
                <div style="font-size:64px;margin-bottom:16px;">🔌</div>
                <div style="font-size:22px;font-weight:700;color:#FFFFFF;margin-bottom:12px;">
                    개발 모드 — GCS 미연결
                </div>
                <div style="font-size:15px;color:#888;max-width:500px;margin:0 auto;line-height:1.7;">
                    <code>.env</code>에 <code>GCS_BUCKET_NAME</code>을 설정하면<br>
                    파이프라인 결과를 불러옵니다.<br><br>
                    UI 확인은 사이드바 🧪 테스트 버튼을 사용하세요.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# ── (C) skipped=True: 실제 skip 결과를 UI에 표시 후 중단 ───────
if state.get("skipped"):
    spike_event_raw = state.get("spike_event") or {}
    spike_analysis_raw = state.get("spike_analysis") or {}
    skip_reason = state.get("skip_reason") or "not_significant"
    keyword = spike_event_raw.get("keyword", "—")
    spike_rate = spike_analysis_raw.get("spike_rate", spike_event_raw.get("spike_rate", 0.0))
    confidence = spike_analysis_raw.get("confidence", 0.0)
    is_significant = spike_analysis_raw.get("is_significant", False)
    partial_warning = spike_analysis_raw.get("partial_data_warning")
    node_insights_raw = state.get("node_insights") or {}

    st.markdown("### 🐬 DOLPIN 분석 대시보드")
    st.markdown("---")
    st.markdown(
        f"""
        <div class="skip-banner">
            <div style="font-size:40px;margin-bottom:12px;">📉</div>
            <div style="font-size:20px;font-weight:700;color:#7EB8F7;margin-bottom:8px;">
                스파이크가 유의미하지 않아 분석이 스킵되었습니다
            </div>
            <div style="font-size:14px;color:#888;margin-bottom:4px;">
                skip_reason: <code style="color:#AAA;">{skip_reason}</code>
                &nbsp;|&nbsp; is_significant: <code style="color:#AAA;">{is_significant}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("키워드", keyword)
    with sc2:
        st.metric("Spike Rate", f"{spike_rate}x")
    with sc3:
        st.metric("분석 신뢰도", f"{int(confidence * 100)}%")
    with sc4:
        msg_count = len(spike_event_raw.get("messages", []))
        st.metric("수집 메시지", f"{msg_count}건")

    if partial_warning:
        st.warning(f"데이터 경고: {partial_warning}")

    _errs = state.get("error_logs") or []
    if _errs:
        with st.expander(f"⚠️ 파이프라인 에러 {len(_errs)}건", expanded=False):
            for err in _errs:
                st.markdown(
                    f'<div class="insight-box">'
                    f'<span class="label">{err.get("stage","?")}</span>'
                    f' [{err.get("error_type","?")}] {err.get("message","")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if show_insights and node_insights_raw:
        with st.expander("Node Insights", expanded=True):
            for node_id, insight in node_insights_raw.items():
                st.markdown(
                    f'<div class="insight-box"><span class="label">{node_id}</span>'
                    f' &nbsp;{insight}</div>',
                    unsafe_allow_html=True,
                )
    st.stop()

# ── (D) 정상 결과: error_logs 배너 ───────────────────────────
_errs = state.get("error_logs") or []
if _errs:
    st.warning(f"⚠️ 파이프라인 에러 {len(_errs)}건 발생 — 파이프라인 상태 패널에서 확인하세요.")


# ============================================================
# 편의 변수
# ============================================================

spike_event = state["spike_event"]
spike_analysis = state.get("spike_analysis") or {}
sentiment_result = state.get("sentiment_result") or {}
causality_result = state.get("causality_result") or {}
legal_risk = state.get("legal_risk") or {
    "overall_risk_level": "low", "clearance_status": "clear",
    "confidence": 0.0, "rag_required": False, "rag_performed": False,
    "rag_confidence": None, "risk_assessment": None, "recommended_action": [],
    "referenced_documents": [], "signals": None,
}
amplification_summary = state.get("amplification_summary")
playbook = state.get("playbook") or {
    "situation_type": "monitoring", "priority": "low",
    "recommended_actions": [], "key_risks": [],
    "key_opportunities": [], "target_channels": [],
}
executive_brief = state.get("executive_brief") or {
    "summary": "분석 결과 없음", "severity_score": 0, "trend_direction": "stable",
    "issue_polarity": "neutral", "spike_summary": None, "sentiment_summary": None,
    "legal_summary": None, "action_summary": None, "opportunity_summary": None,
    "analysis_status": {}, "user_message": None,
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "analysis_duration_seconds": 0.0,
}
node_insights = state.get("node_insights") or {}
route2_decision = state.get("route2_decision", "sentiment_only")


# ============================================================
# ══ LAYER 1: 상황 인지 (Header) ══
# ============================================================

st.markdown('<p class="section-header">Layer 1 · 상황 인지</p>', unsafe_allow_html=True)

mode_badge = (
    '<span class="badge-full">⚡ 심층 분석 모드 (Full Analysis)</span>'
    if route2_decision == "full_analysis"
    else '<span class="badge-simple">🔍 단순 분석 모드 (Sentiment Only)</span>'
)

severity = executive_brief.get("severity_score", 0)
sev_color = "#FF3B3B" if severity >= 8 else "#FF9500" if severity >= 5 else "#30D158"

st.markdown(
    f"""
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:20px 24px;
                border:1px solid rgba(255,255,255,0.1);margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
            <div style="flex:1;min-width:300px;">
                <div style="font-size:11px;color:#888;margin-bottom:6px;letter-spacing:1px;">
                    EXECUTIVE BRIEF
                </div>
                <div style="font-size:17px;font-weight:700;color:#FFFFFF;line-height:1.4;">
                    {executive_brief['summary']}
                </div>
                <div style="margin-top:10px;font-size:13px;color:#AAA;">
                    🕐 분석 완료: {executive_brief['generated_at'][:19].replace('T',' ')} UTC &nbsp;|&nbsp;
                    ⏱️ {executive_brief['analysis_duration_seconds']}초 소요
                </div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
                {mode_badge}
                <div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:8px 16px;text-align:center;">
                    <div style="font-size:10px;color:#888;letter-spacing:1px;">SEVERITY</div>
                    <div style="font-size:28px;font-weight:800;color:{sev_color};line-height:1.1;">{severity}</div>
                    <div style="font-size:10px;color:#666;">/10</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    _spike_rate = spike_analysis.get("spike_rate", 0.0) or 0.0
    st.metric(
        label="📈 Spike Rate",
        value=f"{_spike_rate}x",
        delta=f"+{_spike_rate - 1:.1f}x vs baseline",
    )

with col2:
    act_pct = int((spike_analysis.get("actionability_score", 0.0) or 0.0) * 100)
    st.metric(
        label="🎯 Actionability",
        value=f"{act_pct}%",
        delta="즉각 대응 필요" if act_pct >= 80 else "모니터링 유지",
    )

with col3:
    conf_pct = int((spike_analysis.get("confidence", 0.0) or 0.0) * 100)
    st.metric(
        label="🔬 분석 신뢰도",
        value=f"{conf_pct}%",
        delta=f"messages={len(spike_event.get('messages', []))}건",
    )

with col4:
    _trend_map = {"escalating": "📈 악화", "stable": "➡️ 안정", "declining": "📉 개선"}
    trend_label = _trend_map.get(executive_brief.get("trend_direction", "stable"), "—")
    st.metric(
        label="🌡️ 트렌드",
        value=trend_label,
        delta=spike_analysis.get("spike_nature", "—"),
    )

if show_insights and node_insights.get("spike_analyzer"):
    st.markdown(
        f'<div class="insight-box"><span class="label">spike_analyzer</span>'
        f' &nbsp;{node_insights["spike_analyzer"]}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# ══ LAYER 2: 데이터 근거 (Body) ══
# ============================================================

st.markdown('<p class="section-header">Layer 2 · 데이터 근거</p>', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("**감정 분포 (Sentiment Distribution)**")
    dist = sentiment_result.get("sentiment_distribution") or {
        "support": 0.0, "disappointment": 0.0, "boycott": 0.0,
        "meme": 0.0, "fanwar": 0.0, "neutral": 1.0,
    }
    fig_donut = build_sentiment_donut(dist)
    st.plotly_chart(fig_donut, use_container_width=True, key="donut_chart")

    if show_insights and node_insights.get("sentiment"):
        st.markdown(
            f'<div class="insight-box"><span class="label">sentiment</span>'
            f' &nbsp;{node_insights["sentiment"]}</div>',
            unsafe_allow_html=True,
        )

    if sentiment_result.get("secondary_sentiment"):
        sec = sentiment_result["secondary_sentiment"]
        sec_msgs = sentiment_result.get("representative_messages", {}).get(sec, [])
        if sec_msgs:
            with st.expander(f"💬 대표 메시지 ({sec})", expanded=False):
                for msg in sec_msgs:
                    st.markdown(
                        f'<div style="padding:6px 0;color:#CCC;font-size:13px;">• {msg}</div>',
                        unsafe_allow_html=True,
                    )

with right_col:
    keyword = spike_event.get("keyword", "—")
    st.markdown(f"**언급량 추이 — '{keyword}'** `{spike_event.get('time_window', '3h')} window`")

    fig_line = build_volume_timeline(spike_event)
    st.plotly_chart(fig_line, use_container_width=True, key="line_chart")

    vol_cols = st.columns(3)
    with vol_cols[0]:
        st.metric("기준 언급량", f"{spike_event.get('baseline', 0):,}", help="baseline volume")
    with vol_cols[1]:
        _cur = spike_event.get("current_volume", 0)
        _base = spike_event.get("baseline", 0)
        st.metric("현재 언급량", f"{_cur:,}", delta=f"+{_cur - _base:,}")
    with vol_cols[2]:
        vi = spike_analysis.get("viral_indicators") or {}
        st.metric("크로스플랫폼", f"{len(vi.get('cross_platform', []))}개",
                  delta=vi.get("max_rise_rate", "—"))

if route2_decision == "full_analysis" and amplification_summary:
    st.markdown("---")
    st.markdown("**🌐 확산 허브 계정** *(full_analysis 경로)*")

    causality_hubs = causality_result.get("hub_accounts", [])
    if causality_hubs:
        detail_data = []
        for hub in causality_hubs:
            detail_data.append({
                "계정 ID": hub["account_id"],
                "타입": hub.get("account_type", "general"),
                "팔로워": f"{hub.get('follower_count', 0):,}",
                "영향력": hub["influence_score"],
            })
        df_hubs = pd.DataFrame(detail_data)
        st.dataframe(
            df_hubs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "영향력": st.column_config.ProgressColumn(
                    "영향력", help="0~1 스코어", format="%.2f",
                    min_value=0, max_value=1,
                ),
            },
        )
        if show_insights and node_insights.get("causality"):
            st.markdown(
                f'<div class="insight-box"><span class="label">causality</span>'
                f' &nbsp;{node_insights["causality"]}</div>',
                unsafe_allow_html=True,
            )
elif route2_decision == "full_analysis" and not amplification_summary:
    st.info("확산 허브 계정 데이터가 없습니다 (amplification 경로 미수행).")

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# ══ LAYER 3: 전략적 대응 (Bottom) ══
# ============================================================

st.markdown('<p class="section-header">Layer 3 · 전략적 대응</p>', unsafe_allow_html=True)

clearance = legal_risk.get("clearance_status", "clear")
if clearance != "clear":
    risk_level = legal_risk.get("overall_risk_level", "medium").upper()
    risk_label = "🔴 HIGH RISK" if clearance == "high_risk" else "🟠 REVIEW NEEDED"

    rag_conf = legal_risk.get("rag_confidence")
    rag_str = f" | RAG 신뢰도: {rag_conf*100:.0f}%" if rag_conf else ""
    conf_str = f"{legal_risk.get('confidence', 0.0) * 100:.0f}%"

    risk_assessment = legal_risk.get("risk_assessment")
    if risk_assessment:
        v_list = risk_assessment.get("legal_violation", [])
        violations = "".join([f"<p>⚠️ {_html.escape(v)}</p>" for v in v_list])
        raw_analysis = risk_assessment.get("analysis", "")
        safe_analysis = _html.escape(strip_html(raw_analysis))
        analysis_text = (
            f"<p style='margin-top:8px;color:#FFD0D0;'>{safe_analysis}</p>"
        )
    else:
        violations = ""
        analysis_text = ""

    rec_actions = "".join([f"<p>→ {a}</p>" for a in legal_risk.get("recommended_action", [])])
    docs = legal_risk.get("referenced_documents", [])
    doc_links = " · ".join([
        f'<a href="{d["link"]}" style="color:#FF9F9F;">{d["title"]}</a>'
        for d in docs
    ]) if docs else "—"

    _legal_html = (
        f'<div class="legal-warning">'
        f'<h4>⚖️ 법적 리스크 경고 &nbsp;{risk_label} &nbsp;'
        f'<span style="font-size:12px;color:#FFAAAA;">리스크 레벨: {risk_level} &middot; 신뢰도: {conf_str}{rag_str}</span>'
        f'</h4>'
        + violations
        + analysis_text
        + f'<div style="margin-top:12px;color:#FFCCCC;font-size:13px;font-weight:600;">권장 조치:</div>'
        f'<div style="color:#FFAAAA;font-size:13px;">{rec_actions}</div>'
        f'<div style="margin-top:10px;font-size:12px;color:#FF8888;">📄 참조 문서: {doc_links}</div>'
        f'</div>'
    )
    st.markdown(_legal_html, unsafe_allow_html=True)
    if show_insights and node_insights.get("LegalRAG"):
        st.markdown(
            f'<div class="insight-box"><span class="label">LegalRAG</span>'
            f' &nbsp;{node_insights["LegalRAG"]}</div>',
            unsafe_allow_html=True,
        )
else:
    st.success("✅ 법적 리스크 없음 (clearance: clear) — 별도 법률 대응 불필요")

st.markdown("<br>", unsafe_allow_html=True)

situation_map = {
    "crisis": "🚨 위기",
    "opportunity": "🌟 기회",
    "amplification": "🚀 확산",
    "monitoring": "👁️ 모니터링",
}
priority_map_label = {"urgent": "🔴 긴급", "high": "🟠 높음", "medium": "🟡 보통", "low": "🟢 낮음"}

sit_label = situation_map.get(playbook.get("situation_type", "monitoring"),
                               playbook.get("situation_type", "monitoring"))
pri_label = priority_map_label.get(playbook.get("priority", "low"),
                                    playbook.get("priority", "low"))

st.markdown(
    f"**대응 전략 (Playbook)** &nbsp; {sit_label} &nbsp; 우선순위: {pri_label}",
    unsafe_allow_html=True,
)

actions = playbook.get("recommended_actions", [])
for i, action in enumerate(actions):
    urgency = action.get("urgency", "low")
    action_label = get_action_label(action.get("action", ""))
    badge = render_urgency_badge(urgency)
    desc = action.get("description", "")
    draft = action.get("draft")
    legal_basis = action.get("legal_basis")

    with st.expander(f"{action_label} — {desc}", expanded=(i == 0)):
        st.markdown(f'<div style="margin-bottom:8px;">{badge}</div>', unsafe_allow_html=True)

        if draft:
            st.markdown("**📝 초안 텍스트:**")
            st.info(draft)

        if legal_basis:
            st.markdown(f"**⚖️ 법적 근거:** {legal_basis}")

        target_posts = action.get("target_posts")
        if target_posts:
            st.markdown("**🔗 대상 게시물:**")
            for tp in target_posts:
                st.markdown(f"- [{tp['source']} · {tp['source_message_id']}]({tp['url']})")

if show_insights and node_insights.get("playbook"):
    st.markdown(
        f'<div class="insight-box"><span class="label">playbook</span>'
        f' &nbsp;{node_insights["playbook"]}</div>',
        unsafe_allow_html=True,
    )

risk_opp_cols = st.columns(2)
with risk_opp_cols[0]:
    key_risks = playbook.get("key_risks", [])
    if key_risks:
        st.markdown("**⚠️ 핵심 리스크**")
        for r in key_risks:
            st.markdown(
                f'<div style="color:#FF9F9F;font-size:13px;padding:2px 0;">• {r}</div>',
                unsafe_allow_html=True,
            )

with risk_opp_cols[1]:
    key_opps = playbook.get("key_opportunities", [])
    if key_opps:
        st.markdown("**🌟 핵심 기회**")
        for o in key_opps:
            st.markdown(
                f'<div style="color:#9FFF9F;font-size:13px;padding:2px 0;">• {o}</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")
with st.expander("🔧 파이프라인 분석 상태", expanded=False):
    analysis_status = executive_brief.get("analysis_status", {})
    status_cols = st.columns(5)
    stage_names = {
        "spike_analyzer": "Spike Analyzer",
        "sentiment": "Sentiment",
        "causality": "Causality",
        "legal_rag": "Legal RAG",
        "playbook": "Playbook",
    }
    status_icons = {"success": "✅", "failed": "❌", "skipped": "⏭️", "partial": "⚠️"}

    for col, (k, v) in zip(status_cols, analysis_status.items()):
        icon = status_icons.get(v, "❓")
        with col:
            st.metric(label=stage_names.get(k, k), value=f"{icon} {v}")

    if _errs:
        st.markdown("**에러 로그**")
        for err in _errs:
            st.markdown(
                f'<div class="insight-box">'
                f'<span class="label">{err.get("stage","?")}</span>'
                f' [{err.get("error_type","?")}] {err.get("message","")}'
                f'</div>',
                unsafe_allow_html=True,
            )

    if show_insights:
        st.markdown("**Node Insights 전체**")
        for node_id, insight in node_insights.items():
            st.markdown(
                f'<div class="insight-box"><span class="label">{node_id}</span>'
                f' &nbsp;{insight}</div>',
                unsafe_allow_html=True,
            )
