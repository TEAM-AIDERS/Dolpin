"""
LangGraph State TypedDict 정의
DOLPIN 워크플로우의 전체 데이터 흐름을 정의합니다.

버전: v1.1 (260114)
참조: [SPEC] JSON Schema v1.1
"""

from typing import TypedDict, List, Dict, Any, Literal, Optional
from datetime import datetime


# ============================================================
# 공통 타입 정의
# ============================================================

Source = Literal["twitter", "theqoo", "instiz", "google_trends"]
SpikeNature = Literal["positive", "negative", "mixed", "neutral"]
DataCompleteness = Literal["confirmed", "partial", "mixed"]


# ============================================================
# 1. KafkaMessage (입력)
# ============================================================

class ContentData(TypedDict, total=False):
    """Post 타입일 때의 컨텐츠 데이터"""
    text: str
    author_id: str
    metrics: Dict[str, int]  # {"likes": 100, "retweets": 50, ...}


class TrendData(TypedDict, total=False):
    """Trend 타입일 때의 트렌드 데이터"""
    interest_score: int  # 0~100
    is_partial: bool
    rising_queries: List[Dict[str, str]]  # [{"query": "...", "value": "+500%"}, ...]
    top_queries: List[str]
    region_stats: List[Dict[str, Any]]  # [{"geo": "KR", "value": 100}, ...]


class KafkaMessage(TypedDict):
    """Kafka에서 들어오는 원본 메시지"""
    message_id: str  # UUID
    type: Literal["post", "trend"]
    source: Source
    collected_at: str  # ISO 8601
    keyword: str
    content_data: ContentData  # type="post"일 경우
    trend_data: TrendData      # type="trend"일 경우


# ============================================================
# 2. SpikeEvent (SpikeAnalyzer 입력)
# ============================================================

class Message(TypedDict):
    """통합된 메시지 형식"""
    id: str  # UUID
    source_message_id: str  # 원본 ID (트윗 ID, 게시글 번호)
    text: str
    timestamp: str  # ISO 8601
    source: Source
    author_id: str
    metrics: Dict[str, int]
    is_anonymized: bool
    detected_language: str


class SpikeEvent(TypedDict):
    """급등 이벤트 (전처리된 통합 메시지)"""
    keyword: str
    spike_rate: float
    baseline: int
    current_volume: int
    detected_at: str  # ISO 8601
    time_window: Literal["1h", "3h", "24h"]
    messages: List[Message]
    raw_kafka_message_ids: List[str]


# ============================================================
# 3. SpikeAnalysisResult (SpikeAnalyzer 출력)
# ============================================================

class ViralIndicators(TypedDict):
    """바이럴 지표"""
    is_trending: bool
    has_breakout: bool
    max_rise_rate: str  # "+500%" 또는 "Breakout"
    breakout_queries: List[str]
    cross_platform: List[Source]
    international_reach: float  # 0.0 ~ 1.0


class SpikeAnalysisResult(TypedDict):
    """급등 분석 결과"""
    is_significant: bool
    spike_rate: float
    spike_type: Literal["organic", "media_driven", "coordinated"]
    spike_nature: SpikeNature
    peak_timestamp: str  # ISO 8601
    duration_minutes: int
    confidence: float  # 0.0 ~ 1.0
    actionability_score: float  # 0.0 ~ 1.0
    data_completeness: DataCompleteness
    partial_data_warning: Optional[str]
    viral_indicators: ViralIndicators


# ============================================================
# 4. SentimentAnalysisResult (SentimentAgent 출력)
# ============================================================

class SentimentDistribution(TypedDict):
    """감정 분포 - SentimentAgent에서 반환하는 6개 라벨"""
    support: float
    disappointment: float
    boycott: float
    meme: float
    fanwar: float
    neutral: float


class LexiconMatch(TypedDict):
    """렉시콘 매칭 정보"""
    count: int
    type: str  # "agreement_slang", "meme_positive", ...
    terms: List[str] 


class SentimentAnalysisResult(TypedDict):
    """감정 분석 결과"""
    sentiment_distribution: SentimentDistribution
    dominant_sentiment: str
    secondary_sentiment: Optional[str]
    has_mixed_sentiment: bool
    sentiment_shift: Optional[Literal["worsening", "improving", "stable"]]
    representative_messages: Dict[str, List[str]]
    meme_keywords: Optional[List[str]]
    fanwar_targets: Optional[List[str]]
    lexicon_matches: Optional[Dict[str, LexiconMatch]]
    analyzed_count: int
    confidence: float  # 0.0 ~ 1.0


# ============================================================
# 5. CausalityAnalysisResult (CausalityAgent 출력)
# ============================================================

class HubAccount(TypedDict):
    """네트워크 허브 계정"""
    account_id: str
    influence_score: float  # 0.0 ~ 1.0
    follower_count: int
    account_type: Literal["influencer", "fan_account", "media", "general"]


class RetweetNetworkMetrics(TypedDict):
    """리트윗 네트워크 지표"""
    centralization: float  # 0.0 ~ 1.0
    avg_degree: float


class CausalityAnalysisResult(TypedDict):
    """인과관계 분석 결과"""
    trigger_source: Literal["influencer", "media", "organic", "unknown"]
    hub_accounts: List[HubAccount]
    retweet_network_metrics: RetweetNetworkMetrics
    cascade_pattern: Literal["viral", "echo_chamber", "coordinated"]
    estimated_origin_time: Optional[str]  # ISO 8601
    key_propagation_paths: List[str]


# ============================================================
# 6. LegalRiskResult (LegalRAGAgent 출력)
# ============================================================

class RiskAssessment(TypedDict):
    """RAG 수행 시 리스크 평가"""
    risk_level: Literal["Critical", "High", "Medium", "Low"]
    legal_violation: List[str]
    analysis: str


class ReferencedDocument(TypedDict):
    """참조 문서"""
    title: str
    link: str


class LegalSignals(TypedDict):
    """경량 체크 신호 (디버깅용)"""
    legal_keywords_detected: bool
    matched_keywords: List[str]
    reason: Literal["keyword_match", "pattern_match", "none"]


class LegalRiskResult(TypedDict):
    """법률 리스크 검토 결과"""
    overall_risk_level: Literal["low", "medium", "high", "critical"]
    clearance_status: Literal["clear", "review_needed", "high_risk"]
    confidence: float  # 0.0 ~ 1.0
    rag_required: bool
    rag_performed: bool
    rag_confidence: Optional[float]  # 0.0 ~ 1.0 (rag_performed=True일 때만)
    risk_assessment: Optional[RiskAssessment]
    recommended_action: List[str]
    referenced_documents: List[ReferencedDocument]
    signals: Optional[LegalSignals]


class LegalRAGInput(TypedDict):
    """Legal RAG Agent 입력 파라미터"""
    messages: List[str]  # spike_event["messages"][].text
    spike_nature: str  # "positive" | "negative" | "mixed" | "neutral"
    dominant_sentiment: str  # "support" | "boycott" | "meme_negative" 등
    keyword: str  # spike_event["keyword"]
    spike_rate: float
    fanwar_targets: Optional[List[str]]


# ============================================================
# 7. PlaybookResult (PlaybookAgent 출력)
# ============================================================

class TargetPost(TypedDict):
    """타겟 게시물"""
    id: str
    source: Source
    source_message_id: str
    url: str


class RecommendedAction(TypedDict):
    """권장 조치"""
    action: Literal[
        "issue_statement",
        "amplify_viral",
        "legal_response",
        "monitor_only",
        "engage_influencers",
        "internal_review",
        "prepare_communication"
    ]
    urgency: Literal["immediate", "high", "medium", "low"]
    description: str
    draft: Optional[str]  # issue_statement, legal_response일 때
    target_posts: Optional[List[TargetPost]]  # amplify_viral일 때
    legal_basis: Optional[str]  # legal_response일 때


class PlaybookResult(TypedDict):
    """대응 전략"""
    situation_type: Literal["crisis", "opportunity", "monitoring", "amplification"]
    priority: Literal["urgent", "high", "medium", "low"]
    recommended_actions: List[RecommendedAction]
    key_risks: List[str]
    key_opportunities: List[str]
    target_channels: List[str]


# ============================================================
# 8. ExecBrief (최종 출력)
# ============================================================

class AnalysisStatus(TypedDict):
    """분석 상태 (에러 처리용)"""
    spike_analyzer: Literal["success", "failed", "skipped"]
    sentiment: Literal["success", "failed", "skipped"]
    causality: Literal["success", "failed", "skipped"]
    legal_rag: Literal["success", "failed", "skipped"]
    playbook: Literal["success", "failed", "partial"]


class ExecBrief(TypedDict):
    """임원 브리핑 (최종 출력)"""
    summary: str
    severity_score: int  # 1-10
    trend_direction: Literal["escalating", "stable", "declining"]
    issue_polarity: Literal["positive", "negative", "mixed"]
    spike_summary: Optional[str]
    sentiment_summary: Optional[str]
    legal_summary: Optional[str]
    action_summary: Optional[str]
    opportunity_summary: Optional[str]  # 긍정 이슈만
    analysis_status: AnalysisStatus
    user_message: Optional[str]  # 에러 발생 시
    generated_at: str  # ISO 8601
    analysis_duration_seconds: float


# ============================================================
# 9. ErrorLog
# ============================================================

class ErrorLog(TypedDict):
    """에러 로그 (디버깅용)"""
    stage: Literal["spike_analyzer", "router1", "lexicon_lookup", "sentiment", "router2", "causality", "router3", "legal_rag", "amplification", "playbook", "exec_brief"]
    error_type: Literal["timeout", "schema_error", "exception", "api_error"]
    message: str
    occurred_at: str  # ISO 8601
    trace_id: str
    details: Optional[Dict[str, Any]]


# ============================================================
# 10. Amplification Summary
# ============================================================

class AmplificationHubAccount(TypedDict):
    """확산 기회의 허브 계정"""
    account_id: str
    influence_score: float


class AmplificationMessage(TypedDict):
    """대표 긍정 메시지"""
    text: str
    engagement: int


class AmplificationSummary(TypedDict):
    """긍정 바이럴 기회 요약"""
    top_platforms: List[Source]
    hub_accounts: List[AmplificationHubAccount]
    representative_messages: List[AmplificationMessage]


# ============================================================
# 11. AnalysisState (LangGraph 전체 상태)
# ============================================================

class AnalysisState(TypedDict):
    """LangGraph 워크플로우 전체 상태"""
    
    # 입력
    spike_event: SpikeEvent
    
    # 메타데이터
    trace_id: str
    workflow_start_time: str  # ISO 8601

    "sentiment_model_path": "Aerisbin/sentiment-agent-v1"
    
    # Router 결정
    route1_decision: Optional[Literal["skip", "analyze"]]
    route2_decision: Optional[Literal["sentiment_only", "full_analysis"]]
    route3_decision: Optional[Literal["legal", "amplification"]]
    positive_viral_detected: Optional[bool]
    
    # 각 에이전트/노드 결과
    spike_analysis: Optional[SpikeAnalysisResult]
    lexicon_matches: Optional[Dict[str, LexiconMatch]]
    sentiment_result: Optional[SentimentAnalysisResult]
    causality_result: Optional[CausalityAnalysisResult]
    legal_risk: Optional[LegalRiskResult]
    amplification_summary: Optional[AmplificationSummary]
    playbook: Optional[PlaybookResult]
    
    # 각 노드의 핵심 요약
    node_insights: Dict[str, str]
    
    # 최종 출력
    executive_brief: Optional[ExecBrief]
    
    # 에러 처리
    error_logs: List[ErrorLog]
    
    # Skip 처리
    skipped: bool
    skip_reason: Optional[Literal["not_significant"]]
