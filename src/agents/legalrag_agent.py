import logging
from typing import List, Dict, Any, Optional, Literal
import json

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..dolpin_langgraph.state import LegalRiskResult, LegalRAGInput
from ..server.mcp_client import get_mcp_client

logger = logging.getLogger("legal_agent")

# -- 1. Pyndatic Schemas -- 

class RiskAssessmentSchema(BaseModel):
    risk_level: Literal["Critical", "High", "Medium", "Low"]
    legal_violation: List[str] = Field(description="관련 법률/조항/판례 키워드")
    analysis: str = Field(description="요약 분석")
    
# RAG 결과 
class ReferencedDocumentSchema(BaseModel):
    title: str
    link: str = Field(default="", description="문서 링크 (필수)")
    
# Quick Risk Check 
class SignalInfoSchema(BaseModel):
    risk_detected: bool  
    legal_keywords_detected: bool
    matched_keywords: List[str]
    reason: Literal["keyword_match", "pattern_match", "none"]
    
# LLM이 생성할 최종 출력 스키마
class LegalRiskOutputSchema(BaseModel):
    overall_risk_level: Literal["low", "medium", "high", "critical"]
    clearance_status: Literal["clear", "review_needed", "high_risk"]
    confidence: float
    rag_required: bool
    rag_performed: bool
    rag_confidence: Optional[float]
    risk_assessment: Optional[RiskAssessmentSchema] = None
    recommended_action: List[str] = Field(default_factory=list)
    referenced_documents: List[ReferencedDocumentSchema] = Field(default_factory=list)
    signals: Optional[SignalInfoSchema] = None
    

# -- 2. Agent 클래스 구현 -- 
LEGAL_KEYWORDS = [
    "고소", "소송", "법적", "재판",
    "표절", "저작권", "도용", "무단사용",
    "명예훼손", "허위사실", "딥페이크", "사생활", "유출", "폭로",
    "전속계약", "탈퇴", "역바이럴"
]

class LegalRAGAgent:
    def __init__(self, model_name="gpt-4o"):
        self.mcp = get_mcp_client()
        self.planner_llm = ChatOpenAI(model=model_name, temperature=0)
        self.synthesizer_llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(LegalRiskOutputSchema)
    # 메인 로직: Quick Check -> Agentic Loop -> Synthesis
    async def check_legal_risk(self, query_context: LegalRAGInput) -> LegalRiskResult:
        logger.info(f"⚖️ Legal Agent 분석 시작 (키워드: {query_context.get('keyword')})")
        # 1. Quick Risk Check 
        signals = self._quick_risk_check(query_context)
        
        rag_required = signals["risk_detected"]
        should_perform_rag = rag_required

        rag_context = []
        rag_performed = False
        retrieval_quality_factor = 1.0  # 검색 결과의 양적 신뢰도 가중치 
        
        # 2. Agentic RAG Loop (지능형 반복 검색)
        if should_perform_rag:
            rag_context = await self._perform_agentic_search(query_context, signals)
            rag_performed = bool(rag_context)

            # Retrieval Quality (문서 개수 가중치)
            doc_count = len(rag_context)
            if doc_count >= 3: retrieval_quality_factor = 1.0
            elif doc_count == 2: retrieval_quality_factor = 0.7
            elif doc_count == 1: retrieval_quality_factor = 0.4
            else: retrieval_quality_factor = 0.0

        # 3. Final Synthesis (결과 생성, 타입 검증)
        final_pydantic_result = await self._synthesize_result(
            query_context, signals, rag_context, rag_required, rag_performed, retrieval_quality_factor
        )
        
        return final_pydantic_result.model_dump()
    # 리스크 판단 
    def _quick_risk_check(self, context: LegalRAGInput) -> dict:
        messages_text = " ".join(context["messages"])
        
        # 1. 법적 키워드 탐지
        detected_keywords = [k for k in LEGAL_KEYWORDS if k in messages_text]
        keyword_risk = bool(detected_keywords)
        
        # 2. 컨텍스트 기반 위험 판단
        contextual_risk = False
        if context["spike_nature"] in ["negative", "mixed"]:
            contextual_risk = True
        if context["dominant_sentiment"] in ["boycott", "meme_negative"]:
            contextual_risk = True
        if context.get("fanwar_targets"):
            contextual_risk = True
        
        risk_detected = keyword_risk or contextual_risk
        
        return {
            "risk_detected": risk_detected, 
            "legal_keywords_detected": keyword_risk,
            "matched_keywords": detected_keywords,
            "reason": (
                "keyword_match" if keyword_risk 
                else "pattern_match" if contextual_risk 
                else "none"
            )
        }
    # Resource Prompt 기반 반복 검색
    async def _perform_agentic_search(self, context: LegalRAGInput, signals: dict) -> List[Dict]:
        max_iterations = 3
        collected_dict = {}
        # 검색 계획 수립 
        planner_prompt = ChatPromptTemplate.from_template("""
        당신은 '법률 조사 에이전트'입니다. 현재 상황에 가장 적합한 도구를 선택하여 검색하세요.
        (반복: {iteration}/{max_iterations})

        [입력 상황]
        - 키워드: {keyword}
        - spike 성격: {nature}
        - 주요 감정: {sentiment}
        - 탐지된 법적 신호: {signals}
        - 현재까지 확보된 정보 요약: {collected_summary}
        
        [사용 가능한 MCP 도구]
        1. search_statutes: 저작권법, 상표법 등 '성문법' 근거가 필요할 때 사용.
        2. search_precedents: 한국저작권위원회 등의 '실제 판례 및 상담 사례'가 필요할 때 사용.
        3. search_internal_policy: '사내 대응 지침(SM 매뉴얼)'이나 내부 가이드라인 확인 시 사용.

        [행동 규칙]
        - 정보가 충분하면 'STOP' 출력.
        - 검색이 필요하면 'CALL: [도구명] QUERY: [검색어]' 형식으로 출력.
          예: CALL: search_statutes QUERY: 저작권법 제10조 저작인격권
          예: CALL: search_precedents QUERY: 아이돌 안무 유사성 판례
        """)
        # 프롬프트 → LLM 실행 체인 
        planner_chain = planner_prompt | self.planner_llm

        for i in range(max_iterations):
            summary = str([item.get('content', '')[:50] for item in collected_dict.values()]) if collected_dict.values() else "없음"
            
            response = await planner_chain.ainvoke({
                "keyword": context.get('keyword', ''),
                "nature": context['spike_nature'],
                "sentiment": context['dominant_sentiment'],
                "signals": str(signals['matched_keywords']),
                "collected_summary": summary,
                "iteration": i+1,
                "max_iterations": max_iterations
            })
            
            decision = response.content.strip()
            if "STOP" in decision: break
            
            if "CALL:" in decision and "QUERY:" in decision:
                try:
                    # 명령어 파싱 
                    parts = decision.split("QUERY:")
                    tool_part = parts[0].replace("CALL:", "").strip()
                    query_text = parts[1].strip()
                    results = []
                    
                    # LLM이 선택한 도구에 따라 MCP 메서드 매핑
                    if tool_part == "search_statutes":
                        results = await self.mcp.legal_search_statutes(query=query_text)
                    elif tool_part == "search_precedents":
                        results = await self.mcp.legal_search_precedents(query=query_text)
                    elif tool_part == "search_internal_policy":
                        results = await self.mcp.legal_search_policies(query=query_text)
                    else:
                        logger.warning(f"알 수 없는 도구 요청: {tool_part}")
                        continue

                    if results:
                        # 중복 제거 로직: title이 같으면 덮어씌워 고유 문서만 유지
                        for r in results:
                            r['source_type'] = tool_part
                            doc_key = r.get('title', r.get('content', '')[:30])
                            collected_dict[doc_key] = r
                        
                except Exception as e:
                    logger.error(f"   -> MCP 호출 실패 ({decision}): {e}")
        
        return list(collected_dict.values())
    
    # 최종 리포트 생성 단계 
    # 검색 결과 → LLM 판단 → 시스템이 신뢰도 계산 → 최종 출력
    async def _synthesize_result(
        self, context: LegalRAGInput, signals: dict, rag_context: List[Dict],
        rag_required: bool, rag_performed: bool, retrieval_quality_factor: float
    ) -> LegalRiskOutputSchema:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            'LegalRiskResult' 리포트 작성 가이드:
            - overall_risk_level에 따른 clearance_status 강제 매핑:
              * critical/high -> high_risk
              * medium -> review_needed
              * low -> clear
            - rag_confidence: 검색된 법률 정보의 적합도(0.0~1.0) 평가
            """),
            ("human", "상황: {spike_nature}, 리스크신호: {signals}, 법률참조: {rag_context}")
        ])
        
        chain = prompt | self.synthesizer_llm
        
        result = await chain.ainvoke({
            "spike_nature": context["spike_nature"],
            "signals": str(signals),
            "rag_context": json.dumps(rag_context, ensure_ascii=False)
        })
        
        # 메타데이터 및 신뢰도 보정 (Hallucination 방지)
        result.rag_required = rag_required
        result.rag_performed = rag_performed
        result.signals = SignalInfoSchema(**signals)
        
        # RAG 미수행 시 
        if not rag_performed:
            result.rag_confidence = None
            result.confidence = 0.75 if context['spike_nature'] in ["negative", "mixed"] else 0.95
        # 최종 신뢰도 계산 (질적 점수 * 양적 가중치)
        else: 
            llm_qualitative_score = result.rag_confidence if result.rag_confidence is not None else 0.5
            result.confidence = round(llm_qualitative_score * retrieval_quality_factor, 2)
            result.rag_confidence = llm_qualitative_score # (rag_confidence는 LLM 점수 그대로 유지)
        return result
    
async def check_legal_risk(query_context: LegalRAGInput) -> LegalRiskResult:
    return await LegalRAGAgent().check_legal_risk(query_context)