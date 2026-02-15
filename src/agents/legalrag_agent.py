import logging
from typing import List, Dict, Any, Optional, Literal
import json

from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..dolpin_langgraph.state import LegalRiskResult, LegalRAGInput
from ..server.mcp_client import get_mcp_client

logger = logging.getLogger("legal_agent")

# -- 1. Pydantic Schemas with Validators -- 

class RiskAssessmentSchema(BaseModel):
    risk_level: Literal["critical", "high", "medium", "low"]
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
    
# LLM이 생성할 최종 출력 스키마 + 검증 로직
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
    
    # field_validator (개별 필드 검증)
    @field_validator('confidence')
    @classmethod
    # 신뢰도는 0.0 ~ 1.0 범위
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v
    
    @field_validator('rag_confidence')
    @classmethod
     # rag_confidence는 None 이거나 0.0 ~ 1.0 범위
    def validate_rag_confidence(cls, v):
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"rag_confidence must be between 0.0 and 1.0 or None, got {v}")
        return v
    
    # 필드 간 관계 검증: LLM이 매핑을 실수했을 경우 자동으로 보정
    @model_validator(mode='after')
    def validate_risk_level_mapping(self):
        mapping = {
            "critical": "high_risk",
            "high": "high_risk",
            "medium": "review_needed",
            "low": "clear"
        }
        
        expected_status = mapping.get(self.overall_risk_level)
        
        if self.clearance_status != expected_status:
            logger.warning(
                f"❌ Risk-Status 매핑 불일치 감지!\n"
                f"   overall_risk_level: {self.overall_risk_level}\n"
                f"   Expected: {expected_status}, Got: {self.clearance_status}\n"
                f"   → 자동 보정 적용"
            )
            self.clearance_status = expected_status
        
        return self
    # rag_performed와 rag_confidence의 일관성 검증
    @model_validator(mode='after')
    def validate_rag_confidence_consistency(self):
        if not self.rag_performed and self.rag_confidence is not None:
            logger.warning(
                f"⚠️  RAG 미수행인데 rag_confidence가 설정됨\n"
                f"   rag_performed: {self.rag_performed}\n"
                f"   rag_confidence: {self.rag_confidence}\n"
                f"   → rag_confidence를 None으로 설정"
            )
            self.rag_confidence = None
        
        return self


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
        self.synthesizer_llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(
            LegalRiskOutputSchema,
            method="json_schema"
        )
    
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
            if doc_count >= 3:
                retrieval_quality_factor = 1.0
            elif doc_count == 2:
                retrieval_quality_factor = 0.7
            elif doc_count == 1:
                retrieval_quality_factor = 0.4
            else:
                retrieval_quality_factor = 0.0

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
        collected_info = []
        # 검색 계획 수립 
        planner_prompt = ChatPromptTemplate.from_template("""
        당신은 '법률 조사 에이전트'입니다. 현재 상황에 가장 적합한 도구를 선택하여 검색하세요.
        (반복: {iteration}/{max_iterations})

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
            summary = str([item.get('content', '')[:50] for item in collected_info]) if collected_info else "없음"
            
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
            if "STOP" in decision:
                break
            
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
                        # 출처 정보 추가 
                        for r in results:
                            r['source_type'] = tool_part
                        collected_info.extend(results)  # Agent의 작업 메모리
                        
                except Exception as e:
                    logger.error(f"   -> MCP 호출 실패 ({decision}): {e}")
        
        return collected_info
    
    # 최종 리포트 생성 단계 
    async def _synthesize_result(
        self, context: LegalRAGInput, signals: dict, rag_context: List[Dict],
        rag_required: bool, rag_performed: bool, retrieval_quality_factor: float
    ) -> LegalRiskOutputSchema:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            'LegalRiskResult' 리포트 작성 가이드:
            
            [Risk Level 형식 (소문자 필수)]
            - overall_risk_level: "low", "medium", "high", "critical" (소문자)
            - risk_assessment.risk_level: "low", "medium", "high", "critical" (소문자)
            
            [Risk-Status 매핑 규칙]
            다음 규칙을 반드시 따르세요:
            - critical 또는 high → clearance_status는 "high_risk"
            - medium → clearance_status는 "review_needed"
            - low → clearance_status는 "clear"
            
            [신뢰도 범위]
            - confidence: 0.0 ~ 1.0 (필수)
            - rag_confidence: 0.0 ~ 1.0 또는 null (검색 미수행 시 null)
            
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
        
        # 메타데이터 설정
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
            result.rag_confidence = llm_qualitative_score
        
        return result


# -- 3. 모듈 레벨 래퍼 함수 --
async def check_legal_risk(query_context: LegalRAGInput) -> LegalRiskResult:
    agent = LegalRAGAgent()
    return await agent.check_legal_risk(query_context)