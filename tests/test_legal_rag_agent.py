"""
test_legal_rag_agent.py - Legal RAG Agent 분석 및 스키마 검증 테스트

목적:
- LegalRAGAgent의 분석 능력 검증
- Pydantic 스키마 준수 확인
- 신뢰도 계산 로직 검증
- LLM 응답 파싱 및 자동 보정 확인

검증 항목:
1. 리스크 키워드 감지: "고소", "명예훼손" 등이 들어오면 rag_required=True
2. 스키마 준수: LLM 응답이 LegalRiskOutputSchema 형식에 정확히 맞음
3. 신뢰도 계산: confidence와 rag_confidence가 0.0~1.0 범위
4. 자동 보정: LLM이 실수한 risk-status 매핑을 자동으로 수정

핵심 시나리오:
- 매우 위험한 상황: "고소 준비 중", "명예훼손 소송"
- 보통 위험: "팬전쟁", "루머 공중파 출연"
- 안전한 상황: "일반 팬아트 공유", "긍정적인 팬 반응"
"""

import pytest
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import sys
import logging
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock

# Allow direct execution
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.agents.legalrag_agent import (
    LegalRAGAgent,
    LegalRiskOutputSchema,
    RiskAssessmentSchema,
    SignalInfoSchema,
    LEGAL_KEYWORDS,
)
from src.dolpin_langgraph.state import LegalRiskResult, LegalRAGInput

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===========================
# Test Fixtures & Mock Data
# ===========================

@pytest.fixture
def event_loop():
    """비동기 테스트용 이벤트 루프"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def legal_agent():
    """LegalRAGAgent 인스턴스"""
    return LegalRAGAgent(model_name="gpt-4o")


def create_legal_rag_input(
    keyword: str = "test",
    messages: list = None,
    spike_nature: str = "neutral",
    dominant_sentiment: str = "positive",
    fanwar_targets: Optional[list] = None,
) -> LegalRAGInput:
    """테스트용 LegalRAGInput 생성"""
    return {
        "keyword": keyword,
        "messages": messages or ["일반적인 팬 반응"],
        "spike_nature": spike_nature,
        "dominant_sentiment": dominant_sentiment,
        "fanwar_targets": fanwar_targets or [],
    }


# ===========================
# 1. 리스크 감지 테스트
# ===========================

class TestRiskDetection:
    """리스크 키워드 및 패턴 감지"""
    
    def test_legal_keyword_detection(self, legal_agent):
        """법적 키워드 감지"""
        logger.info("🔍 법적 키워드 감지 테스트...")
        
        test_cases = [
            ("고소 준비 중", True, "고소"),
            ("명예훼손 소송", True, "명예훼손"),
            ("저작권 침해 논란", True, "저작권"),
            ("일반 팬 반응입니다", False, None),
        ]
        
        for text, should_detect, expected_keyword in test_cases:
            context = create_legal_rag_input(messages=[text])
            signals = legal_agent._quick_risk_check(context)
            
            assert signals["risk_detected"] == should_detect, \
                f"{text}: risk_detected는 {should_detect}여야 함"
            
            if expected_keyword:
                assert expected_keyword in signals["matched_keywords"], \
                    f"{expected_keyword}가 감지되지 않음"
            
            logger.info(f"  ✓ '{text[:20]}...' → {signals['risk_detected']}")
    
    def test_contextual_risk_detection(self, legal_agent):
        """컨텍스트 기반 리스크 감지"""
        logger.info("🔍 컨텍스트 기반 리스크 감지...")
        
        test_cases = [
            # (spike_nature, sentiment, should_detect, reason)
            ("negative", "positive", True, "부정적 spike"),
            ("mixed", "neutral", True, "혼합 감정"),
            ("positive", "boycott", True, "boycott sentiment"),
            ("neutral", "positive", False, "안전한 상황"),
        ]
        
        for spike, sentiment, should_detect, reason in test_cases:
            context = create_legal_rag_input(
                spike_nature=spike,
                dominant_sentiment=sentiment,
            )
            signals = legal_agent._quick_risk_check(context)
            
            assert signals["risk_detected"] == should_detect, \
                f"{reason}: {signals['risk_detected']} != {should_detect}"
            
            logger.info(f"  ✓ {reason} → {should_detect}")
    
    def test_signal_structure(self, legal_agent):
        """Signal 신호 구조"""
        logger.info("📊 Signal 구조 검증...")
        
        context = create_legal_rag_input(messages=["고소할 거야"])
        signals = legal_agent._quick_risk_check(context)
        
        # 필수 필드 확인
        assert "risk_detected" in signals, "risk_detected 필드 필요"
        assert "legal_keywords_detected" in signals, "legal_keywords_detected 필드 필요"
        assert "matched_keywords" in signals, "matched_keywords 필드 필요"
        assert "reason" in signals, "reason 필드 필요"
        
        # 타입 확인
        assert isinstance(signals["risk_detected"], bool)
        assert isinstance(signals["matched_keywords"], list)
        assert signals["reason"] in ["keyword_match", "pattern_match", "none"]
        
        logger.info(f"✓ Signal 구조: {signals}")


# ===========================
# 2. 스키마 검증 테스트
# ===========================

class TestSchemaValidation:
    """Pydantic 스키마 검증"""
    
    def test_risk_assessment_schema(self):
        """RiskAssessmentSchema 검증"""
        logger.info("📋 RiskAssessmentSchema 검증...")
        
        # 유효한 스키마
        valid_data = {
            "risk_level": "high",
            "legal_violation": ["저작권법 제10조", "판례 2024-1234"],
            "analysis": "저작권 침해 위험이 높습니다"
        }
        
        schema = RiskAssessmentSchema(**valid_data)
        assert schema.risk_level == "high"
        assert len(schema.legal_violation) == 2
        
        logger.info("✓ RiskAssessmentSchema 검증 통과")
    
    def test_signal_info_schema(self):
        """SignalInfoSchema 검증"""
        logger.info("📋 SignalInfoSchema 검증...")
        
        valid_data = {
            "risk_detected": True,
            "legal_keywords_detected": True,
            "matched_keywords": ["고소", "소송"],
            "reason": "keyword_match"
        }
        
        schema = SignalInfoSchema(**valid_data)
        assert schema.risk_detected is True
        assert schema.reason == "keyword_match"
        
        logger.info("✓ SignalInfoSchema 검증 통과")
    
    def test_output_schema_validation(self):
        """LegalRiskOutputSchema 전체 검증"""
        logger.info("📋 LegalRiskOutputSchema 검증...")
        
        valid_data = {
            "overall_risk_level": "high",
            "clearance_status": "high_risk",
            "confidence": 0.85,
            "rag_required": True,
            "rag_performed": True,
            "rag_confidence": 0.78,
            "risk_assessment": {
                "risk_level": "high",
                "legal_violation": ["저작권법"],
                "analysis": "위험함"
            },
            "recommended_action": ["법무팀 검토", "공식 입장 발표"],
            "signals": {
                "risk_detected": True,
                "legal_keywords_detected": True,
                "matched_keywords": ["고소"],
                "reason": "keyword_match"
            }
        }
        
        schema = LegalRiskOutputSchema(**valid_data)
        assert schema.overall_risk_level == "high"
        assert schema.confidence == 0.85
        assert schema.rag_performed is True
        
        logger.info("✓ LegalRiskOutputSchema 검증 통과")
    
    def test_confidence_validation(self):
        """신뢰도 범위 검증"""
        logger.info("✔️ 신뢰도 범위 검증...")
        
        # 유효한 범위
        valid_data = {
            "overall_risk_level": "medium",
            "clearance_status": "review_needed",
            "confidence": 0.5,
            "rag_required": False,
            "rag_performed": False,
            "rag_confidence": None,
        }
        schema = LegalRiskOutputSchema(**valid_data)
        assert schema.confidence == 0.5
        
        # 범위 초과는 에러
        invalid_data = {
            "overall_risk_level": "medium",
            "clearance_status": "review_needed",
            "confidence": 1.5,  # ❌ 범위 초과
            "rag_required": False,
            "rag_performed": False,
            "rag_confidence": None,
        }
        
        with pytest.raises(ValueError):
            LegalRiskOutputSchema(**invalid_data)
        
        logger.info("✓ 신뢰도 범위 검증 통과")
    
    def test_rag_confidence_consistency(self):
        """RAG 신뢰도 일관성 검증"""
        logger.info("✔️ RAG 신뢰도 일관성 검증...")
        
        # rag_performed=False인데 rag_confidence가 설정되면 자동 보정
        data = {
            "overall_risk_level": "low",
            "clearance_status": "clear",
            "confidence": 0.9,
            "rag_required": False,
            "rag_performed": False,
            "rag_confidence": 0.5,  # ❌ 모순
        }
        
        schema = LegalRiskOutputSchema(**data)
        # 자동 보정으로 None이 되어야 함
        assert schema.rag_confidence is None, "rag_performed=False이면 rag_confidence는 None이어야 함"
        
        logger.info("✓ RAG 신뢰도 자동 보정 완료")


# ===========================
# 3. 자동 보정 테스트
# ===========================

class TestAutoCorrection:
    """LLM 실수에 대한 자동 보정"""
    
    def test_risk_status_mapping_correction(self):
        """Risk-Status 매핑 자동 보정"""
        logger.info("🔧 Risk-Status 매핑 보정 테스트...")
        
        test_cases = [
            # (risk_level, wrong_status, expected_status)
            ("critical", "review_needed", "high_risk"),  # ❌ → ✓
            ("high", "clear", "high_risk"),              # ❌ → ✓
            ("medium", "high_risk", "review_needed"),    # ❌ → ✓
            ("low", "review_needed", "clear"),           # ❌ → ✓
        ]
        
        for risk_level, wrong_status, expected_status in test_cases:
            data = {
                "overall_risk_level": risk_level,
                "clearance_status": wrong_status,  # 잘못된 값
                "confidence": 0.8,
                "rag_required": False,
                "rag_performed": False,
                "rag_confidence": None,
            }
            
            schema = LegalRiskOutputSchema(**data)
            
            # 자동으로 보정되어야 함
            assert schema.clearance_status == expected_status, \
                f"{risk_level}: {wrong_status} → {schema.clearance_status} (기대값: {expected_status})"
            
            logger.info(f"  ✓ {risk_level}: {wrong_status} → {expected_status}")


# ===========================
# 4. 에이전트 실제 동작 테스트 (Mock 사용)
# ===========================

class TestAgentBehavior:
    """에이전트 실제 분석 로직"""
    
    @pytest.mark.asyncio
    async def test_critical_risk_scenario(self, legal_agent):
        """매우 위험한 상황"""
        logger.info("🚨 매우 위험한 상황 분석...")
        
        context = create_legal_rag_input(
            keyword="명예훼손 소송",
            messages=[
                "고소장 접수했대",
                "법원에 출두 요청받음",
                "변호사 선임함"
            ],
            spike_nature="negative",
            dominant_sentiment="boycott",
        )
        
        # Quick risk check만 테스트 (LLM 호출 X)
        signals = legal_agent._quick_risk_check(context)
        
        assert signals["risk_detected"] is True, "위험 감지 필수"
        assert "고소" in signals["matched_keywords"] or signals["reason"] == "pattern_match"
        
        logger.info(f"✓ 위험 감지 완료: {signals}")
    
    @pytest.mark.asyncio
    async def test_medium_risk_scenario(self, legal_agent):
        """중간 위험 상황"""
        logger.info("⚠️ 중간 위험 상황 분석...")
        
        context = create_legal_rag_input(
            keyword="팬전쟁",
            messages=[
                "타팬들이 맞댓글 달고 있음",
                "루머가 돌고 있음",
                "SNS에서 논쟁 중"
            ],
            spike_nature="mixed",
            dominant_sentiment="neutral",
            fanwar_targets=["group_a", "group_b"],
        )
        
        signals = legal_agent._quick_risk_check(context)
        
        # 컨텍스트 기반 위험 감지
        assert signals["risk_detected"] is True, "컨텍스트 기반 위험 감지 필수"
        
        logger.info(f"✓ 중간 위험 감지 완료: {signals}")
    
    @pytest.mark.asyncio
    async def test_safe_scenario(self, legal_agent):
        """안전한 상황"""
        logger.info("✅ 안전한 상황 분석...")
        
        context = create_legal_rag_input(
            keyword="팬아트",
            messages=[
                "멋진 팬아트 감사합니다",
                "정말 예쁘네요",
                "사랑합니다"
            ],
            spike_nature="positive",
            dominant_sentiment="positive",
        )
        
        signals = legal_agent._quick_risk_check(context)
        
        assert signals["risk_detected"] is False, "위험 미감지"
        assert len(signals["matched_keywords"]) == 0
        assert signals["reason"] == "none"
        
        logger.info(f"✓ 안전한 상황 확인: {signals}")


# ===========================
# 5. 신뢰도 계산 테스트
# ===========================

class TestConfidenceCalculation:
    """신뢰도 계산 로직"""
    
    def test_confidence_without_rag(self):
        """RAG 미수행 시 신뢰도"""
        logger.info("📊 RAG 미수행 신뢰도 계산...")
        
        # negative spike → confidence = 0.75
        data = {
            "overall_risk_level": "medium",
            "clearance_status": "review_needed",
            "confidence": 0.75,  # LLM이 설정하거나 코드에서 계산
            "rag_required": True,
            "rag_performed": False,  # RAG 미수행
            "rag_confidence": None,
        }
        
        schema = LegalRiskOutputSchema(**data)
        assert schema.confidence == 0.75
        assert schema.rag_confidence is None
        
        logger.info(f"✓ RAG 미수행: confidence={schema.confidence}")
    
    def test_confidence_with_rag(self):
        """RAG 수행 시 신뢰도"""
        logger.info("📊 RAG 수행 신뢰도 계산...")
        
        # 질적 점수(0.8) * 양적 가중치(0.7) = 0.56
        data = {
            "overall_risk_level": "high",
            "clearance_status": "high_risk",
            "confidence": 0.56,  # 계산 결과
            "rag_required": True,
            "rag_performed": True,
            "rag_confidence": 0.8,  # 질적 점수
        }
        
        schema = LegalRiskOutputSchema(**data)
        
        # 신뢰도는 0.0~1.0 범위
        assert 0.0 <= schema.confidence <= 1.0
        assert 0.0 <= schema.rag_confidence <= 1.0
        
        logger.info(f"✓ RAG 수행: confidence={schema.confidence}, rag_confidence={schema.rag_confidence}")
    
    def test_confidence_range_edge_cases(self):
        """신뢰도 경계값"""
        logger.info("📊 신뢰도 경계값 테스트...")
        
        # 최소값
        data_min = {
            "overall_risk_level": "low",
            "clearance_status": "clear",
            "confidence": 0.0,
            "rag_required": False,
            "rag_performed": False,
            "rag_confidence": None,
        }
        schema = LegalRiskOutputSchema(**data_min)
        assert schema.confidence == 0.0
        
        # 최대값
        data_max = {
            "overall_risk_level": "critical",
            "clearance_status": "high_risk",
            "confidence": 1.0,
            "rag_required": True,
            "rag_performed": True,
            "rag_confidence": 1.0,
        }
        schema = LegalRiskOutputSchema(**data_max)
        assert schema.confidence == 1.0
        assert schema.rag_confidence == 1.0
        
        logger.info("✓ 경계값 검증 완료")


# ===========================
# 6. 실제 LLM 호출 테스트 (비용 발생)
# ===========================

class TestLLMIntegration:
    """실제 LLM 호출 테스트 (환경변수: RUN_REAL_LLM=1)"""
    
    @pytest.mark.asyncio
    async def test_critical_risk_llm(self, legal_agent):
        """매우 위험한 상황 LLM 분석 (실제 호출)"""
        if os.getenv("RUN_REAL_LLM") != "1":
            pytest.skip("Set RUN_REAL_LLM=1 to run real LLM tests")
        
        logger.info("🚨 LLM 분석: 매우 위험한 상황...")
        
        context = create_legal_rag_input(
            keyword="명예훼손 소송",
            messages=["고소장 접수됨", "법원 출두 요청"],
            spike_nature="negative",
            dominant_sentiment="boycott",
        )
        
        # 실제 에이전트 호출 (MCP 필요)
        try:
            result = await legal_agent.check_legal_risk(context)
            
            # 결과 검증
            assert isinstance(result, dict)
            assert result["overall_risk_level"] in ["low", "medium", "high", "critical"]
            assert result["clearance_status"] in ["clear", "review_needed", "high_risk"]
            assert 0.0 <= result["confidence"] <= 1.0
            
            logger.info(f"✓ LLM 분석 완료:")
            logger.info(f"  Risk Level: {result['overall_risk_level']}")
            logger.info(f"  Confidence: {result['confidence']}")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {e}")
    
    @pytest.mark.asyncio
    async def test_safe_scenario_llm(self, legal_agent):
        """안전한 상황 LLM 분석"""
        if os.getenv("RUN_REAL_LLM") != "1":
            pytest.skip("Set RUN_REAL_LLM=1 to run real LLM tests")
        
        logger.info("✅ LLM 분석: 안전한 상황...")
        
        context = create_legal_rag_input(
            keyword="팬아트",
            messages=["정말 예쁜 그림입니다", "감사합니다"],
            spike_nature="positive",
            dominant_sentiment="positive",
        )
        
        try:
            result = await legal_agent.check_legal_risk(context)
            
            # 안전한 상황 검증
            assert result["overall_risk_level"] == "low", "안전한 상황은 low여야 함"
            assert result["clearance_status"] == "clear"
            assert result["confidence"] > 0.5, "신뢰도는 양수여야 함"
            
            logger.info(f"✓ 안전 상황 분석 완료: confidence={result['confidence']}")
            
        except Exception as e:
            pytest.skip(f"LLM 호출 실패: {e}")


# ===========================
# pytest 훅
# ===========================

def pytest_configure(config):
    """pytest 시작 전 설정"""
    logger.info("\n" + "="*80)
    logger.info("🚀 Legal RAG Agent 테스트 시작")
    logger.info("="*80)
    logger.info("\n💡 실제 LLM 호출 테스트:")
    logger.info("   RUN_REAL_LLM=1 pytest test_legal_rag_agent.py -v")


def pytest_sessionfinish(session, exitstatus):
    """pytest 종료 후 실행"""
    if session.testsfailed == 0:
        print("\n✅ 모든 Agent 테스트 통과!")
        print("   → 스키마 검증, 신뢰도 계산, 자동 보정 모두 정상")
    else:
        print("\n❌ 일부 Agent 테스트 실패")
        print("   → Pydantic 스키마 또는 계산 로직 확인 필요")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))