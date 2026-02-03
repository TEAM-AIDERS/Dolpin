"""
Custom Lexicon MCP Server - 공통 참조 지식 베이스
팬덤 특화 표현을 조회하는 Read-only Lookup MCP

입력 텍스트에서 팬덤 특화 표현, 행동 트리거, 맥락 신호를 구조적으로 제공한다.
판단이나 추론을 수행하지 않으며, 하위 Agent들이 활용할 정적 지식 조회 인터페이스다.

버전: 260203 - entry 생성 실패 시 fallback 생성 로직 포함
"""

import logging
import json 
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 1. 데이터 모델
# ============================================================

@dataclass
class LexiconEntry:
    """CSV에서 로드한 팬덤 특화 표현 엔트리"""
    term: str
    normalized_form: str
    type: Literal["boycott_action", "context_marker", "fandom_slang"]
    sentiment_label: str
    trigger_type: str  # action, emotion, context
    action_strength: str  # collective, declaration, none
    fandom_scope: str  # fandom, global
    target_entity: str  # agency, artist, self, fandom, unknown
    polarity: Literal["positive", "negative", "neutral"]
    intensity: Literal["high", "mid", "low"]
    risk_flag: Literal["alert", "watch", "none"]
    example_text: str
    usage_mode: str  # literal, ambiguous
    notes: str
    created_at: str
    updated_at: str


@dataclass
class LexiconMatch:
    """텍스트 매칭 결과"""
    term: str
    entry: LexiconEntry
    context_window: str  # 매칭 문맥


@dataclass
class AnalysisContext:
    """Agent들의 공통 분석 컨텍스트"""
    text: str
    matches: List[LexiconMatch]
    aggregated_signals: Dict[str, Any]  # sentiment, triggers, risks 종합


# ============================================================
# 2. LexiconServer 클래스
# ============================================================

class LexiconServer:
    """팬덤 특화 표현 데이터베이스 - MCP 도구의 기반"""
    
    def __init__(self, csv_path: str = "custom_lexicon.csv"):
        """CSV 파일로부터 lexicon 로드"""
        self.csv_path = Path(csv_path)
        
        # 1. 인코딩 처리 + 명시 로그
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            self.encoding_used = 'utf-8'
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {self.csv_path}, trying cp949...")
            self.df = pd.read_csv(self.csv_path, encoding='cp949')
            self.encoding_used = 'cp949'
        
        logger.info(f"Loaded {len(self.df)} lexicon entries from {csv_path}")
        logger.info(f"Encoding used: {self.encoding_used}")
        
        # 2. CSV Validation (무효한 행 필터링)
        self._validate_csv()
        
        # 3. 효율성을 위한 인덱싱
        self.by_term = self._index_by_term()  # 중복 지원
        self.by_type = self._index_by_column('type')
        self.by_trigger = self._index_by_column('trigger_type')
        self.by_risk = self._index_by_column('risk_flag')
        
        # 4. 성능 개선: 정규식 컴파일 (길이순)
        self._compile_term_regex()
    
    def _validate_csv(self):
        """CSV 데이터 검증 및 필터링 (Flexible 모드)"""
        # 필수 컬럼: term과 sentiment_label만 필수
        required_columns = ['term', 'sentiment_label']
        
        missing = set(required_columns) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 1. 빈 term 제거 (유일한 엄격 체크)
        before_count = len(self.df)
        self.df = self.df[self.df['term'].notna() & (self.df['term'] != '')]
        removed_empty = before_count - len(self.df)
        if removed_empty > 0:
            logger.info(f"Removed {removed_empty} rows with empty term")
        
        # 2. 선택적 필드 정규화 (기본값으로 채우기)
        defaults = {
            'type': 'unknown',
            'polarity': 'neutral',
            'intensity': 'low',
            'risk_flag': 'none',
            'trigger_type': 'context',
            'action_strength': 'none',
            'fandom_scope': 'global',
            'target_entity': 'unknown',
            'normalized_form': 'term',
            'example_text': '',
            'usage_mode': 'literal',
            'notes': '',
            'created_at': '',
            'updated_at': ''
        }
        
        for col, default in defaults.items():
            if col not in self.df.columns:
                self.df[col] = default
            else:
                # NaN과 빈 문자열을 기본값으로 대체
                if col == 'normalized_form':
                    # normalized_form은 term 값으로 사용
                    self.df[col] = self.df[col].fillna(self.df['term'])
                    # 빈 문자열 처리
                    mask = self.df[col].apply(lambda x: isinstance(x, str) and x.strip() == '')
                    self.df.loc[mask, col] = self.df.loc[mask, 'term']
                else:
                    self.df[col] = self.df[col].fillna(default)
                    self.df[col] = self.df[col].apply(
                        lambda x: default if (isinstance(x, str) and x.strip() == '') else x
                    )
        
        # 3. 동적 유효값 학습 (화이트리스트 대신)
        self.valid_types = set(self.df['type'].unique())
        self.valid_polarities = set(self.df['polarity'].unique())
        self.valid_intensities = set(self.df['intensity'].unique())
        self.valid_risks = set(self.df['risk_flag'].unique())
        
        # 4. 통계 정보 로깅
        logger.info(f"CSV Validation Complete:")
        logger.info(f"  - Total rows: {len(self.df)}")
        logger.info(f"  - Types discovered: {self.valid_types}")
        logger.info(f"  - Polarities: {self.valid_polarities}")
        logger.info(f"  - Intensities: {self.valid_intensities}")
        logger.info(f"  - Risk flags: {self.valid_risks}")

    
    def _index_by_term(self) -> Dict[str, List[Dict]]:
        """term 인덱싱 (중복 term 지원)"""
        index = {}
        for _, row in self.df.iterrows():
            term = str(row['term']).strip()
            if term not in index:
                index[term] = []
            index[term].append(row.to_dict())
        
        # 중복 term 로그
        duplicates = {t: len(rows) for t, rows in index.items() if len(rows) > 1}
        if duplicates:
            logger.warning(f"Duplicate terms found: {duplicates}")
        
        return index
    
    def _compile_term_regex(self):
        """최적화: 길이 긴 term부터 정규식 컴파일 (캡처 그룹 제거)"""
        # 길이 역순 정렬 (긴 term 먼저 매칭)
        terms_sorted = sorted(
            self.df['term'].unique(),
            key=lambda t: len(str(t)),
            reverse=True
        )
        
        try:
            # 캡처 그룹 () 제거 - 간단한 패턴
            escaped_terms = [re.escape(str(term)) for term in terms_sorted]
            pattern = "|".join(escaped_terms)  # ← 그룹 제거
            self.term_pattern = re.compile(pattern, re.IGNORECASE)
            self.terms_sorted = terms_sorted
            
            # lower 변환 lookup dict (O(1) 매치)
            self.term_lower_to_term = {str(t).lower(): t for t in terms_sorted}
            
            logger.info(f"Compiled regex pattern for {len(terms_sorted)} terms")
        except Exception as e:
            logger.error(f"Failed to compile term regex: {e}")
            self.term_pattern = None
            self.terms_sorted = terms_sorted
            self.term_lower_to_term = {str(t).lower(): t for t in terms_sorted}
    
    def _index_by_column(self, column: str) -> Dict[str, List[Dict]]:
        """컬럼 기반 인덱싱"""
        index = {}
        for _, row in self.df.iterrows():
            key = row[column]
            if key not in index:
                index[key] = []
            index[key].append(row.to_dict())
        return index
    
    # ========== 기본 조회 도구 ==========
    
    def lookup_term(self, term: str) -> Optional[Dict[str, Any]]:
        """단일 용어 조회 (첫 번째 항목)"""
        term_clean = str(term).strip()
        if term_clean in self.by_term and len(self.by_term[term_clean]) > 0:
            return self.by_term[term_clean][0]
        return None
    
    def lookup_terms(self, terms: List[str]) -> List[Dict[str, Any]]:
        """다중 용어 조회"""
        results = []
        for term in terms:
            entry = self.lookup_term(term)
            if entry:
                results.append(entry)
        return results
    
    def search_by_type(self, type_: str) -> List[Dict[str, Any]]:
        """타입별 조회 (boycott_action, context_marker, fandom_slang)"""
        return self.by_type.get(type_, [])
    
    def search_by_trigger(self, trigger_type: str) -> List[Dict[str, Any]]:
        """트리거 타입별 조회 (action, emotion, context)"""
        return self.by_trigger.get(trigger_type, [])
    
    def search_by_risk(self, risk_flag: str) -> List[Dict[str, Any]]:
        """위험도 수준별 조회 (alert, watch, none)"""
        return self.by_risk.get(risk_flag, [])
    
    # ========== 분석용 도구 ==========
    
    def extract_matches(self, text: str) -> List[LexiconMatch]:
        """
        텍스트에서 매칭되는 용어 추출

        개선사항:
        - LexiconEntry 생성 실패 시 fallback entry 생성
        - 매치가 손실되지 않도록 보장
        """
        matches = []
        
        # text 원본 사용 (인덱스가 원문 기준)
        if self.term_pattern:
            for match in self.term_pattern.finditer(text): 
                matched_term_lower = match.group(0).lower()
                start_idx = match.start()
                end_idx = match.end()
                
                # context window 추출 (양쪽 30자)
                ctx_start = max(0, start_idx - 30)
                ctx_end = min(len(text), end_idx + 30)
                context = text[ctx_start:ctx_end]
                
                # O(1) 매치: dict 조회로 원본 term 찾기
                term = self.term_lower_to_term.get(matched_term_lower)
                if term:
                    # 중복 term 모두 처리
                    for entry_dict in self.by_term.get(term, []):
                        try:
                            entry = LexiconEntry(**entry_dict)
                            matches.append(LexiconMatch(
                                term=term,
                                entry=entry,
                                context_window=context
                            ))
                        except TypeError as e:
                            # ===== Fallback: 기본값으로 LexiconEntry 생성 =====
                            logger.warning(
                                f"Entry 생성 실패 (term: {term}): {e}. "
                                f"기본값으로 Entry를 생성합니다."
                            )
                        
                            fallback_entry = self._create_fallback_entry(term, entry_dict, error=str(e))
                            matches.append(LexiconMatch(
                                term=term,
                                entry=fallback_entry,
                                context_window=context
                            ))
        else:
            # Fallback: 정규식 실패 시 길이순 순회
            logger.debug("Using fallback term matching (regex unavailable)")
            text_lower = text.lower()
            for term in self.terms_sorted:
                term_lower = str(term).lower()
                if term_lower in text_lower:
                    idx = text_lower.find(term_lower)
                    start = max(0, idx - 30)
                    end = min(len(text), idx + len(term_lower) + 30)
                    context = text[start:end]
                    
                    for entry_dict in self.by_term.get(term, []):
                        try:
                            entry = LexiconEntry(**entry_dict)
                            matches.append(LexiconMatch(
                                term=term,
                                entry=entry,
                                context_window=context
                            ))
                        except TypeError as e:
                            # ===== Fallback =====
                            logger.warning(
                                f"LexiconEntry 생성 실패 (term: {term}): {e}. "
                                f"fallback entry 생성합니다."
                            )
                            fallback_entry = self._create_fallback_entry(term, entry_dict, error=str(e))
                            matches.append(LexiconMatch(
                                term=term,
                                entry=fallback_entry,
                                context_window=context
                            ))
        
        logger.debug(f"Found {len(matches)} matches in text")
        return matches
    
    def _create_fallback_entry(
        self, 
        term: str, 
        entry_dict: Dict[str, Any], 
        error: str
    ) -> LexiconEntry:
        """
        LexiconEntry 생성 실패 시 fallback entry 생성
    
        Args:
            term: 매칭된 용어
            entry_dict: 원본 dict (일부 필드만 있을 수 있음)
            error: 에러 메시지
    
        Returns:
            exiconEntry: 기본값으로 채워진 fallback entry
        """
        # 안전한 값 추출 (없으면 기본값)

        def safe_get(key: str, default: Any) -> Any:
            value = entry_dict.get(key, default)
            # None이거나 빈 문자열이면 기본값 사용
            if value is None or (isinstance(value, str) and value.strip() == ''):
                return default
            return value
    
        # Fallback LexiconEntry 생성
        fallback = LexiconEntry(
            term=term,
            normalized_form=safe_get('normalized_form', term),  # term으로 대체
            type=safe_get('type', 'unknown'),  # 기본값
            sentiment_label=safe_get('sentiment_label', 'neutral'),
            trigger_type=safe_get('trigger_type', 'context'),
            action_strength=safe_get('action_strength', 'none'),
            fandom_scope=safe_get('fandom_scope', 'global'),
            target_entity=safe_get('target_entity', 'unknown'),
            polarity=safe_get('polarity', 'neutral'),
            intensity=safe_get('intensity', 'low'),
            risk_flag=safe_get('risk_flag', 'none'),
            example_text=safe_get('example_text', ''),
            usage_mode=safe_get('usage_mode', 'literal'),
            notes=f"[FALLBACK] {error}",  # 👈 에러 정보 기록
            created_at=safe_get('created_at', ''),
            updated_at=safe_get('updated_at', '')
        )
    
        logger.debug(f"Created fallback entry for term: {term}")
        return fallback

    def analyze_text(self, text: str) -> AnalysisContext:
        """텍스트 전체 분석 - Agent들의 입력 컨텍스트"""
        matches = self.extract_matches(text)
        
        # 신호 종합
        aggregated = {
            "total_matches": len(matches),
            "matched_terms": [m.term for m in matches],
            "sentiment_mix": self._aggregate_sentiment([m.entry for m in matches]),
            "action_triggers": self._extract_triggers([m.entry for m in matches]),
            "risk_flags": [m.entry.risk_flag for m in matches if m.entry.risk_flag != "none"],
            "target_entities": list(set(m.entry.target_entity for m in matches)),
        }
        
        return AnalysisContext(
            text=text,
            matches=matches,
            aggregated_signals=aggregated
        )
    
    def _aggregate_sentiment(self, entries: List[LexiconEntry]) -> Dict[str, Any]:
        """감정 신호 종합"""
        if not entries:
            return {"positive": 0, "negative": 0, "neutral": 0}
        
        sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}
        for e in entries:
            if e.polarity in sentiment_count:
                sentiment_count[e.polarity] += 1
        
        return sentiment_count
    
    def _extract_triggers(self, entries: List[LexiconEntry]) -> List[Dict[str, Any]]:
        """트리거 신호 추출 (BUG FIX: return 추가!)"""
        triggers = []
        for e in entries:
            if e.trigger_type and e.trigger_type != "none":
                triggers.append({
                    "term": e.term,
                    "trigger_type": e.trigger_type,
                    "action_strength": e.action_strength,
                    "type": e.type,
                })
        return triggers  # ✅ 이 줄이 없었음!
    
    # ========== MCP Tool 실행 ==========
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """MCP 도구 실행 (lookup_term 지원 추가!)"""
        
        # [1] lookup_term 지원
        if tool_name == "lookup_term":
            term = params.get("term")
            result = self.lookup_term(term)
            return {
                "tool": "lookup_term",
                "term": term,
                "entry": result,
                "found": result is not None
            }
        
        # [2] lookup_terms 지원
        elif tool_name == "lookup_terms":
            terms = params.get("terms", [])
            results = self.lookup_terms(terms)
            return {
                "tool": "lookup_terms",
                "terms": terms,
                "entries": results,
                "count": len(results)
            }
        
        # [3] analyze_text (핵심)
        elif tool_name == "analyze_text":
            text = params.get("text")
            context = self.analyze_text(text)
            
            matches_data = [
                {
                    "term": m.term,
                    "entry": asdict(m.entry),
                    "context_window": m.context_window
                }
                for m in context.matches
            ]
            
            return {
                "tool": "analyze_text",
                "text": text,
                "matches": matches_data,
                "aggregated_signals": context.aggregated_signals
            }
        
        # [4] get_sentiment_context
        elif tool_name == "get_sentiment_context":
            text = params.get("text")
            context = self.analyze_text(text)
            
            sentiment_signals = [
                {
                    "term": m.term,
                    "polarity": m.entry.polarity,
                    "intensity": m.entry.intensity,
                    "sentiment_label": m.entry.sentiment_label
                }
                for m in context.matches
            ]
            
            return {
                "tool": "get_sentiment_context",
                "text": text,
                "sentiment_signals": sentiment_signals,
                "sentiment_mix": context.aggregated_signals["sentiment_mix"]
            }
        
        # [5] get_routing_context
        elif tool_name == "get_routing_context":
            text = params.get("text")
            context = self.analyze_text(text)
            
            routing_signals = [
                {
                    "term": m.term,
                    "type": m.entry.type,
                    "target_entity": m.entry.target_entity,
                    "fandom_scope": m.entry.fandom_scope
                }
                for m in context.matches
            ]
            
            return {
                "tool": "get_routing_context",
                "text": text,
                "routing_signals": routing_signals,
                "risk_flags": context.aggregated_signals["risk_flags"]
            }
        
        # [6] get_causality_context
        elif tool_name == "get_causality_context":
            text = params.get("text")
            context = self.analyze_text(text)
            
            causality_signals = [
                {
                    "term": m.term,
                    "normalized_form": m.entry.normalized_form,
                    "trigger_type": m.entry.trigger_type,
                    "action_strength": m.entry.action_strength
                }
                for m in context.matches
            ]
            
            return {
                "tool": "get_causality_context",
                "text": text,
                "causality_signals": causality_signals,
                "action_triggers": context.aggregated_signals["action_triggers"]
            }
        
        # [7] get_playbook_context
        elif tool_name == "get_playbook_context":
            text = params.get("text")
            context = self.analyze_text(text)
            
            action_signals = [
                {
                    "term": m.term,
                    "type": m.entry.type,
                    "action_strength": m.entry.action_strength,
                    "risk_flag": m.entry.risk_flag
                }
                for m in context.matches
            ]
            
            return {
                "tool": "get_playbook_context",
                "text": text,
                "action_signals": action_signals,
                "risk_flags": context.aggregated_signals["risk_flags"],
                "target_entities": context.aggregated_signals["target_entities"]
            }
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def create_mcp_tools(self) -> List[Dict[str, Any]]:
        """MCP 도구 정의 (inputSchema 통일)"""
        return [
            {
                "name": "lookup_term",
                "description": "단일 용어 조회",
                "inputSchema": {  # ← inputSchema로 통일
                    "type": "object",
                    "properties": {"term": {"type": "string"}},
                    "required": ["term"]
                }
            },
            {
                "name": "lookup_terms",
                "description": "다중 용어 조회",
                "inputSchema": {
                    "type": "object",
                    "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
                    "required": ["terms"]
                }
            },
            {
                "name": "analyze_text",
                "description": "텍스트 분석 및 팬덤 표현 추출",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            },
            {
                "name": "get_sentiment_context",
                "description": "SentimentAgent용 컨텍스트",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            },
            {
                "name": "get_routing_context",
                "description": "RouterAgent용 컨텍스트",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            },
            {
                "name": "get_causality_context",
                "description": "CausalityAgent용 컨텍스트",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            },
            {
                "name": "get_playbook_context",
                "description": "PlaybookAgent용 컨텍스트",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            }
        ]


