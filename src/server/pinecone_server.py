import logging
import json
import os
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from embedder import SMCEmbeddings
from pinecone import Pinecone
import mcp.types as types
from mcp.server import Server
import time
from mcp.server.stdio import stdio_server

logger = logging.getLogger("legal-mcp")
logging.basicConfig(level=logging.INFO)

ARTIST_KEYWORD_CATEGORIES = {
    "direct_legal_action": {
        "description": "직접적 법적 대응",
        "keywords": ["고소", "소송", "법적", "재판", "법원", "소장", "피소"]
    },
    "creative_work_risk": {
        "description": "창작물 리스크",
        "keywords": ["표절", "저작권", "도용", "무단사용", "저작인접권", "불법복제", "음악저작권"]
    },
    "artist_personal_risk": {
        "description": "아티스트 신변 및 디지털 범죄",
        "keywords": ["명예훼손", "허위사실", "딥페이크", "사생활", "유출", "폭로", "모욕", "명예훼손죄", "사이버폭력"]
    },
    "contract_management_risk": {
        "description": "계약 및 경영 리스크",
        "keywords": ["전속계약", "탈퇴", "역바이럴", "계약", "계약금", "위반", "계약자", "경영권"]
    }
}

# -- 1. 검색 결과의 기본 엔트리 -- 
@dataclass
class LegalDocumentEntry:
    id: str                                   
    title: str                                
    text: str                                
    source: str                              
    relevance_score: float                   
    source_type: str                         
    category: str                           
    reference: str                          
    metadata: Dict[str, Any] 

# 법령 엔트리
@dataclass
class StatuteEntry(LegalDocumentEntry):
    pass

# 판례 엔트리
@dataclass
class PrecedentEntry(LegalDocumentEntry):
    pass

# 정책 엔트리 
@dataclass
class PolicyEntry(LegalDocumentEntry):
   pass

# 통합 검색 결과 
@dataclass
class LegalSearchResult:
    query: str
    statutes: List[StatuteEntry]
    precedents: List[PrecedentEntry]
    policies: List[PolicyEntry]
    timestamp: str


# -- 2. 예외 처리 클래스 -- 
class LegalMCPError(Exception):
    pass

class StatuteSearchError(LegalMCPError):
    pass

class PrecedentSearchError(LegalMCPError):
    pass

class PolicySearchError(LegalMCPError):
    pass

# -- 3. 검색 설정 클래스 -- 
class SearchSettings:
    MIN_RELEVANCE_SCORE = 0.5     
    TOP_K = 10
    VECTOR_WEIGHT = 0.4
    BM25_WEIGHT = 0.3
    METADATA_WEIGHT = 0.3
    INDEX_NAME = "dolpin-legal-v1"

 

# -- 4. 핵심 검색 로직을 담당하는 추상 기반 클래스 -- 

class LegalSearcher(ABC):
    def __init__(self, pinecone_client: Pinecone, embeddings: SMCEmbeddings):
        self.pc = pinecone_client
        self.embeddings = embeddings
        logger.info(f"{self.__class__.__name__} initialized (Registry managed)")
        self.index = pinecone_client.Index(SearchSettings.INDEX_NAME)
        
    # 각 도메인별 필터
    @abstractmethod
    def get_filter_spec(self, params: Dict[str, Any]) -> Optional[Dict]:
        pass
    
    # Pinecone 결과를 Dataclass로 변환
    @abstractmethod
    def format_entry(self, match: Dict[str, Any]) -> LegalDocumentEntry:
        pass
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        spike_nature: Optional[str] = None,
        dominant_sentiment: Optional[str] = None,
        legal_keywords: Optional[List[str]] = None,
        **params
    ) -> List[LegalDocumentEntry]:
        try:
            logger.info(
                f"[{self.__class__.__name__}] query={query[:50]}... "
                f"spike={spike_nature}, sentiment={dominant_sentiment}"
            )
            
            start_time = time.time()
            
            # ===== Step 1: Vector 검색 + Context Filter =====
            vector_results = await self._vector_search(
                query=query,
                top_k=top_k * 2,
                spike_nature=spike_nature,
                dominant_sentiment=dominant_sentiment,
                legal_keywords=legal_keywords,
                **params
            )
            
            if not vector_results:
                logger.info("No vector results found")
                return []
            
            # ===== Step 2: BM25 점수 계산 (Hybrid) =====
            hybrid_results = await self._calculate_bm25_scores(query, vector_results)
            
            # ===== Step 3: 점수 결합 (Reranking) =====
            reranked = self._combine_scores(hybrid_results)
            
            # ===== Step 4: Score Threshold 필터링 =====
            filtered = [
                doc for doc in reranked
                if doc.get('final_score', 0) >= SearchSettings.MIN_RELEVANCE_SCORE
            ]
            
            elapsed = time.time() - start_time
            logger.info(
                f"Results: {len(filtered)} docs in {elapsed:.2f}s "
                f"(threshold: {SearchSettings.MIN_RELEVANCE_SCORE})"
            )
            
            return filtered[:top_k]
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _vector_search(
        self,
        query: str,
        top_k: int,
        spike_nature: Optional[str],
        dominant_sentiment: Optional[str],
        legal_keywords: Optional[List[str]],
        **params
    ) -> List[Dict[str, Any]]:
        # Step 1: Vector 검색 + Context Filter
        try:
            # 기본 필터
            filter_spec = self.get_filter_spec(params)
            
            # 컨텍스트 기반 필터
            context_filter = self._build_context_filter(
                spike_nature=spike_nature,
                dominant_sentiment=dominant_sentiment,
                legal_keywords=legal_keywords
            )
            
            # 필터 병합
            if context_filter:
                if filter_spec:
                    filter_spec = {"$and": [filter_spec, context_filter]}
                else:
                    filter_spec = context_filter
            
            logger.debug(f"Filter: {filter_spec}")
            
            # 임베딩 생성
            try:
                embedding = await asyncio.to_thread(
                    self.embeddings.embed_query, query
                )
                logger.debug(f"Embedding: {len(embedding)} dims")
            except Exception as e:
                logger.error(f"Embedding failed: {e}, using fallback")
                embedding = [0.0] * 1536
            
            # Pinecone 쿼리
            async def query_pinecone():
                return self.index.query( 
                    vector=embedding,
                    top_k=top_k,
                    filter=filter_spec,
                    include_metadata=True
                )
            
            results = await asyncio.to_thread(query_pinecone)
            
            # 결과 변환
            documents = []
            for match in results.get("matches", []):
                try:
                    entry_dict = asdict(self.format_entry(match))
                    entry_dict['vector_score'] = match.get('score', 0.0)
                    documents.append(entry_dict)
                except Exception as e:
                    logger.warning(f"Format failed: {e}")
                    continue
            
            return documents
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    # Step 2: BM25 점수 계산
    async def _calculate_bm25_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        legal_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        1. legal_keywords 우선 매칭 (정제된 키워드 사용)
        2. 한국어 조사 처리 (표절이, 표절을, 표절은 모두 매칭)
        3. 메타데이터도 함께 검색
        4. 점수 정규화
        """
        try:
            # 쿼리에서 단어 추출
            query_terms = set(query.lower().split())
            
            # 2️legal_keywords가 있으면 우선 사용
            if legal_keywords:
                keywords_to_match = [kw.lower() for kw in legal_keywords]
            else:
                keywords_to_match = list(query_terms)
            
            if not keywords_to_match:
                for doc in documents:
                    doc['bm25_score'] = 0.0
                return documents
            
            logger.debug(f"BM25 matching {len(keywords_to_match)} keywords in {len(documents)} docs")
            
            # 한국어 조사 목록
            KOREAN_PARTICLES = [
                '이', '을', '는', '은', '가', '를',
                '로', '에', '에게', '한테', '께', '과', '와',
                '아', '어', '여', '여요', '네', '군', '구나'
            ]
            
            for doc in documents:
                text = doc.get('text', '').lower()
                metadata_text = json.dumps(doc.get('metadata', {})).lower()
                
                # 메인 텍스트 + 메타데이터 함께 검색
                full_text = text + " " + metadata_text
                
                matched_count = 0.0
                
                for keyword in keywords_to_match:
                    # 1. 정확한 단어 매칭 (100%)
                    if keyword in full_text:
                        matched_count += 1.0
                    # 2. 한국어 조사 포함 매칭 (80%)
                    else:
                        for particle in KOREAN_PARTICLES:
                            if f"{keyword}{particle}" in full_text:
                                matched_count += 0.8
                                break
                
                # BM25 점수 = 매칭된 키워드 수 / 전체 키워드 수
                bm25_score = matched_count / len(keywords_to_match)
                
                # 점수 정규화 (0.0 ~ 1.0)
                doc['bm25_score'] = min(bm25_score, 1.0)
                
                logger.debug(
                    f"BM25 score for '{doc.get('title', 'Unknown')}': "
                    f"{matched_count:.1f}/{len(keywords_to_match)} = {doc['bm25_score']:.2f}"
                )
            
            return documents
        
        except Exception as e:
            logger.warning(f"BM25 failed: {e}, using default")
            for doc in documents:
                doc['bm25_score'] = 0.5
            return documents
    
    def _combine_scores(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Step 3: 점수 결합 (Reranking)
        for doc in documents:
            metadata_score = self._calculate_metadata_score(doc)
            
            final_score = (
                doc.get('vector_score', 0) * SearchSettings.VECTOR_WEIGHT +
                doc.get('bm25_score', 0) * SearchSettings.BM25_WEIGHT +
                metadata_score * SearchSettings.METADATA_WEIGHT
            )
            
            doc['final_score'] = final_score
        
        return sorted(documents, key=lambda x: x['final_score'], reverse=True)
    
    def _calculate_metadata_score(self, doc: Dict[str, Any]) -> float:
        # 메타데이터 신뢰도 점수
        score = 0.0
        
        source_trust = {
            "법률지식베이스": 1.0,
            "Korean Courts": 0.95,
            "Internal": 0.85
        }
        score += source_trust.get(doc.get('source', ''), 0.5) * 0.5
        
        if doc.get('category') and doc.get('category') != 'unknown':
            score += 0.5
        
        return min(score, 1.0)
    
    def _build_context_filter(
        self,
        spike_nature: Optional[str],
        dominant_sentiment: Optional[str],
        legal_keywords: Optional[List[str]]
    ) -> Optional[Dict]:
        # 컨텍스트 기반 필터
        filters = {}
        
        if legal_keywords:
            category = self._infer_category_from_keywords(legal_keywords)
            if category:
                filters["category"] = {"$eq": category}
                logger.debug(f"Category: {category}")
                return filters
        
        if dominant_sentiment == "criticism":
            filters["category"] = {"$in": ["defamation", "competition"]}
            return filters
        
        if dominant_sentiment == "boycott":
            filters["category"] = {"$in": ["contract", "labor"]}
            return filters
        
        if spike_nature == "negative":
            filters["category"] = {"$eq": "defamation"}
            return filters
        
        return None if not filters else filters
    
    # 아티스트 특화 카테고리 추론
    def _infer_category_from_keywords(self, legal_keywords: List[str]) -> Optional[str]:
        for category, category_info in ARTIST_KEYWORD_CATEGORIES.items():
            keywords_list = category_info.get('keywords', [])
            
            for keyword in legal_keywords:
                if keyword in keywords_list:
                    description = category_info.get('description', '')
                    logger.debug(f"Category {category} ({description}) from {keyword}")
                    return category
        
        return None
    
# -- 5. 도메인별 실제 구현 -- 

# 법령 검색
class StatuteSearcher(LegalSearcher):  
      
    def get_filter_spec(self, params: Dict[str, Any]) -> Optional[Dict]:
        return {"source_type": {"$eq": "law"}} if not params else None
    
    def format_entry(self, match: Dict[str, Any]) -> StatuteEntry:
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")
        try:
            if isinstance(text, str) and "\\" in text:
                text = text.encode('utf-8').decode('unicode_escape')
        except Exception as e:
            logger.warning(f"Failed to decode text: {e}")
        
        return StatuteEntry(
            id=match["id"],
            title=metadata.get("reference", "Unknown"),
            text=text[:500] if text else "",
            source="법률지식베이스",
            relevance_score=match.get("score", 0),
            source_type=metadata.get("source_type", "law"),
            category=metadata.get("category", "unknown"),
            reference=metadata.get("reference", "unknown"),
            metadata=metadata
        )
    
# 판례 검색
class PrecedentSearcher(LegalSearcher):
    def get_filter_spec(self, params: Dict[str, Any]) -> Optional[Dict]:
        return {"source_type": {"$eq": "precedent"}}
    
    def format_entry(self, match: Dict[str, Any]) -> PrecedentEntry:
        metadata = match.get("metadata", {})
        return PrecedentEntry(
            id=match["id"],
            title=metadata.get("reference", "Unknown"),
            text=metadata.get("text", "")[:500],
            source="Korean Courts",
            relevance_score=match.get("score", 0),
            source_type=metadata.get("source_type", "unknown"),
            category=metadata.get("category", "unknown"),
            reference=metadata.get("reference", "unknown"),
            metadata=metadata
        )

# 정책 검색 
class PolicySearcher(LegalSearcher):
    def get_filter_spec(self, params: Dict[str, Any]) -> Optional[Dict]:
        return {"source_type": {"$eq": "policy"}}
    
    def format_entry(self, match: Dict[str, Any]) -> PolicyEntry:
        metadata = match.get("metadata", {})
        return PolicyEntry(
            id=match["id"],
            title=metadata.get("reference", "Unknown"),
            text=metadata.get("text", "")[:500],
            source="Internal",
            relevance_score=match.get("score", 0),
            source_type=metadata.get("source_type", "unknown"),
            category=metadata.get("category", "unknown"),
            reference=metadata.get("reference", "unknown"),
            metadata=metadata
        )


# -- 6. Server Setup -- 

searchers: Dict[str, LegalSearcher] = {}
# 서버 시작 시 한 번만 호출
# 모든 검색기를 미리 생성해서 registry에 등록 (call_tool에서는 조회만) 
def initialize_searchers(pc: Pinecone, embedder: SMCEmbeddings):
    global searchers
    
    logger.info(" Initializing Searcher Registry...")
    
    searchers["search-statutes"] = StatuteSearcher(pc, embedder)
    searchers["search-precedents"] = PrecedentSearcher(pc, embedder)
    searchers["search-internal-policy"] = PolicySearcher(pc, embedder)
    
    logger.info(f"Registry initialized: {list(searchers.keys())}")
    

server = Server("legal-rag-mcp")
pinecone_client: Optional[Pinecone] = None
embeddings_model: Optional[SMCEmbeddings] = None

# MCP 도구 등록 
def register_legal_tools():
    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search-statutes",
                description="한국 법령 검색",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "spike_nature": {"type": "string"},
                        "dominant_sentiment": {"type": "string"},
                        "legal_keywords": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="search-precedents",
                description="판례 검색",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "spike_nature": {"type": "string"},
                        "dominant_sentiment": {"type": "string"},
                        "legal_keywords": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="search-internal-policy",
                description="내부 정책 검색",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "spike_nature": {"type": "string"},
                        "dominant_sentiment": {"type": "string"},
                        "legal_keywords": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["query"]
                }
            )
        ]
    
    @server.call_tool()
    # 기존 인스턴스 재사용 
    async def call_tool(name: str, arguments: dict):
        try:
            logger.info(f"Tool called: {name}")
            
            # Registry에서 기존 인스턴스 조회
            searcher = searchers.get(name)
            if not searcher:
                raise ValueError(f"Unknown tool: {name}")
            
            # 컨텍스트 정보
            spike_nature = arguments.get("spike_nature")
            dominant_sentiment = arguments.get("dominant_sentiment")
            legal_keywords = arguments.get("legal_keywords")
            
            logger.debug(
                f"Context: spike={spike_nature}, sentiment={dominant_sentiment}, "
                f"keywords={legal_keywords}"
            )
            
            # 검색 수행
            results = await searcher.search(
                query=arguments.get("query"),
                top_k=arguments.get("top_k", 10),
                spike_nature=spike_nature,
                dominant_sentiment=dominant_sentiment,
                legal_keywords=legal_keywords
            )
            
            results_dict = [asdict(r) for r in results]
            
            return [types.TextContent(
                type="text",
                text=json.dumps(results_dict, indent=2, ensure_ascii=False)
            )]
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]


# -- 7. 메인 서버 -- 
async def main():
    global pinecone_client, embeddings_model
    
    # Pinecone 초기화
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not set")
    
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    embeddings_model = SMCEmbeddings()
    
    # 서버 시작 시 Registry 초기화 (Searcher 미리 생성)
    initialize_searchers(pinecone_client, embeddings_model)
    
    # 도구 등록
    register_legal_tools()
    
    logger.info("Starting Legal RAG MCP Server...")
    logger.info(f"   Min Score Threshold: {SearchSettings.MIN_RELEVANCE_SCORE}")
    logger.info(f"   Weights - Vector: {SearchSettings.VECTOR_WEIGHT}, "
                f"BM25: {SearchSettings.BM25_WEIGHT}, "
                f"Metadata: {SearchSettings.METADATA_WEIGHT}")
    
    # MCP 서버 실행
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="legal-rag-mcp",
                server_version="4.0.0",
                capabilities=server.get_capabilities(),
            ),
        )


if __name__ == "__main__":
    import asyncio
    from mcp.server.models import InitializationOptions
    
    asyncio.run(main())