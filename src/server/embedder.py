import os
import json
import abc
from typing import Any, List, Dict, TYPE_CHECKING
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

if TYPE_CHECKING:
    import openai

load_dotenv()

# 인터페이스 틀 정의 
class Embedder(abc.ABC):
    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

# OpenAI 기반 추상 클래스 
class BaseOpenAIEmbeddings(Embedder, abc.ABC):
    def __init__(self, model: str = "text-embedding-3-large", dimensions: int = 1536, **kwargs: Any) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("`pip install openai`를 설치해주세요.")
        
        # 내부 변수 저장 
        self.openai = openai
        self.model = model              # 사용할 임베딩 모델 
        self.dimensions = dimensions    # 벡터 차원 수 
        self.client = self._initialize_client(**kwargs) # 클라이언트 생성 

    # OpenAI 클라이언트 생성 방식 틀 
    @abc.abstractmethod
    def _initialize_client(self, **kwargs: Any) -> Any:
        pass
    
    # 단일 쿼리 텍스트를 벡터로 변환
    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        if "dimensions" not in kwargs and self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(input=text, model=self.model, **kwargs)
        return response.data[0].embedding

    # 대량의 문서를 효율적으로 변환하기 위한 배치 메서드
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts, 
            model=self.model, 
            dimensions=self.dimensions
        )
        # 벡터 리스트 하나씩 꺼내서 리턴 
        return [d.embedding for d in response.data]

# 실제 OpenAI API 클라이언트 연결 
class SMCEmbeddings(BaseOpenAIEmbeddings):
    def _initialize_client(self, **kwargs: Any) -> Any:
        return self.openai.OpenAI(**kwargs)

# Pinecone 인덱서 정의
class LegalPineconeIndexer:
    def __init__(self, index_name: str, embedding_model: SMCEmbeddings):
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY가 환경 변수에 설정되지 않았습니다.")
        
        # Pinecone 클라이언트 초기화
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.embeddings = embedding_model
        
        # Pinecone 인덱스 생성 
        if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
            print(f"인덱스 '{index_name}' 생성 중...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embeddings.dimensions,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)

    # JSON 지식 베이스를 읽어 Pinecone에 업로드
    def upsert_knowledge_base(self, json_file: str):
        if not os.path.exists(json_file):
            print(f"파일을 찾을 수 없습니다: {json_file}")
            return

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"총 {len(data)}건의 데이터 처리를 시작합니다.")
        
        # Pinecone 업로드 최적화를 위한 배치 처리 
        batch_size = 5
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            vectors = []
            for j, item in enumerate(batch):
                try:
                    # 2. 개별 아이템별로 임베딩 생성
                    content = item['page_content']
                    
                    # 단일 텍스트가 너무 길면 잘라줌...
                    if len(content) > 20000: # 대략적인 글자수 기준
                        content = content[:20000]
                    
                    embed = self.embeddings.embed_query(content)
                    
                    # 3. 메타데이터 유니코드 이스케이프 
                    safe_metadata = {}
                    safe_metadata["text"] = item['page_content'].encode('unicode_escape').decode('ascii')
                    for key, value in item['metadata'].items():
                        if isinstance(value, str):
                            safe_metadata[key] = value.encode('unicode_escape').decode('ascii')
                        else:
                            safe_metadata[key] = value

                    vectors.append({
                        "id": f"sm_law_{i + j}",
                        "values": embed,
                        "metadata": safe_metadata
                    })
                except Exception as e:
                    print(f"아이템 {i+j} 임베딩 중 에러 발생 (건너뜀): {e}")
                    continue
            try:
                self.index.upsert(vectors=vectors)
                print(f"[{i + len(batch)} / {len(data)}] 업로드 완료...")
            except Exception as e:
                print(f"업로드 중 에러 발생 (배치 {i}): {e}")
                continue

if __name__ == "__main__":
    # 1. 임베딩 모델 설정 
    sm_embeddings = SMCEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"), 
        dimensions=1536
    )
    
    # 2. Pinecone 인덱서 초기화 
    indexer = LegalPineconeIndexer(
        index_name="dolpin-legal-v1", 
        embedding_model=sm_embeddings
    )
    
    # 3. 데이터 업로드 실행
    indexer.upsert_knowledge_base("legalsource.json")
    
    print("모든 법률 지식 베이스가 Pinecone에 성공적으로 적재되었습니다.")