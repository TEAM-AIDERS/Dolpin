# 🐬 Dolpin — K-pop Fandom Issue Analysis & Response System

> **K-pop 기획사 홍보팀을 위한 Multi-Agent 기반 실시간 팬덤 이슈 분석 및 대응 시스템**

K-pop 아티스트 관련 소셜·커뮤니티 데이터를 실시간으로 수집하고, 이슈 스파이크 탐지 → 감성 분류 → 인과관계 분석 → 법적 리스크 판단 → 대응 전략 생성까지의 전 과정을 Multi-Agent 파이프라인으로 자동화합니다. 분석 결과는 Slack 알림과 Streamlit 대시보드로 홍보팀에 즉시 전달됩니다.

---

## 📌 목차

1. [시스템 개요](#1-시스템-개요)
2. [아키텍처](#2-아키텍처)
3. [주요 기능](#3-주요-기능)
4. [소스 코드 설명](#4-소스-코드-설명)
5. [사용 기술 및 오픈소스](#5-사용-기술-및-오픈소스)
6. [디렉토리 구조](#6-디렉토리-구조)
7. [How to Build](#7-how-to-build)
8. [How to Install](#8-how-to-install)
9. [How to Test](#9-how-to-test)
10. [샘플 데이터](#10-샘플-데이터)
11. [팀원](#11-팀원)

---

## 1. 시스템 개요

### 배경 및 필요성

K-pop에서 연예인 이미지는 매출과 직결됩니다. SNS 중심의 정보·반응 확산 속도가 빨라지면서 팬덤 이슈에 얼마나 빠르게, 적절하게 대응하는가가 PR 기회(역주행·리브랜딩·광고)와 리스크(이미지 하락·은퇴·엔터사 주가 폭락)를 가르는 핵심 변수가 되었습니다.

그러나 현재 K-pop 기획사 홍보팀은 여러 플랫폼의 팬덤 이슈를 수동으로 모니터링하고 있어 이슈 탐지 지연, 대응 방향 판단 지연으로 초기 대응 골든타임을 놓치는 문제가 반복됩니다.

| Problem | Solution |
|---------|----------|
| 수작업 모니터링의 한계 | 멀티 플랫폼 팬덤 데이터 모니터링 자동화 |
| 실시간 이슈 탐지 및 분석 시스템 부재 | AI Agent 기반 실시간 감지 및 분석 |
| 범용 브랜드 모니터링 툴의 한계 | K-pop 팬덤 도메인 특화 PR 대응 의사결정 지원 |

### Dolpin의 접근

- **실시간 수집**: Twitter, 인스티즈, Google Trends 3개 소스를 Kafka 스트리밍으로 통합
- **자동 필터링**: Z-Score + Actionability Score 기반 SpikeAnalyzer로 LLM 호출의 약 80%를 사전 차단, false positive를 약 25% 감소
- **Multi-Agent 분석**: LangGraph StateGraph로 11개 노드, 3개 조건부 라우터 구성
- **즉시 전달**: 분석 완료 후 1분 이내 Slack 알림 + Streamlit 대시보드로 실시간 시각화

### 기존 솔루션 대비 차별점

| 구분 | Brandwatch | Meltwater | **Dolpin** |
|------|-----------|-----------|------------|
| 주 대상 | 브랜드/마케팅/PR 팀 | 브랜드/마케팅/PR 팀 | **K-pop 기획사 홍보팀** |
| 분석 범위 | 범용 브랜드 여론 | 범용 브랜드/미디어 여론 | **K-pop 팬덤 이슈 특화** |
| 의사결정 지원 | 일부 알림·인사이트 | 일부 알림·인사이트 | **대응 판단 및 요약 리포트 중심** |
| 차별점 | AI 기반 카테고리화 | 폭넓은 채널 커버리지 | **Multi-Agent 기반 단계별 분석 워크플로우** |

### 기대 효과

| | 내용 |
|--|------|
| **① 이슈 조기 감지** | 분석 시작 후 1분 이내 Slack 알림 제공, 초기 대응 시간 단축 → 이슈 확산 리스크 완화 |
| **② 의사결정 지원** | 핵심 이슈와 감정 원인 중심 요약, 대응 전략 추천 및 PR 가이드라인 반영, 모니터링·보고서 작성 부담 완화 |
| **③ 운영 확장성** | 팬덤 용어·기업별 대응 정책 반영, 법무 기준 연계, PR Bot·자동 리포트 확장 가능 |

---

## 2. 아키텍처

> 아래 위치에 아키텍처 다이어그램 이미지를 삽입하세요.

```
[아키텍처 다이어그램 이미지]
assets/architecture.png
```

<!-- 이미지 준비 후 아래 주석을 해제하세요 -->
<!-- ![Dolpin Architecture](assets/architecture.png) -->

### 전체 데이터 흐름

```
[데이터 수집]
Twitter MCP Server (Node.js)
인스티즈 Playwright 크롤러      →  UnifiedCollector  →  KafkaProducer  →  Kafka Topic (3 partitions)
Google Trends API                                                          (Confluent Cloud)

[분석 파이프라인]
Kafka Topic  →  KafkaConsumer  →  AnalysisState 변환  →  LangGraph graph.ainvoke()

[LangGraph 워크플로우]
SpikeAnalyzer → Router1 → LexiconLookup → Sentiment → Router2
                   ↓                                       ↓            ↓
                 END(skip)                            Causality    Playbook(fast)
                                                         ↓
                                                      Router3
                                                    ↙         ↘
                                              LegalRAG    Amplification
                                                    ↘         ↙
                                                    Playbook
                                                       ↓
                                                   ExecBrief → END

[출력]
Slack 알림  +  결과 저장 (JSON)
```

### 라우터 분기 기준

| 라우터 | 분기 기준 | 경로 |
|--------|-----------|------|
| Router 1 | `actionability_score ≥ 0.3` | 분석 진행 / skip(END) |
| Router 2 | `positive_viral_detected` 플래그 | Playbook(빠른 대응) / Causality |
| Router 3 | `positive_viral_detected` 플래그 (Router 2에서 설정) | Amplification / LegalRAG |

---

## 3. 주요 기능

### 3.1 실시간 멀티소스 수집 (`src/pipeline/`)

| 수집기 | 소스 | 방식 |
|--------|------|------|
| `TwitterCollector` | X(Twitter) | Node.js MCP 서버 (X API v2) |
| `InstizCollector` | 인스티즈 커뮤니티 | Playwright 기반 웹 크롤링 |
| `GoogleTrendsCollector` | Google Trends | pytrends 라이브러리 |

3개 소스의 이기종 데이터를 `KafkaMessage` Pydantic 스키마로 통합 정규화한 뒤 Kafka 토픽에 적재합니다. 중복 데이터 방지 캐시와 사용자 ID 익명화(`user_***` 형식)가 수집 단계에서 적용됩니다.

### 3.2 Spike 탐지 (`src/agents/spike_analyzer.py`)

- **Z-Score**: 최근 24시간 언급량(5분 단위 288개 슬라이딩 윈도우)의 평균·표준편차 대비 현재 값 편차 측정 (최소 12개 이상 데이터 수집 후 활성화)
- **가속도 지표**: 1분 EPS vs 5분 EPS 비율 비교로 급증 속도 측정
- **Actionability Score**: `spike_intensity × 0.5 + keyword_risk × 0.5`
  - `keyword_risk`는 MCP 렉시콘 서버에서 `action_strength`, `risk_flag` 기반으로 산출
- GCP Cloud Monitoring으로 실시간 메트릭 전송

### 3.3 팬덤 감성 분류 (`src/agents/sentiment_agent.py`)

- K-pop 팬덤 특화 데이터셋으로 파인튜닝된 HuggingFace 모델 (`Aerisbin/sentiment-agent-v1`) 사용
- 6개 클래스: `support` / `disappointment` / `boycott` / `meme` / `fanwar` / `neutral`
- **2단계 라우팅**: 모델 예측 후 렉시콘 기반 오버라이드 적용
  - boycott ≥ 0.60, fanwar ≥ 0.55, meme ≥ 0.45 임계값에서 오버라이드

### 3.4 이슈 인과관계 분석 (`src/agents/causality_agent.py`)

- NetworkX 방향성 그래프(DiGraph)로 메시지 간 확산 경로 모델링
- 매개 중심성(betweenness)·연결 중심성(degree) 계산으로 핵심 확산 노드 추출
- 중심화 지수(centralization index) 기반 패턴 분류
  - `> 0.65`: coordinated (조율된 집단 행동)
  - `> 0.35`: viral (자연 확산)
  - `≤ 0.35`: echo_chamber (에코챔버)

### 3.5 법적 리스크 분석 (`src/agents/legalrag_agent.py`)

3단계 Agentic RAG:
1. **Quick Risk Check**: 18개 법적 키워드 + 컨텍스트 기반 사전 필터
2. **Agentic Loop**: GPT-4o 플래너가 `CALL: [tool] QUERY: [query]` / `STOP` 패턴으로 최대 3회 반복 검색 (법령 / 판례 / 내부정책 3개 MCP 도구 선택)
3. **Synthesis**: `with_structured_output(LegalRiskOutputSchema)` 로 구조화 출력 생성, Pydantic `model_validator`로 Risk-Status 매핑 불일치 자동 보정

### 3.6 대응 플레이북 생성 (`src/agents/playbook_agent.py`)

- Rule-based 전략 분류 (crisis / opportunity / monitoring / default) 후 GPT-4-turbo로 상세 대응안 작성
- ExecBrief는 LLM 없이 규칙 기반 함수만으로 요약 생성 (비용 최적화)

### 3.7 결과 제공: Slack + Dashboard (`src/integrations/slack/`, `dashboard.py`)

Agent 분석 결과(감정 유형·분포, 확산 원인 키워드, 법적 리스크 수준, 바이럴 가능성 점수)를 기반으로 대응 판단을 분류하고 결과를 전달합니다.

**대응 판단 분류:**

| 판단 | 조건 | 긴급도 |
|------|------|--------|
| 🔴 Crisis | 법적 리스크 감지 / 부정 급증 | Urgent / High |
| 🟢 Opportunity | 긍정 급증 / 바이럴 가능성 | Medium |
| 🔵 Monitoring | 부정 일정 수준 이상 | Low |

- **Slack**: ExecBrief 요약 + 대응 행동 추천을 Block Kit 포맷으로 지정 채널에 자동 전송
- **Streamlit Dashboard** (`dashboard.py`): 실시간 감성 분포 시각화, Spike Rate, Actionability Score, 트렌드 변화 추적

---

## 4. 소스 코드 설명

### Kafka 파이프라인

`KafkaProducer`는 `linger.ms=10000`, `batch.size=32768`, `gzip` 압축을 적용해 비용을 최적화하고, `acks=all`, `retries=5`로 전송 신뢰성을 확보합니다. 파티션 키는 `{keyword}-{YYYYMMDDHH}` 복합 키로, 아티스트별 시간대 기준 순서를 보장하면서 Hot Partition을 시간 축으로 분산합니다.

`KafkaConsumer`는 메시지 수신 실패 시 DLQ(`failed_events_log.jsonl`)에 기록하고, LangGraph 워크플로우를 별도 스레드의 새 이벤트 루프에서 실행합니다 (`legal_rag_node`가 비동기이므로).

### MCP 서버

| 서버 | 역할 | 도구 |
|------|------|------|
| `lexicon_server.py` | 렉시콘 분석 | `lexicon_analyze` |
| `pinecone_server.py` | 법률 문서 검색 | `search_statutes`, `search_precedents`, `search_internal_policy` |
| X MCP Server (Node.js) | Twitter 데이터 수집 | X API v2 래핑 |

법률 지식 베이스(`legalsource.json`)는 `embedder.py`로 Pinecone 인덱스(`dolpin-legal-v1`)에 적재합니다. 임베딩 모델은 `text-embedding-3-large` (dimensions=1536)입니다.

### LangGraph StateGraph

`AnalysisState`를 공유 상태로 11개 노드가 순차적으로 읽고 씁니다. 각 노드의 출력 필드:

| 필드 | 담당 노드 |
|------|-----------|
| `spike_analysis` | SpikeAnalyzer |
| `lexicon_result` | LexiconLookup |
| `sentiment_result` | Sentiment |
| `causality_result` | Causality |
| `legal_risk` | LegalRAG |
| `amplification_summary` | Amplification |
| `playbook` | Playbook |
| `executive_brief` | ExecBrief |

---

## 5. 사용 기술 및 오픈소스

### 기술 스택

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=flat-square)
![Kafka](https://img.shields.io/badge/Confluent_Kafka-3_Partitions-231F20?style=flat-square&logo=apachekafka)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-00B388?style=flat-square)
![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4?style=flat-square&logo=googlecloud)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Slack](https://img.shields.io/badge/Slack-SDK-4A154B?style=flat-square&logo=slack)

### 전체 기술 스택 (발표자료 기준)

| 계층 | 기술 | 용도 | 선택 이유 |
|------|------|------|-----------|
| 데이터 수집 | Playwright | 인스티즈 웹 크롤링 | 로그인 자동화 및 동적 페이지 처리 지원 |
| 데이터 수집 | pytrends | Google Trends 수집 | 비공식 API 없이 트렌드 관심도 수집 가능 |
| 데이터 수집 | Twitter API v2 (MCP) | Twitter 게시글 수집 | 공식 API 기반, 반응 지표 포함 수집 가능 |
| 스트리밍 | Apache Kafka (Confluent Cloud) | 실시간 이벤트 처리 | 수집·분석 계층 간 느슨한 결합, 고가용성 |
| 분석 | LangGraph | Multi-Agent 파이프라인 | 조건부 분기 및 상태 기반 워크플로우 구성 |
| 분석 | OpenAI GPT-4o / GPT-4-turbo | 법적 리스크 평가 및 전략 생성 | 구조화 출력 및 자연어 생성 품질 확보 |
| 감정 분석 | HuggingFace Transformers | 로컬 감정 분류 추론 | LLM 호출 없이 6-class 분류, 비용 절감 |
| 벡터 검색 | Pinecone | RAG 법률 문서 검색 | 의미 기반 유사 문서 검색 지원 |
| 저장 | Google Cloud Storage | 분석 결과 보존 | 관리형 오브젝트 스토리지, 대시보드 연동 용이 |
| 알림 | Slack Webhook | 리포트 전송 | 협업 채널 즉시 전달, Block Kit 포맷 지원 |
| 대시보드 | Streamlit | 웹 시각화 | Python 기반 빠른 프로토타입 구현 가능 |
| 배포 | Docker + GCP Cloud Run | 컨테이너 기반 운영 | 환경 일관성 보장 및 서버리스 운영 지원 |

### 주요 라이브러리

| 라이브러리 | 버전 | 역할 |
|-----------|------|------|
| `langgraph` | ≥0.0.1 | Multi-Agent 워크플로우 오케스트레이션 |
| `langchain-openai` | ≥0.0.1 | LLM 연동 및 LCEL 체인 |
| `transformers` | ≥4.35 | HuggingFace 감성 분석 모델 |
| `torch` | ≥2.0 | 딥러닝 추론 (CPU) |
| `confluent-kafka` | ≥2.0.0 | Kafka Producer/Consumer |
| `pydantic` | ≥2.0.0 | 데이터 스키마 검증 |
| `networkx` | - | 인과관계 그래프 분석 |
| `pinecone` | - | 벡터 DB 클라이언트 |
| `mcp` | ≥0.5.0 | Model Context Protocol 클라이언트 |
| `playwright` | ≥1.40.0 | 커뮤니티 데이터 크롤링 |
| `pytrends` | ≥4.9.2 | Google Trends 데이터 수집 |
| `slack_sdk` | ≥3.19.0 | Slack 메시지 전송 |
| `streamlit` | - | 실시간 분석 대시보드 |
| `google-cloud-storage` | - | 분석 결과 저장 |
| `google-cloud-monitoring` | - | GCP 메트릭 전송 |
| `pytest` | ≥9.0.0 | 테스트 프레임워크 |

### 오픈소스

| 오픈소스 | 용도 | 링크 |
|---------|------|------|
| LangGraph | Multi-Agent 워크플로우 | https://github.com/langchain-ai/langgraph |
| LangChain | LLM 체인 | https://github.com/langchain-ai/langchain |
| Aerisbin/sentiment-agent-v1 | 팬덤 감성 분류 모델 | https://huggingface.co/Aerisbin/sentiment-agent-v1 |
| X MCP Server (오픈소스 래핑) | Twitter API v2 MCP | Node.js 기반 오픈소스 활용 |
| NetworkX | 그래프 분석 | https://networkx.org |
| Playwright | 웹 크롤링 | https://playwright.dev/python |
| Confluent Kafka | 스트리밍 인프라 | https://docs.confluent.io |
| Pinecone | 벡터 데이터베이스 | https://www.pinecone.io |

---

## 6. 디렉토리 구조

```
Dolpin/
├── src/
│   ├── agents/                     # AI 에이전트 구현
│   │   ├── spike_analyzer.py       # Z-Score + 가속도 기반 스파이크 탐지
│   │   ├── sentiment_agent.py      # 6클래스 팬덤 감성 분류
│   │   ├── causality_agent.py      # NetworkX 기반 인과관계 그래프 분석
│   │   ├── legalrag_agent.py       # 3단계 Agentic RAG 법적 리스크 분석
│   │   ├── playbook_agent.py       # 대응 전략 생성
│   │   └── demos/                  # 에이전트별 단독 실행 데모
│   │
│   ├── dolpin_langgraph/           # LangGraph 워크플로우
│   │   ├── graph.py                # StateGraph 정의 (11 노드, 3 라우터)
│   │   ├── nodes.py                # 각 노드 구현
│   │   ├── edges.py                # 조건부 라우팅 로직
│   │   └── state.py                # AnalysisState 공유 상태 정의
│   │
│   ├── pipeline/                   # 데이터 파이프라인
│   │   ├── collector.py            # UnifiedCollector (3개 소스 통합)
│   │   ├── kafka_producer.py       # Kafka 메시지 발행
│   │   ├── kafka_consumer.py       # Kafka 메시지 소비 + 파이프라인 실행
│   │   ├── transformer.py          # KafkaMessage → AnalysisState 변환
│   │   ├── result_store.py         # 분석 결과 저장
│   │   └── collectors/             # 소스별 수집기
│   │       ├── twitter.py          # Twitter MCP 클라이언트
│   │       ├── community.py        # 인스티즈 Playwright 크롤러
│   │       └── google_trends.py    # Google Trends pytrends
│   │
│   ├── server/                     # MCP 서버
│   │   ├── lexicon_server.py       # 렉시콘 분석 MCP 서버
│   │   ├── pinecone_server.py      # 법률 문서 검색 MCP 서버
│   │   ├── mcp_client.py           # MCP 클라이언트
│   │   └── embedder.py             # Pinecone 법률 지식베이스 임베딩
│   │
│   ├── schemas/
│   │   └── kafka_schema.py         # KafkaMessage Pydantic 스키마
│   │
│   └── integrations/
│       └── slack/                  # Slack 연동
│           ├── formatter.py        # 분석 결과 → Slack 블록 포맷
│           └── sender.py           # Slack API 전송
│
├── tests/                          # 테스트
│   ├── test_mcp_mock.py            # Mock 기반 전체 워크플로우 테스트
│   ├── test_kafka_to_spike.py      # Kafka → SpikeAnalyzer 통합 테스트
│   ├── test_spike_to_sentiment.py  # Spike → Sentiment 연결 테스트
│   ├── test_sentiment_to_causality.py
│   ├── test_legal_rag_agent.py     # LegalRAGAgent 단위 테스트
│   ├── test_causality_agent.py     # CausalityAgent 단위 테스트
│   ├── test_slack_integration.py   # Slack 전송 테스트
│   ├── test_pinecone_mcp.py        # Pinecone MCP 서버 테스트
│   ├── test_real_smoke.py          # 실제 외부 시스템 연결 스모크 테스트
│   ├── mcp_lexicon_mock.py         # 렉시콘 MCP Mock
│   └── outputs/                    # 테스트 결과 JSON 저장
│
├── consumer_main.py                # Cloud Run Consumer 진입점 (헬스체크 서버 포함)
├── dashboard.py                    # 모니터링 대시보드
├── legalsource.json                # 법률 지식베이스 원본 데이터
├── custom_lexicon.csv              # 팬덤 커스텀 렉시콘 데이터
├── Dockerfile.consumer             # Consumer Cloud Run 이미지
├── Dockerfile.collector            # Collector Cloud Run 이미지
├── Dockerfile.dashboard            # Dashboard 이미지
├── deploy.sh                       # 배포 스크립트
├── requirements.txt                # Python 의존성
├── pytest.ini                      # pytest 설정
└── .env                            # 환경변수 (로컬 실행용, git 제외)
```

---

## 7. How to Build

### 사전 준비

- Python 3.11 이상
- Node.js 18 이상 (Twitter MCP 서버 실행 시)
- Docker (선택, 컨테이너 실행 시)
- Confluent Cloud 계정 (Kafka 토픽 접근)
- OpenAI API Key
- Pinecone API Key
- Slack Bot Token

### 레포지토리 클론

```bash
git clone https://github.com/TEAM-AIDERS/Dolpin.git
cd Dolpin
```

### 가상환경 생성 및 활성화

```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Playwright 브라우저 설치 (커뮤니티 크롤러 사용 시)

```bash
python -m playwright install chromium
```

### Twitter MCP 서버 빌드 (Twitter 수집 사용 시)

```bash
cd src/server/x-v2-server
npm ci
npm run build
cd ../../..
```

---

## 8. How to Install

### 의존성 설치

```bash
# PyTorch CPU 버전 먼저 설치 (용량 최적화)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
pip install -r requirements.txt
```

### 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```bash
# Kafka (Confluent Cloud)
KAFKA_BOOTSTRAP_SERVERS=<your-cluster>.confluent.cloud:9092
KAFKA_API_KEY=<your-api-key>
KAFKA_API_SECRET=<your-api-secret>
KAFKA_TOPIC=dolpin-events

# 실행 모드 (REPLAY: 과거 메시지 재처리 / REALTIME: 최신 메시지만)
MODE=REALTIME

# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone
PINECONE_API_KEY=<your-pinecone-key>

# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C...

# HuggingFace (감성 분석 모델 다운로드)
HF_TOKEN=hf_...

# GCP (Cloud Monitoring 메트릭 전송, 선택)
GCP_PROJECT_ID=<your-project-id>
```

### 법률 지식베이스 Pinecone 적재 (최초 1회)

```bash
python -m src.server.embedder
```

실행하면 `legalsource.json`의 법률 문서가 Pinecone 인덱스 `dolpin-legal-v1`에 임베딩되어 업로드됩니다.

---

## 9. How to Test

### 테스트 실행

```bash
# 전체 테스트 실행
pytest tests/ -v

# Mock 기반 워크플로우 통합 테스트 (외부 시스템 불필요)
pytest tests/test_mcp_mock.py -v

# 에이전트 단위 테스트
pytest tests/test_causality_agent.py -v
pytest tests/test_legal_rag_agent.py -v

# Kafka → SpikeAnalyzer 연결 테스트
pytest tests/test_kafka_to_spike.py -v

# Slack 전송 테스트
pytest tests/test_slack_integration.py -v

# 실제 외부 시스템 연결 스모크 테스트 (API 키 필요)
pytest tests/test_real_smoke.py -v -m integration
```

### 테스트 구성

| 테스트 파일 | 설명 | 외부 시스템 필요 |
|------------|------|---------------|
| `test_mcp_mock.py` | 5개 시나리오로 전체 워크플로우 Mock 테스트 | X |
| `test_kafka_to_spike.py` | Kafka 메시지 → SpikeAnalyzer 분기 테스트 | X |
| `test_spike_to_sentiment.py` | Spike → 감성 분석 연결 테스트 | X |
| `test_sentiment_to_causality.py` | 감성 → 인과관계 연결 테스트 | X |
| `test_causality_agent.py` | NetworkX 그래프 분석 단위 테스트 | X |
| `test_legal_rag_agent.py` | LegalRAGAgent 리스크 판별 테스트 | X |
| `test_slack_integration.py` | Slack 메시지 포맷·전송 테스트 | Slack API |
| `test_pinecone_mcp.py` | Pinecone MCP 서버 검색 테스트 | Pinecone |
| `test_real_smoke.py` | 실제 파이프라인 E2E 스모크 테스트 | 전체 |

### 테스트 시나리오 (Mock)

`test_mcp_mock.py`는 다음 5개 시나리오를 검증합니다.

| 케이스 | 입력 | 예상 분기 |
|--------|------|-----------|
| 1 | 긍정 바이럴 (spike_rate=3.5, 응원 메시지) | Router1 통과 → Router2 Playbook fast-track |
| 2 | 저볼륨 (spike_rate=1.1) | Router1 skip → END |
| 3 | 보이콧 이슈 (spike_rate=4.0, 부정 키워드) | Router1 통과 → Causality → LegalRAG |
| 4 | 팬워 이슈 | Router1 통과 → Causality → LegalRAG |
| 5 | 밈 바이럴 | Router1 통과 → Causality → Amplification |

### 단독 실행 (로컬 파이프라인 테스트)

```bash
# 수집기 단독 실행 (키워드 직접 입력)
python -m src.pipeline.collector

# Consumer 단독 실행 (Kafka 메시지 수신 + 분석 파이프라인)
python consumer_main.py

# 에이전트 데모 실행
python src/agents/demos/run_causality_demo.py
python src/agents/demos/run_sentiment_adapter_demo.py
```

---

## 10. 샘플 데이터

### mock_result.json

`mock_result.json`: Mock 파이프라인 실행 결과 예시 파일입니다. 실제 외부 시스템 없이 전체 분석 결과의 구조를 확인할 수 있습니다.

### tests/outputs/

Mock 테스트 실행 시 케이스별 실제 결과가 저장됩니다.

- `mock_result_case1_positive_viral.json`: 긍정 바이럴 케이스 분석 결과
- `mock_result_case2_skip.json`: 저볼륨 스킵 케이스 결과

### legalsource.json

Pinecone에 적재되는 법률 지식베이스 원본입니다. 저작권법, 명예훼손, 초상권 등 K-pop 팬덤 이슈 관련 법률 조항 및 판례 데이터를 포함합니다.

### custom_lexicon.csv

팬덤 특화 커스텀 렉시콘 데이터입니다. 각 키워드에 `type`, `polarity`, `risk` 속성이 정의되어 있으며, MCP 렉시콘 서버에서 Actionability Score 산출에 활용됩니다.

| 카테고리 | 예시 term | type | 설명 |
|---------|-----------|------|------|
| 팬덤 행동 신호 | 총공 | boycott_action | 집단 행동 가능성 신호 |
| 팬덤 행동 신호 | 탈덕 | support_action | 팬덤 이탈 신호 |
| 감정 왜곡 표현 | 악개 | fanwar_target | 팬워 유발 대상 |
| 팬덤 맥락/서치 방지 | 성명문 | context_marker | 공식 입장 요구 신호 (risk: alert) |
| 긍정 신호 | 최애, 머글 | fandom_slang | 팬덤 내 일반 긍정 표현 |

`action_strength`(none/declaration/collective)와 `risk_flag`(none/watch/alert) 조합으로 가중치를 산출합니다.

---

## 11. 팀원

**18팀 AIDERS**

| 이름 | 역할 | 담당 구현 |
|------|------|-----------|
| 함하경 | 팀장 / 시스템 아키텍처 | Kafka 파이프라인, SpikeAnalyzerAgent, GCP 연동, 배포 |
| 박시원 | AI 에이전트 | SentimentAgent, LegalRAGAgent, PlaybookAgent, MCP 서버 |
| 강민서 | LangGraph 워크플로우 | StateGraph 설계, CausalityAgent, 노드/엣지 구현 |

---

## 🔑 환경변수 요약

| 변수 | 설명 | 필수 |
|------|------|------|
| `KAFKA_BOOTSTRAP_SERVERS` | Confluent Cloud 브로커 주소 | ✅ |
| `KAFKA_API_KEY` | Kafka SASL 인증 키 | ✅ |
| `KAFKA_API_SECRET` | Kafka SASL 인증 시크릿 | ✅ |
| `KAFKA_TOPIC` | 메시지 토픽명 | ✅ |
| `MODE` | `REPLAY` or `REALTIME` | ✅ |
| `OPENAI_API_KEY` | OpenAI API 키 | ✅ |
| `PINECONE_API_KEY` | Pinecone API 키 | ✅ |
| `SLACK_BOT_TOKEN` | Slack Bot OAuth 토큰 | ✅ |
| `SLACK_CHANNEL_ID` | Slack 알림 채널 ID | ✅ |
| `HF_TOKEN` | HuggingFace 모델 접근 토큰 | ✅ |
| `GCP_PROJECT_ID` | GCP 프로젝트 ID (메트릭 전송) | 선택 |
