# 🐬 Dolpin

> K-pop 기획사 홍보팀을 위한 Multi-Agent 기반 실시간 팬덤 이슈 분석 및 대응 시스템

K-pop 아티스트 관련 SNS·커뮤니티 데이터를 실시간으로 수집하고, 이슈 스파이크 탐지 → 감성 분류 → 인과관계 분석 → 법적 리스크 판단 → 대응 전략 생성까지의 전 과정을 Multi-Agent 파이프라인으로 자동화합니다. 분석 완료 후 1분 이내 Slack 알림과 Streamlit 대시보드로 결과를 전달합니다.

**팀명**: 18팀 AIDERS &nbsp;|&nbsp; **팀원**: 함하경, 박시원, 강민서

---

## 프로젝트 배경

K-pop에서 연예인 이미지는 매출과 직결됩니다. SNS 중심의 정보 확산 속도가 빨라지면서 팬덤 이슈에 얼마나 빠르게, 적절하게 대응하는가가 PR 기회(역주행·리브랜딩)와 리스크(이미지 하락·엔터사 주가 폭락)를 가릅니다.

| Problem | Solution |
|---------|----------|
| 수작업 모니터링의 한계 | 멀티 플랫폼 팬덤 데이터 모니터링 자동화 |
| 실시간 이슈 탐지 시스템 부재 | AI Agent 기반 실시간 감지 및 분석 |
| 범용 브랜드 모니터링 툴의 한계 | K-pop 팬덤 도메인 특화 PR 대응 의사결정 지원 |

---

## 아키텍처

> 아키텍처 다이어그램 이미지를 아래 위치에 삽입하세요.

<!-- ![Dolpin Architecture](assets/architecture.png) -->

```
[데이터 수집]
Twitter (X API MCP) ┐
인스티즈 (Playwright) ├─→ UnifiedCollector → KafkaProducer → Kafka Topic (3 partitions)
Google Trends       ┘

[분석 파이프라인]
KafkaConsumer → AnalysisState 변환 → LangGraph graph.ainvoke()

spike_analyzer → router1 ──(skip)──→ END
                    │
                    └──→ lexicon_lookup → sentiment → router2 ──→ playbook(fast)
                                                          │
                                                          └──→ causality → router3 ──→ legal_rag ──┐
                                                                               │                   ├──→ playbook → exec_brief → END
                                                                               └──→ amplification ─┘
[결과 전달]
Slack 알림 + Streamlit Dashboard + Google Cloud Storage
```

---

## 주요 기능

### Spike 기반 이슈 탐지
- Z-Score (최근 24시간 슬라이딩 윈도우 288개), 가속도 지표(EPS 1분 vs 5분), Actionability Score로 분석 우선순위 판별
- LLM 호출의 약 80% 사전 차단, false positive 약 25% 감소

### 6-Class 팬덤 감성 분류
- HuggingFace 파인튜닝 모델(`Aerisbin/sentiment-agent-v1`) 기반 로컬 추론
- `support / disappointment / boycott / meme / fanwar / neutral` 분류
- MCP 렉시콘 기반 2단계 오버라이드 적용

### 이슈 인과관계 분석
- NetworkX DiGraph로 확산 경로 모델링, 매개 중심성·연결 중심성으로 핵심 노드 추출
- 중심화 지수 기반 패턴 분류: `coordinated / viral / echo_chamber`

### 법적 리스크 분석 (3단계 Agentic RAG)
1. 18개 법적 키워드 + 컨텍스트 기반 Quick Risk Check
2. GPT-4o 플래너가 `법령 / 판례 / 내부정책` MCP 도구 선택, 최대 3회 반복 검색
3. Pydantic 구조화 출력 + `model_validator` 자동 보정

### 대응 전략 생성 및 결과 전달
- Rule-based 전략 분류 후 GPT-4-turbo로 상세 대응안 작성
- 대응 판단: 🔴 Crisis / 🟢 Opportunity / 🔵 Monitoring
- Slack Block Kit 포맷 자동 전송, Streamlit Dashboard 실시간 시각화

---

## 기술 스택

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=flat-square)
![Kafka](https://img.shields.io/badge/Confluent_Kafka-231F20?style=flat-square&logo=apachekafka)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-00B388?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4?style=flat-square&logo=googlecloud)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Slack](https://img.shields.io/badge/Slack-SDK-4A154B?style=flat-square&logo=slack)

| 계층 | 기술 | 용도 |
|------|------|------|
| 데이터 수집 | Playwright, pytrends, Twitter API v2 (MCP) | 인스티즈 크롤링, Google Trends, Twitter 수집 |
| 스트리밍 | Apache Kafka (Confluent Cloud) | 실시간 이벤트 스트리밍 |
| AI 프레임워크 | LangGraph, LangChain, OpenAI GPT-4o/turbo | Multi-Agent 파이프라인, LLM 체인 |
| 감정 분석 | HuggingFace Transformers | 로컬 6-class 감성 분류 |
| 벡터 검색 | Pinecone (`text-embedding-3-large`) | RAG 법률 문서 검색 |
| 저장 | Google Cloud Storage | 분석 결과 보존 |
| 알림·시각화 | Slack Webhook, Streamlit | 리포트 전송, 대시보드 |
| 배포 | Docker + GCP Cloud Run | 컨테이너 기반 서버리스 운영 |

---

## 디렉토리 구조

```
Dolpin/
├── src/
│   ├── agents/                     # AI 에이전트
│   │   ├── spike_analyzer.py       # Z-Score + 가속도 기반 스파이크 탐지
│   │   ├── sentiment_agent.py      # 6-class 팬덤 감성 분류
│   │   ├── causality_agent.py      # NetworkX 인과관계 그래프 분석
│   │   ├── legalrag_agent.py       # 3단계 Agentic RAG 법적 리스크 분석
│   │   ├── playbook_agent.py       # 대응 전략 생성
│   │   └── demos/                  # 에이전트별 단독 실행 데모
│   ├── dolpin_langgraph/           # LangGraph 워크플로우
│   │   ├── graph.py                # StateGraph 정의 (11 노드, 3 라우터)
│   │   ├── nodes.py                # 노드 구현
│   │   ├── edges.py                # 조건부 라우팅 로직
│   │   └── state.py                # AnalysisState 공유 상태
│   ├── pipeline/                   # 데이터 파이프라인
│   │   ├── collector.py            # UnifiedCollector (3개 소스 통합)
│   │   ├── kafka_producer.py       # Kafka 메시지 발행
│   │   ├── kafka_consumer.py       # Kafka 메시지 소비 + 파이프라인 실행
│   │   ├── transformer.py          # KafkaMessage → AnalysisState 변환
│   │   ├── result_store.py         # 분석 결과 저장
│   │   └── collectors/             # 소스별 수집기 (twitter / community / google_trends)
│   ├── server/                     # MCP 서버
│   │   ├── lexicon_server.py       # 렉시콘 분석 MCP 서버
│   │   ├── pinecone_server.py      # 법률 문서 검색 MCP 서버
│   │   ├── mcp_client.py           # MCP 클라이언트
│   │   └── embedder.py             # Pinecone 법률 지식베이스 임베딩
│   ├── schemas/
│   │   └── kafka_schema.py         # KafkaMessage Pydantic 스키마
│   └── integrations/slack/         # Slack 연동 (formatter / sender)
│
├── tests/                          # 테스트
│   ├── test_mcp_mock.py            # Mock 기반 전체 워크플로우 테스트
│   ├── test_kafka_to_spike.py
│   ├── test_spike_to_sentiment.py
│   ├── test_sentiment_to_causality.py
│   ├── test_legal_rag_agent.py
│   ├── test_causality_agent.py
│   ├── test_slack_integration.py
│   ├── test_real_smoke.py          # 실제 외부 시스템 E2E 스모크 테스트
│   └── outputs/                    # 테스트 결과 JSON
│
├── consumer_main.py                # Cloud Run Consumer 진입점
├── dashboard.py                    # Streamlit 모니터링 대시보드
├── legalsource.json                # 법률 지식베이스 원본 데이터
├── custom_lexicon.csv              # 팬덤 커스텀 렉시콘
├── mock_result.json                # Mock 파이프라인 결과 예시
├── Dockerfile.consumer             # Consumer Cloud Run 이미지
├── Dockerfile.collector            # Collector Cloud Run 이미지
├── Dockerfile.dashboard            # Dashboard 이미지
├── requirements.txt
└── pytest.ini
```

---

## How to Build

**사전 준비**: Python 3.11+, Node.js 18+ (Twitter MCP 사용 시), Docker (선택)

```bash
# 1. 레포지토리 클론
git clone https://github.com/TEAM-AIDERS/Dolpin.git
cd Dolpin

# 2. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Playwright 브라우저 설치 (커뮤니티 크롤러 사용 시)
python -m playwright install chromium

# 4. Twitter MCP 서버 빌드 (Twitter 수집 사용 시)
cd src/server/x-v2-server && npm ci && npm run build && cd ../../..
```

---

## How to Install

```bash
# PyTorch CPU 버전 먼저 설치
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
pip install -r requirements.txt
```

`.env` 파일을 프로젝트 루트에 생성합니다.

```bash
# Kafka (Confluent Cloud)
KAFKA_BOOTSTRAP_SERVERS=<your-cluster>.confluent.cloud:9092
KAFKA_API_KEY=<your-api-key>
KAFKA_API_SECRET=<your-api-secret>
KAFKA_TOPIC=dolpin-events
MODE=REALTIME                   # REPLAY: 과거 메시지 재처리 / REALTIME: 최신 메시지

# OpenAI / Pinecone / Slack / HuggingFace
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=<your-pinecone-key>
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C...
HF_TOKEN=hf_...

# GCP (Cloud Monitoring 메트릭 전송, 선택)
GCP_PROJECT_ID=<your-project-id>
```

법률 지식베이스 Pinecone 적재 (최초 1회):

```bash
python -m src.server.embedder
```

---

## How to Test

```bash
# 전체 테스트
pytest tests/ -v

# Mock 기반 워크플로우 통합 테스트 (외부 시스템 불필요)
pytest tests/test_mcp_mock.py -v

# 에이전트 단위 테스트
pytest tests/test_causality_agent.py tests/test_legal_rag_agent.py -v

# 실제 외부 시스템 E2E 스모크 테스트 (API 키 필요)
pytest tests/test_real_smoke.py -v -m integration
```

`test_mcp_mock.py`는 다음 5개 시나리오를 검증합니다.

| 케이스 | 입력 | 예상 결과 |
|--------|------|-----------|
| 1 | 긍정 바이럴 (spike_rate=3.5) | Router1 통과 → Playbook fast-track |
| 2 | 저볼륨 (spike_rate=1.1) | Router1 skip → END |
| 3 | 보이콧 이슈 | Causality → LegalRAG |
| 4 | 팬워 이슈 | Causality → LegalRAG |
| 5 | 밈 바이럴 | Causality → Amplification |

에이전트 단독 실행:

```bash
# 수집기 단독 실행
python -m src.pipeline.collector

# Consumer + 분석 파이프라인 실행
python consumer_main.py

# 대시보드 실행
streamlit run dashboard.py
```

---

## 샘플 데이터

| 파일 | 설명 |
|------|------|
| `mock_result.json` | Mock 파이프라인 분석 결과 예시 (외부 시스템 없이 결과 구조 확인 가능) |
| `tests/outputs/` | Mock 테스트 케이스별 실제 결과 JSON |
| `legalsource.json` | Pinecone 적재용 법률 지식베이스 원본 (저작권법·명예훼손·초상권 등) |
| `custom_lexicon.csv` | 팬덤 특화 커스텀 렉시콘. `type / polarity / risk` 속성으로 키워드 위험도를 정의하며, 팬덤 행동 신호(총공·탈덕), 감정 왜곡 표현(악개), 서치 방지(성명문), 긍정 신호(최애·머글) 4개 카테고리 포함 |

---

## 오픈소스

| 오픈소스 | 링크 |
|---------|------|
| LangGraph | https://github.com/langchain-ai/langgraph |
| LangChain | https://github.com/langchain-ai/langchain |
| Aerisbin/sentiment-agent-v1 | https://huggingface.co/Aerisbin/sentiment-agent-v1 |
| NetworkX | https://networkx.org |
| Playwright | https://playwright.dev/python |
| Pinecone | https://www.pinecone.io |
| Confluent Kafka | https://docs.confluent.io |
| Streamlit | https://streamlit.io |
