#!/usr/bin/env bash
# GCP Cloud Run 배포 스크립트
# 실행 전: gcloud auth login && gcloud auth configure-docker asia-northeast3-docker.pkg.dev
set -euo pipefail

# ============================================================
# 설정
# ============================================================
PROJECT_ID="dolpin-aiders"       # GCP 프로젝트 ID
REGION="asia-northeast3"               # 서울 리전 (Kafka 서버와 동일)
GCS_BUCKET="dolpin-aiders-results"     # GCS 버킷 이름 (결과 저장용)

REPO="dolpin"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"
SA_NAME="dolpin-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# ============================================================
# 1. API 활성화
# ============================================================
echo "=== GCP API 활성화 ==="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}"

# ============================================================
# 2. Artifact Registry 저장소 생성
# ============================================================
echo "=== Artifact Registry 생성 ==="
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet || echo "이미 존재함, 스킵"

# ============================================================
# 3. 서비스 계정 생성 및 권한 부여
# ============================================================
echo "=== 서비스 계정 설정 ==="
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="Dolpin Service Account" \
  --project="${PROJECT_ID}" \
  --quiet || echo "이미 존재함, 스킵"

for ROLE in \
  roles/storage.objectAdmin \
  roles/secretmanager.secretAccessor \
  roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet
done

# ============================================================
# 4. GCS 버킷 생성
# ============================================================
echo "=== GCS 버킷 생성 ==="
gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet || echo "이미 존재함, 스킵"

# ============================================================
# 5. Secret Manager에 시크릿 등록
# ============================================================
echo "=== Secret Manager 시크릿 등록 ==="

# .env 파일에서 읽어서 각 키를 Secret Manager에 저장
# 형식: KEY=VALUE (빈 줄, #주석 제외)
while IFS='=' read -r KEY VALUE; do
  [[ -z "$KEY" || "$KEY" == \#* ]] && continue
  VALUE=$(echo "$VALUE" | tr -d '\r' | xargs)  # 공백/CR 제거
  [[ -z "$VALUE" ]] && continue

  SECRET_NAME="dolpin_${KEY}"
  # 시크릿 생성 (이미 있으면 버전만 추가)
  if gcloud secrets describe "${SECRET_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "$VALUE" | gcloud secrets versions add "${SECRET_NAME}" \
      --data-file=- --project="${PROJECT_ID}" --quiet
  else
    echo "$VALUE" | gcloud secrets create "${SECRET_NAME}" \
      --data-file=- --project="${PROJECT_ID}" --quiet
  fi
  echo "  ✅ ${SECRET_NAME}"
done < .env

# GCS_BUCKET_NAME은 .env에 없으므로 별도 추가
echo "${GCS_BUCKET}" | gcloud secrets create "dolpin_GCS_BUCKET_NAME" \
  --data-file=- --project="${PROJECT_ID}" --quiet || \
echo "${GCS_BUCKET}" | gcloud secrets versions add "dolpin_GCS_BUCKET_NAME" \
  --data-file=- --project="${PROJECT_ID}" --quiet

# ============================================================
# 6. Docker 이미지 빌드 & 푸시
# ============================================================
echo "=== Docker 이미지 빌드 ==="

IMAGE_DASHBOARD="${REGISTRY}/dashboard:latest"
IMAGE_CONSUMER="${REGISTRY}/consumer:latest"

docker build -f Dockerfile.dashboard -t "${IMAGE_DASHBOARD}" .
docker push "${IMAGE_DASHBOARD}"

docker build -f Dockerfile.consumer -t "${IMAGE_CONSUMER}" .
docker push "${IMAGE_CONSUMER}"

# ============================================================
# 7. Cloud Run 배포 — 공통 시크릿 목록
# ============================================================
SECRETS="\
OPENAI_API_KEY=dolpin_OPENAI_API_KEY:latest,\
PINECONE_API_KEY=dolpin_PINECONE_API_KEY:latest,\
SLACK_BOT_TOKEN=dolpin_SLACK_BOT_TOKEN:latest,\
SLACK_CHANNEL_ID=dolpin_SLACK_CHANNEL_ID:latest,\
GCS_BUCKET_NAME=dolpin_GCS_BUCKET_NAME:latest"

DASHBOARD_SECRETS="${SECRETS}"

CONSUMER_SECRETS="${SECRETS},\
KAFKA_BOOTSTRAP_SERVERS=dolpin_KAFKA_BOOTSTRAP_SERVERS:latest,\
KAFKA_API_KEY=dolpin_KAFKA_API_KEY:latest,\
KAFKA_API_SECRET=dolpin_KAFKA_API_SECRET:latest,\
KAFKA_TOPIC=dolpin_KAFKA_TOPIC:latest,\
HF_TOKEN=dolpin_HF_TOKEN:latest,\
TWITTER_BEARER_TOKEN=dolpin_TWITTER_BEARER_TOKEN:latest,\
TWITTER_API_KEY=dolpin_TWITTER_API_KEY:latest,\
TWITTER_API_KEY_SECRET=dolpin_TWITTER_API_KEY_SECRET:latest,\
TWITTER_ACCESS_TOKEN=dolpin_TWITTER_ACCESS_TOKEN:latest,\
TWITTER_ACCESS_TOKEN_SECRET=dolpin_TWITTER_ACCESS_TOKEN_SECRET:latest,\
INSTIZ_ID=dolpin_INSTIZ_ID:latest,\
INSTIZ_PW=dolpin_INSTIZ_PW:latest,\
MODE=dolpin_MODE:latest"

# ============================================================
# 8. Dashboard 배포 (공개 접근)
# ============================================================
echo "=== Dashboard 배포 ==="
gcloud run deploy dolpin-dashboard \
  --image="${IMAGE_DASHBOARD}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --service-account="${SA_EMAIL}" \
  --set-secrets="${DASHBOARD_SECRETS}" \
  --set-env-vars="GCP_PROJECT=${PROJECT_ID}" \
  --memory=2Gi \
  --cpu=1 \
  --port=8080 \
  --allow-unauthenticated \
  --quiet

# ============================================================
# 9. Consumer 배포 (상시 실행, 비공개)
# ============================================================
echo "=== Consumer 배포 ==="
gcloud run deploy dolpin-consumer \
  --image="${IMAGE_CONSUMER}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --service-account="${SA_EMAIL}" \
  --set-secrets="${CONSUMER_SECRETS}" \
  --set-env-vars="GCP_PROJECT=${PROJECT_ID}" \
  --memory=4Gi \
  --cpu=2 \
  --port=8080 \
  --min-instances=1 \
  --max-instances=1 \
  --no-allow-unauthenticated \
  --quiet

# ============================================================
# 완료
# ============================================================
echo ""
echo "=== 배포 완료 ==="
DASHBOARD_URL=$(gcloud run services describe dolpin-dashboard \
  --region="${REGION}" --project="${PROJECT_ID}" \
  --format="value(status.url)")
echo "Dashboard URL: ${DASHBOARD_URL}"
