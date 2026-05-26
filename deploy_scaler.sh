#!/usr/bin/env bash
# dolpin-scaler Cloud Run Job + Cloud Scheduler 배포
# 실행 전: gcloud auth login && gcloud auth configure-docker asia-northeast3-docker.pkg.dev
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-dolpin-aiders}"
REGION="${REGION:-asia-northeast3}"
REPO="${REPO:-dolpin}"
JOB_NAME="${JOB_NAME:-dolpin-scaler}"
SCHEDULER_NAME="${SCHEDULER_NAME:-dolpin-scaler-every-5m}"
SCHEDULE="${SCHEDULE:-*/5 * * * *}"
TIME_ZONE="${TIME_ZONE:-Asia/Seoul}"

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"
IMAGE_SCALER="${REGISTRY}/scaler:latest"
SA_NAME="${SA_NAME:-dolpin-sa}"
SA_EMAIL="${SA_EMAIL:-${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"

# ============================================================
# 1. 필요 API 활성화
# ============================================================
echo "=== Enable APIs ==="
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  monitoring.googleapis.com \
  --project="${PROJECT_ID}"

# ============================================================
# 2. 서비스 계정에 Cloud Run 관리 권한 부여
#    (스케일러가 dolpin-consumer의 min-instances를 수정하므로 필요)
# ============================================================
echo "=== Grant Cloud Run Admin role to SA ==="
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.admin" \
  --quiet

# ============================================================
# 3. 이미지 빌드 & 푸시 (Cloud Build)
# ============================================================
echo "=== Build and push scaler image ==="
cp Dockerfile.scaler Dockerfile
cleanup() { rm -f Dockerfile; }
trap cleanup EXIT
gcloud builds submit . --tag="${IMAGE_SCALER}" --project="${PROJECT_ID}"

# ============================================================
# 4. Cloud Run Job 생성 또는 업데이트
# ============================================================
echo "=== Create or update Cloud Run job ==="

JOB_FLAGS=(
  --image="${IMAGE_SCALER}"
  --region="${REGION}"
  --project="${PROJECT_ID}"
  --service-account="${SA_EMAIL}"
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},REGION=${REGION},CONSUMER_SERVICE_NAME=dolpin-consumer"
  --memory=512Mi
  --cpu=1
  --task-timeout=120s
  --max-retries=1
)

if gcloud run jobs describe "${JOB_NAME}" \
    --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud run jobs update "${JOB_NAME}" "${JOB_FLAGS[@]}"
else
  gcloud run jobs create "${JOB_NAME}" "${JOB_FLAGS[@]}"
fi

# ============================================================
# 5. Cloud Scheduler가 Job을 실행할 수 있도록 IAM 부여
# ============================================================
echo "=== Grant scheduler invoker permission ==="
gcloud run jobs add-iam-policy-binding "${JOB_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.invoker" \
  --quiet

# ============================================================
# 6. Cloud Scheduler 생성 또는 업데이트 (5분 주기)
# ============================================================
echo "=== Create or update Cloud Scheduler ==="
RUN_URI="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_NAME}:run"

if gcloud scheduler jobs describe "${SCHEDULER_NAME}" \
    --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "${SCHEDULER_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --schedule="${SCHEDULE}" \
    --time-zone="${TIME_ZONE}" \
    --uri="${RUN_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}"
else
  gcloud scheduler jobs create http "${SCHEDULER_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --schedule="${SCHEDULE}" \
    --time-zone="${TIME_ZONE}" \
    --uri="${RUN_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}"
fi

# ============================================================
# 완료
# ============================================================
echo ""
echo "=== Scaler 배포 완료 ==="
echo "Job:      ${JOB_NAME}"
echo "Image:    ${IMAGE_SCALER}"
echo "Schedule: ${SCHEDULE} (${TIME_ZONE})"
echo ""
echo "수동 실행 테스트:"
echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID} --wait"
echo ""
echo "로그 확인:"
echo "  gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}' --project=${PROJECT_ID} --limit=50"
