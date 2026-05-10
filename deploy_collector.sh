#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-dolpin-aiders}"
REGION="${REGION:-asia-northeast3}"
REPO="${REPO:-dolpin}"
JOB_NAME="${JOB_NAME:-dolpin-collector}"
SCHEDULER_NAME="${SCHEDULER_NAME:-dolpin-collector-every-30m}"
SCHEDULE="${SCHEDULE:-*/30 * * * *}"
TIME_ZONE="${TIME_ZONE:-Asia/Seoul}"
KEYWORD="${KEYWORD:-}"

if [[ -z "${KEYWORD}" ]]; then
  echo "ERROR: Set KEYWORD before running this script."
  echo "Example: KEYWORD=\"아이브\" ./deploy_collector.sh"
  exit 1
fi

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"
IMAGE_COLLECTOR="${REGISTRY}/collector:latest"
SA_NAME="${SA_NAME:-dolpin-sa}"
SA_EMAIL="${SA_EMAIL:-${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"

COMMON_SECRETS="\
KAFKA_BOOTSTRAP_SERVERS=dolpin_KAFKA_BOOTSTRAP_SERVERS:latest,\
KAFKA_API_KEY=dolpin_KAFKA_API_KEY:latest,\
KAFKA_API_SECRET=dolpin_KAFKA_API_SECRET:latest,\
KAFKA_TOPIC=dolpin_KAFKA_TOPIC:latest,\
TWITTER_BEARER_TOKEN=dolpin_TWITTER_BEARER_TOKEN:latest,\
TWITTER_API_KEY=dolpin_TWITTER_API_KEY:latest,\
TWITTER_API_KEY_SECRET=dolpin_TWITTER_API_KEY_SECRET:latest,\
TWITTER_ACCESS_TOKEN=dolpin_TWITTER_ACCESS_TOKEN:latest,\
TWITTER_ACCESS_TOKEN_SECRET=dolpin_TWITTER_ACCESS_TOKEN_SECRET:latest,\
INSTIZ_ID=dolpin_INSTIZ_ID:latest,\
INSTIZ_PW=dolpin_INSTIZ_PW:latest,\
MODE=dolpin_MODE:latest"

echo "=== Enable required APIs ==="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com \
  --project="${PROJECT_ID}"

echo "=== Build and push collector image ==="
cp Dockerfile.collector Dockerfile
cleanup() {
  rm -f Dockerfile
}
trap cleanup EXIT
gcloud builds submit . --tag="${IMAGE_COLLECTOR}" --project="${PROJECT_ID}"

echo "=== Create or update Cloud Run job ==="
if gcloud run jobs describe "${JOB_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud run jobs update "${JOB_NAME}" \
    --image="${IMAGE_COLLECTOR}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --service-account="${SA_EMAIL}" \
    --command=python \
    --args=-m,src.pipeline.collector,"${KEYWORD}" \
    --set-secrets="${COMMON_SECRETS}" \
    --memory=2Gi \
    --cpu=1 \
    --task-timeout=900s \
    --max-retries=1
else
  gcloud run jobs create "${JOB_NAME}" \
    --image="${IMAGE_COLLECTOR}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --service-account="${SA_EMAIL}" \
    --command=python \
    --args=-m,src.pipeline.collector,"${KEYWORD}" \
    --set-secrets="${COMMON_SECRETS}" \
    --memory=2Gi \
    --cpu=1 \
    --task-timeout=900s \
    --max-retries=1
fi

echo "=== Grant scheduler permission to execute the job ==="
gcloud run jobs add-iam-policy-binding "${JOB_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.invoker" \
  --quiet

echo "=== Create or update scheduler job ==="
RUN_URI="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_NAME}:run"

if gcloud scheduler jobs describe "${SCHEDULER_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
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

echo ""
echo "Collector job: ${JOB_NAME}"
echo "Image: ${IMAGE_COLLECTOR}"
echo "Keyword: ${KEYWORD}"
echo "Schedule: ${SCHEDULE} (${TIME_ZONE})"
echo ""
echo "Test manually:"
echo "gcloud run jobs execute ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID} --wait"
