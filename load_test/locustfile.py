"""
Dolpin 서비스 부하 테스트 (Locust)

설치:
  pip install locust

실행 예시:
  # 대시보드만 (비인증)
  locust -f load_test/locustfile.py --host=<DASHBOARD_URL> \\
    --users 50 --spawn-rate 5 --run-time 3m --headless

  # Consumer 헬스체크 포함 (GCP 인증 필요)
  export CONSUMER_URL=<CONSUMER_URL>
  export GCP_TOKEN=$(gcloud auth print-identity-token --audiences="${CONSUMER_URL}")
  locust -f load_test/locustfile.py --host=<DASHBOARD_URL> \\
    --users 50 --spawn-rate 5 --run-time 3m --headless

스케일링 관찰 방법:
  # 테스트 실행 중 별도 터미널에서 min-instances 변화 모니터링
  watch -n 10 "gcloud run services describe dolpin-consumer \\
    --region=asia-northeast3 --project=dolpin-aiders \\
    --format='value(spec.template.metadata.annotations)'"
"""

import os
import subprocess

from locust import HttpUser, between, events, task

CONSUMER_URL = os.getenv("CONSUMER_URL", "").rstrip("/")
GCP_TOKEN = os.getenv("GCP_TOKEN", "")


def _resolve_token() -> str:
    """GCP_TOKEN 환경변수 없으면 gcloud로 자동 발급."""
    if GCP_TOKEN:
        return GCP_TOKEN
    if not CONSUMER_URL:
        return ""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-identity-token", f"--audiences={CONSUMER_URL}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


@events.init.add_listener
def on_init(environment, **kwargs):
    print("\n" + "=" * 60)
    print("Dolpin Load Test")
    print(f"  Dashboard  : {environment.host}")
    print(f"  Consumer   : {CONSUMER_URL or '(skipped — CONSUMER_URL not set)'}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────
# 1. Dashboard 부하 (공개 접근, 전체 유저의 75%)
# ──────────────────────────────────────────────────────────
class DashboardUser(HttpUser):
    """Streamlit 대시보드 엔드포인트 부하 테스트."""
    wait_time = between(1, 3)
    weight = 3

    @task(4)
    def health(self):
        # Streamlit 헬스 엔드포인트
        with self.client.get("/_stcore/health", catch_response=True, name="dashboard/health") as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"health {r.status_code}")

    @task(1)
    def main_page(self):
        with self.client.get("/", catch_response=True, name="dashboard/") as r:
            if r.status_code in (200, 302):
                r.success()
            else:
                r.failure(f"main {r.status_code}")


# ──────────────────────────────────────────────────────────
# 2. Consumer 헬스체크 (IAM 인증, 전체 유저의 25%)
#    → 실제 Kafka 파이프라인 인스턴스 동시성 테스트
# ──────────────────────────────────────────────────────────
class ConsumerUser(HttpUser):
    """Consumer Cloud Run Service 헬스체크 동시성 테스트."""
    wait_time = between(2, 5)
    weight = 1
    # host는 --host 인자 대신 CONSUMER_URL로 재정의
    host = CONSUMER_URL or "http://localhost:8080"

    def on_start(self):
        self._token = _resolve_token()
        if not self._token:
            self._skip = True
            return
        self._skip = False
        self._headers = {"Authorization": f"Bearer {self._token}"}

    @task
    def health(self):
        if getattr(self, "_skip", True):
            return  # Consumer URL 없으면 스킵
        with self.client.get(
            "/",
            headers=self._headers,
            catch_response=True,
            name="consumer/health",
        ) as r:
            if r.status_code == 200:
                r.success()
            elif r.status_code == 403:
                r.failure("403 — GCP 토큰 만료 또는 권한 없음")
            else:
                r.failure(f"consumer health {r.status_code}")
