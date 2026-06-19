"""
Custom metric 기반 Cloud Run 자동 스케일러.
Cloud Monitoring에서 actionability_score를 읽어
dolpin-consumer의 min-instances를 동적으로 조절합니다.

실행 주기: Cloud Scheduler가 5분마다 Cloud Run Job으로 실행.

스케일링 정책:
  actionability_score >= 0.6  →  min_instances = 3  (고위험 스파이크)
  actionability_score >= 0.3  →  min_instances = 2  (중위험)
  actionability_score <  0.3  →  min_instances = 1  (기본)
"""

import logging
import os
import time

from google.cloud import monitoring_v3, run_v2
from google.protobuf import field_mask_pb2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.getenv("REGION", "asia-northeast3")
SERVICE_NAME = os.getenv("CONSUMER_SERVICE_NAME", "dolpin-consumer")
METRIC_TYPE = "custom.googleapis.com/dolpin/analysis/actionability_score"
LOOKBACK_SECONDS = 300  # 최근 5분간 최댓값 기준

# (임계값, min_instances) — 높은 쪽부터 체크
SCALE_TIERS: list[tuple[float, int]] = [
    (0.6, 3),
    (0.3, 2),
    (0.0, 1),
]
MAX_INSTANCES = 5


def _get_max_score(monitoring_client: monitoring_v3.MetricServiceClient) -> float:
    """최근 LOOKBACK_SECONDS 동안의 actionability_score 최댓값 반환.

    메트릭이 아직 한 번도 전송된 적 없으면 GCP가 404를 반환하므로
    해당 케이스는 데이터 없음(0.0)으로 처리한다.
    """
    from google.api_core.exceptions import NotFound

    now = time.time()
    interval = monitoring_v3.TimeInterval(
        {
            "end_time": {"seconds": int(now)},
            "start_time": {"seconds": int(now - LOOKBACK_SECONDS)},
        }
    )
    try:
        results = monitoring_client.list_time_series(
            request={
                "name": f"projects/{PROJECT_ID}",
                "filter": f'metric.type="{METRIC_TYPE}"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )
        max_score = 0.0
        for ts in results:
            for point in ts.points:
                max_score = max(max_score, point.value.double_value)
        return round(max_score, 4)
    except NotFound:
        logger.info("메트릭 없음 (아직 전송된 데이터 없음) → score=0.0으로 처리")
        return 0.0


def _target_min_instances(score: float) -> int:
    for threshold, instances in SCALE_TIERS:
        if score >= threshold:
            return instances
    return 1


def _get_current_min(run_client: run_v2.ServicesClient, service_path: str) -> int:
    service = run_client.get_service(name=service_path)
    return service.template.scaling.min_instance_count


def _update_scaling(
    run_client: run_v2.ServicesClient,
    service_path: str,
    min_instances: int,
) -> None:
    service = run_client.get_service(name=service_path)
    service.template.scaling.min_instance_count = min_instances
    service.template.scaling.max_instance_count = MAX_INSTANCES
    run_client.update_service(
        request={
            "service": service,
            "update_mask": field_mask_pb2.FieldMask(paths=["template.scaling"]),
        }
    ).result()  # 완료까지 블로킹


def main() -> None:
    monitoring_client = monitoring_v3.MetricServiceClient()
    run_client = run_v2.ServicesClient()
    service_path = (
        f"projects/{PROJECT_ID}/locations/{REGION}/services/{SERVICE_NAME}"
    )

    score = _get_max_score(monitoring_client)
    target = _target_min_instances(score)
    current = _get_current_min(run_client, service_path)

    logger.info(
        "actionability_score=%.4f  →  target min_instances=%d  (current=%d)",
        score,
        target,
        current,
    )

    if target != current:
        logger.info("스케일 조정: min_instances %d → %d", current, target)
        _update_scaling(run_client, service_path, target)
        logger.info("완료")
    else:
        logger.info("변경 없음 (min_instances=%d 유지)", current)


if __name__ == "__main__":
    main()
