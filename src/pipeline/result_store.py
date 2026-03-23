"""
분석 결과를 GCS에 저장/로드하는 유틸리티.

환경변수:
    GCS_BUCKET_NAME: 저장할 GCS 버킷 이름
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_BLOB_NAME = "dolpin/latest_result.json"


def _bucket_name() -> Optional[str]:
    return os.getenv("GCS_BUCKET_NAME")


def save_result(result: dict) -> None:
    """최종 AnalysisState를 GCS에 저장."""
    bucket = _bucket_name()
    if not bucket:
        logger.warning("GCS_BUCKET_NAME이 설정되지 않아 결과를 저장하지 않습니다.")
        return

    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(_BLOB_NAME)
    blob.upload_from_string(
        json.dumps(result, ensure_ascii=False, default=str),
        content_type="application/json",
    )
    logger.info(f"✅ 결과 저장 완료: gs://{bucket}/{_BLOB_NAME}")


def load_result() -> Optional[dict]:
    """GCS에서 최신 AnalysisState를 로드. 없으면 None 반환."""
    bucket = _bucket_name()
    if not bucket:
        return None

    try:
        from google.cloud import storage

        client = storage.Client()
        blob = client.bucket(bucket).blob(_BLOB_NAME)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    except Exception as e:
        logger.error(f"GCS 읽기 실패: {e}")
        return None
