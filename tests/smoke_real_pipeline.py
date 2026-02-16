#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


def _build_initial_state(messages: List[Dict[str, Any]], model_path: str, hard_fail_stages: List[str]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "spike_event": {
            "keyword": "smoke_test_keyword",
            "spike_rate": 3.2,
            "baseline": 100,
            "current_volume": 320,
            "detected_at": now,
            "time_window": "1h",
            "messages": messages,
            "raw_kafka_message_ids": ["smoke-1"],
        },
        "trace_id": "smoke-real-e2e",
        "workflow_start_time": now,
        "route1_decision": None,
        "route2_decision": None,
        "route3_decision": None,
        "positive_viral_detected": None,
        "spike_analysis": None,
        "lexicon_matches": None,
        "sentiment_result": None,
        "causality_result": None,
        "legal_risk": None,
        "amplification_summary": None,
        "playbook": None,
        "node_insights": {},
        "executive_brief": None,
        "error_logs": [],
        "skipped": False,
        "skip_reason": None,
        "sentiment_model_path": model_path,
        "hard_fail_stages": hard_fail_stages,
    }


def _env_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> int:
    model_path = os.getenv("SENTIMENT_MODEL_PATH", "models/sentiment_model")
    if not Path(model_path).exists():
        print(f"[FAIL] SENTIMENT_MODEL_PATH not found: {model_path}")
        return 2

    # Recommended default: fail fast on core analysis stages.
    hard_fail_stages = _env_list("SMOKE_HARD_FAIL_STAGES", "sentiment,causality")

    messages = [
        {
            "id": "smoke-msg-1",
            "source_message_id": "smoke-src-1",
            "text": "응원해 최고야",
            "timestamp": "2026-02-14T10:00:00Z",
            "source": "twitter",
            "author_id": "u1",
            "metrics": {"likes": 10, "retweets": 2},
            "is_anonymized": False,
            "detected_language": "ko",
        },
        {
            "id": "smoke-msg-2",
            "source_message_id": "smoke-src-2",
            "text": "논란은 있지만 지켜보자",
            "timestamp": "2026-02-14T10:05:00Z",
            "source": "instiz",
            "author_id": "u2",
            "metrics": {"likes": 5, "replies": 1},
            "is_anonymized": False,
            "detected_language": "ko",
        },
    ]

    from src.dolpin_langgraph.graph import compile_workflow

    app = compile_workflow()
    state = _build_initial_state(messages, model_path, hard_fail_stages)

    try:
        final_state = app.invoke(state)
    except Exception as e:
        print(f"[FAIL] workflow hard-failed: {e}")
        return 1

    if not final_state.get("executive_brief"):
        print("[FAIL] executive_brief missing")
        return 1

    print("[OK] real smoke passed")
    print("route1:", final_state.get("route1_decision"))
    print("route2:", final_state.get("route2_decision"))
    print("route3:", final_state.get("route3_decision"))
    print("errors:", len(final_state.get("error_logs", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

