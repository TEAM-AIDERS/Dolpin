import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_real_smoke():
    if os.getenv("RUN_REAL_SMOKE", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("Set RUN_REAL_SMOKE=1 to run real smoke test.")

    script = Path("tests/smoke_real_pipeline.py")
    assert script.exists(), "smoke script is missing"

    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"real smoke failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

