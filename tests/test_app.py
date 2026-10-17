"""Smoke test: the dashboard renders every tab without raising."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "market_monthly.parquet").exists(),
                    reason="processed data not built")
def test_dashboard_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("📡 Job Market Radar")
    assert len(at.tabs) == 5
    assert any(e.label == "How this works" for e in at.expander)
