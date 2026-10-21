from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_runs_without_errors():
    at = AppTest.from_file(str(APP)).run(timeout=180)
    assert not at.exception