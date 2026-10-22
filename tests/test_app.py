from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_runs_and_shows_the_four_tabs():
    at = AppTest.from_file(str(APP)).run(timeout=180)
    assert not at.exception
    assert at.title[0].value == "Wrongly blocked IP addresses"  # an empty app.py would fail here
    assert len(at.tabs) == 4
