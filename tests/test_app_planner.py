"""Transfer Planner page (AppTest)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "5_Transfer_Planner.py")


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    return at


def test_planner_empty_data(data_root):
    at = _app()
    assert not at.exception
    assert any("update.py" in i.value for i in at.info)


def test_planner_loads_with_ui_data(ui_data):
    at = _app()
    assert not at.exception
    assert [b.label for b in at.button if b.label == "Solve"]


def test_planner_solve_shows_plan(ui_data):
    at = _app()
    next(b for b in at.button if b.label == "Solve").click()
    at.run()
    assert not at.exception
    assert not at.error
    assert any("legal" in s.value for s in at.success)
    assert any(h.value == "Plan" for h in at.subheader)
    assert at.dataframe


def test_planner_multi_gw_with_chip(ui_data):
    at = _app()
    at.slider(key="horizon").set_value(2).run()
    at.selectbox(key="chip1").select("bboost").run()
    next(b for b in at.button if b.label == "Solve").click()
    at.run()
    assert not at.exception
    assert not at.error
    assert any("legal" in s.value for s in at.success)
