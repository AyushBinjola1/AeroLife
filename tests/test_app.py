"""Dashboard renders honest empty states, and replay works against a saved bundle."""

import os

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT, SPEC, small_cfg

PAGES = ["page_overview", "page_fleet", "page_sensors", "page_prediction", "page_maintenance", "page_evaluation"]


def _page(fn: str) -> AppTest:
    script = (f"import sys; sys.path.insert(0, {str(ROOT / 'app')!r})\n"
              f"import importlib, views\nimportlib.reload(views)\nviews.{fn}()\n")
    return AppTest.from_string(script, default_timeout=120)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, d in (("ARTIFACTS", "a"), ("CANONICAL", "c"), ("REPORTS", "r")):
        monkeypatch.setenv(f"AEROLIFE_{k}_DIR", str(tmp_path / d))
    import yaml
    from aerolife.config import load_config
    cfg = load_config()
    cfg["dataset"]["gate_status"] = "blocked"
    path = tmp_path / "blocked.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("AEROLIFE_CONFIG", str(path))
    return tmp_path


def test_main_entry_point_runs(env):
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception


@pytest.mark.parametrize("fn", PAGES)
def test_empty_states(env, fn):
    at = _page(fn).run()
    assert not at.exception
    if fn == "page_overview":
        assert any("has not been verified" in w.value for w in at.warning)
        assert any(m.label == "Trained model" and m.value == "no" for m in at.metric)
    else:
        assert any("unavailable" in i.value for i in at.info)


def test_replay_with_fixture_bundle(env, tables):
    """Uses the logic fixture + a 2-epoch bundle in a temp dir purely to exercise page code paths."""
    from aerolife.pipeline import run_experiment
    from aerolife.splits import explicit_split

    obs, units = tables
    (env / "c").mkdir()
    obs.to_csv(env / "c" / "observations.csv", index=False)
    units.to_csv(env / "c" / "units.csv", index=False)
    split = explicit_split(["u1", "u2", "u3", "u4"], ["u5", "u6"], ["u7", "u8"])
    run_experiment(small_cfg(), SPEC, obs, units, split, env / "a" / "v0")

    at = _page("page_prediction").run()
    assert not at.exception
    assert any("not a live" in c.value for c in at.caption)
    assert at.selectbox[0].options == ["u7", "u8"]           # only held-out units offered
    assert at.slider[0].value == int(obs[obs.unit_id == "u7"].time_index.max())
    assert any(m.label.startswith("Predicted RUL") for m in at.metric)
    at.slider[0].set_value(0).run()
    assert any("Insufficient history" in w.value for w in at.warning)
    # Neither prediction nor retrospective chart may extend past replay time.
    import json
    for chart in at.get("plotly_chart"):
        for trace in json.loads(chart.proto.spec)["data"]:
            values = trace["x"]
            if isinstance(values, dict):
                import base64
                import numpy as np
                values = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
            assert len(values) <= 1
    at.slider[0].set_value(30).run()
    assert not at.exception
    assert any(m.label.startswith("Predicted RUL (step)") for m in at.metric)

    for fn in ("page_fleet", "page_sensors", "page_maintenance"):
        at = _page(fn).run()
        assert not at.exception, fn
        if fn == "page_maintenance":
            summary = at.dataframe[0].value
            assert "Current status" in summary
            assert summary["Plan alert"].notna().all()
            assert summary["Critical alert"].notna().all()


@pytest.mark.parametrize("page", ["page_sensors", "page_prediction"])
def test_single_observation_does_not_create_invalid_slider(env, tables, page):
    obs, units = tables
    (env / "c").mkdir()
    obs.iloc[:1].to_csv(env / "c" / "observations.csv", index=False)
    units.iloc[:1].to_csv(env / "c" / "units.csv", index=False)
    if page == "page_prediction":
        # A minimal bundle stub is enough to exercise short-history UI, without fitting a model.
        script = (f"import sys; sys.path.insert(0, {str(ROOT / 'app')!r})\n"
                  "import importlib, views; importlib.reload(views)\n"
                  "from types import SimpleNamespace\n"
                  "views.get_bundle = lambda: SimpleNamespace(meta={'split': {'test': ['u1']}, 'dataset_name': 'fixture', 'model_version': '0', 'cell': 'lstm', 'selected_seed': 1, 'time_unit': 'step'}, seq_len=5)\n"
                  "views.predict_upto = lambda uid, t: views.get_data()[0].assign(status='insufficient history', predicted_rul=float('nan'))\n"
                  "views.page_prediction()")
        at = AppTest.from_string(script).run()
    else:
        at = _page(page).run()
    assert not at.exception
    assert not at.slider
