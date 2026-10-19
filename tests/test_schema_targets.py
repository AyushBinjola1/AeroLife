import numpy as np
import pandas as pd
import pytest

from aerolife.schema import is_forbidden_input, validate_observations, validate_units
from aerolife.targets import add_rul, censored_units


def test_valid_tables_pass(tables):
    obs, units = tables
    assert validate_observations(obs, ["ch_a", "ch_b"]).ok
    assert validate_units(units, obs).ok


def test_missing_required_and_channels(tables):
    obs, _ = tables
    assert not validate_observations(obs.drop(columns="time_value")).ok
    assert not validate_observations(obs, ["ch_missing"]).ok


def test_label_columns_are_flagged_and_dropped(tables):
    obs, _ = tables
    obs = obs.assign(RUL=1.0, failure_time=2.0, theta_hpc=0.1, health_index=0.5)
    rep = validate_observations(obs, ["ch_a"])
    assert rep.ok
    assert set(rep.dropped_columns) == {"RUL", "failure_time", "theta_hpc", "health_index"}
    assert not is_forbidden_input("ch_a")


def test_duplicates_and_non_monotonic_time(tables):
    obs, _ = tables
    assert not validate_observations(pd.concat([obs, obs.iloc[:1]])).ok
    bad = obs.copy()
    bad.loc[(bad.unit_id == "u1") & (bad.time_index == 5), "time_value"] = -1.0
    assert not validate_observations(bad).ok


def test_non_numeric_and_infinite_channels(tables):
    obs, _ = tables
    assert not validate_observations(obs.assign(ch_b="x")).ok
    assert not validate_observations(obs.assign(ch_b=np.inf)).ok


def test_units_rules(tables):
    obs, units = tables
    u = units.copy(); u.loc[u.unit_id == "u9", "failure_time"] = 99.0
    assert not validate_units(u, obs).ok  # censored unit must not get an invented life
    u = units.copy(); u.loc[u.unit_id == "u1", "failure_time"] = np.nan
    assert not validate_units(u, obs).ok
    u = units.copy(); u.loc[u.unit_id == "u1", "failure_time"] = 10.0
    assert not validate_units(u, obs).ok  # failure before last observation
    u = units.copy(); u.loc[u.unit_id == "u1", "include"] = False
    assert not validate_units(u, obs).ok  # exclusion needs a reason


def test_rul_alignment_and_censoring(tables):
    obs, units = tables
    lab = add_rul(obs, units)
    assert "u9" not in set(lab.unit_id) and censored_units(units) == ["u9"]
    u1 = lab[lab.unit_id == "u1"]
    np.testing.assert_allclose(u1.rul, 40.0 - u1.time_value)
    assert u1.rul.min() == 1.0  # last observation precedes documented failure


def test_cap_keeps_physical_rul(tables):
    obs, units = tables
    lab = add_rul(obs, units, cap=20)
    assert lab.rul_target.max() == 20 and lab.rul.max() > 20


def test_negative_rul_raises(tables):
    obs, units = tables
    units = units.copy(); units.loc[units.unit_id == "u1", "failure_time"] = 30.0
    with pytest.raises(ValueError):
        add_rul(obs, units)


@pytest.mark.parametrize("column,value", [("unit_id", None), ("unit_id", " "),
    ("time_value", np.inf), ("time_value", -1), ("time_index", -1)])
def test_invalid_identifiers_and_time_are_rejected(tables, column, value):
    obs, _ = tables
    bad = obs.copy()
    bad.loc[0, column] = value
    assert not validate_observations(bad).ok


def test_nullable_missing_index_rejected(tables):
    obs, _ = tables
    obs = obs.copy()
    obs["time_index"] = obs.time_index.astype("Int64")
    obs.loc[0, "time_index"] = pd.NA
    assert not validate_observations(obs).ok


def test_duplicate_column_names_rejected(tables):
    obs, _ = tables
    bad = pd.concat([obs, obs[["ch_a"]]], axis=1)
    assert not validate_observations(bad).ok
