import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_proposals, data_quality_report
from src.data_generation import make_raw_dataset
from src.features import build_models
from src.modeling import regression_metrics, time_split


@pytest.fixture(scope="module")
def raw():
    return make_raw_dataset(n_total=2000, seed=7)


@pytest.fixture(scope="module")
def clean(raw):
    return clean_proposals(raw)


def test_raw_has_21_columns_and_issues(raw):
    assert raw.shape == (2000, 21)
    rep = data_quality_report(raw)["value"]
    assert rep["exact duplicate rows"] > 0
    assert rep["zero or negative premiums"] == 9
    assert rep["BMI with missed decimal (>100)"] == 30


def test_cleaning_fixes_everything(clean):
    assert not clean.duplicated().any()
    assert clean["eldest_member_age"].between(18, 100).all()
    assert (clean["annual_premium_inr"] > 0).all()
    assert (clean["sum_insured_inr"] >= 100000).all()
    assert clean["bmi"].dropna().between(10, 60).all()
    assert set(clean["smoker"]) == {"Yes", "No"}
    assert set(clean["policy_type"]) == {"Individual", "Family Floater"}
    assert set(clean["zone"]) == {"Zone A", "Zone B", "Zone C"}
    assert not clean["city"].isin(["Bombay", "Madras", "Bangalore", "Trichy"]).any()
    assert clean["pre_existing_conditions"].notna().all()


def test_time_split_has_no_overlap(clean):
    X_tr, _, X_te, _ = time_split(clean)
    assert X_tr["proposal_date"].max() < X_te["proposal_date"].min()


def test_log_linear_model_is_accurate(clean):
    X_tr, y_tr, X_te, y_te = time_split(clean)
    model = build_models()["Linear Regression (log target)"].fit(X_tr, y_tr)
    m = regression_metrics(y_te, model.predict(X_te))
    assert m["R2"] > 0.95
    assert m["MAPE_%"] < 8
    assert np.isfinite(model.predict(X_te)).all()
