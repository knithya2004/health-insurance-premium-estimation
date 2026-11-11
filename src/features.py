"""Leakage-free preprocessing and model pipelines.

Pipeline = ClinicalImputer -> FeatureEngineer -> ColumnTransformer -> model
Everything that learns from data (imputation medians, scaler, encoder) is fitted
inside the pipeline, so cross-validation and the time-ordered test never see
information from the validation / test rows.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import CONDITIONS, REVISION_DATE, SEED

AGE_BINS = [0, 30, 40, 50, 60, 200]


def _flag_name(condition):
    return "cond_" + condition.lower().replace(" ", "_")


COND_FLAGS = [_flag_name(c) for c in CONDITIONS]

NUM_COLS = [
    "eldest_member_age", "age_over_40", "bmi", "annual_income_lakh",
    "family_size", "is_floater",
    "log_sum_insured", "co_pay_pct", "policy_tenure_years", "claims_last_3y",
    "smoker_flag", "smoker_x_age", "obese_flag",
    "checkup_flag", "rate_revision",
] + COND_FLAGS
# Note: adults/children (sum = family_size) and n_conditions (sum of the five flags)
# are engineered/kept in the data but left out of the model matrix because they are
# exact linear combinations of other columns, which would make the linear-model
# coefficients (and the loadings chart) meaningless.
CAT_COLS = ["zone", "gender", "sales_channel", "occupation"]
FEATURE_INPUT_COLS = [
    "proposal_date", "zone", "policy_type", "eldest_member_age", "gender",
    "adults_covered", "children_covered", "bmi", "smoker", "pre_existing_conditions",
    "sum_insured_inr", "co_pay_pct", "policy_tenure_years", "claims_last_3y",
    "sales_channel", "annual_health_checkup", "occupation", "annual_income_lakh",
]


class ClinicalImputer(BaseEstimator, TransformerMixin):
    """BMI by gender x age band, income by occupation (medians learned on fit data)."""

    def fit(self, X, y=None):
        X = self._with_band(X)
        self.bmi_median_ = X.groupby(["gender", "age_band"], observed=True)["bmi"].median()
        self.bmi_global_ = X["bmi"].median()
        self.income_median_ = X.groupby("occupation")["annual_income_lakh"].median()
        self.income_global_ = X["annual_income_lakh"].median()
        return self

    @staticmethod
    def _with_band(X):
        X = X.copy()
        X["age_band"] = pd.cut(X["eldest_member_age"], AGE_BINS, labels=False)
        return X

    def transform(self, X):
        X = self._with_band(X)
        keys = pd.MultiIndex.from_frame(X[["gender", "age_band"]])
        bmi_fill = pd.Series(self.bmi_median_.reindex(keys).to_numpy(), index=X.index)
        X["bmi"] = X["bmi"].fillna(bmi_fill).fillna(self.bmi_global_)
        inc_fill = X["occupation"].map(self.income_median_)
        X["annual_income_lakh"] = (X["annual_income_lakh"].fillna(inc_fill)
                                   .fillna(self.income_global_))
        return X.drop(columns="age_band")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless rating features built from the proposal form."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        age = X["eldest_member_age"]
        text = X["pre_existing_conditions"].fillna("Nil")
        for cond, flag in zip(CONDITIONS, COND_FLAGS):
            X[flag] = text.str.contains(cond, case=False, regex=False).astype(int)
        X["n_conditions"] = X[COND_FLAGS].sum(axis=1)
        X["smoker_flag"] = (X["smoker"] == "Yes").astype(int)
        X["smoker_x_age"] = X["smoker_flag"] * (age - 18)
        X["obese_flag"] = (X["bmi"] >= 30).astype(int)
        X["age_over_40"] = np.maximum(age - 40, 0)
        X["family_size"] = X["adults_covered"] + X["children_covered"]
        X["is_floater"] = (X["policy_type"] == "Family Floater").astype(int)
        X["log_sum_insured"] = np.log(X["sum_insured_inr"])
        X["checkup_flag"] = (X["annual_health_checkup"] == "Yes").astype(int)
        X["rate_revision"] = (pd.to_datetime(X["proposal_date"])
                              >= pd.Timestamp(REVISION_DATE)).astype(int)
        return X


def make_preprocessor(scale):
    """Scaled numerics + one-hot for linear models; raw numerics for trees."""
    num = StandardScaler() if scale else "passthrough"
    cat = OneHotEncoder(drop="first" if scale else None, handle_unknown="ignore",
                        sparse_output=False)
    return ColumnTransformer(
        [("num", num, NUM_COLS), ("cat", cat, CAT_COLS)],
        verbose_feature_names_out=False,
    )


def _pipe(model, scale):
    return Pipeline([
        ("impute", ClinicalImputer()),
        ("features", FeatureEngineer()),
        ("prep", make_preprocessor(scale)),
        ("model", model),
    ])


def build_models(seed=SEED):
    """The five models compared in the project (all in leakage-free pipelines)."""
    return {
        "Linear Regression": _pipe(LinearRegression(), scale=True),
        "Linear Regression (log target)": _pipe(
            TransformedTargetRegressor(LinearRegression(), func=np.log, inverse_func=np.exp),
            scale=True),
        "Decision Tree": _pipe(
            DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=seed),
            scale=False),
        "Random Forest": _pipe(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                  random_state=seed), scale=False),
        "Gradient Boosting": _pipe(
            GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3,
                                      subsample=0.8, random_state=seed), scale=False),
    }


def transformed_frame(pipe, X):
    """Run all steps except the final model and return a named DataFrame (for SHAP)."""
    Xt = pipe[:-1].transform(X)
    names = pipe.named_steps["prep"].get_feature_names_out()
    return pd.DataFrame(Xt, columns=names, index=X.index)
