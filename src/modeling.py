"""Time-ordered split, 5-fold CV model comparison and test evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from .config import SEED, SPLIT_DATE, TARGET
from .features import FEATURE_INPUT_COLS


def time_split(df, split_date=SPLIT_DATE):
    """Train on everything before split_date, test on the most recent months."""
    train = df[df["proposal_date"] < pd.Timestamp(split_date)]
    test = df[df["proposal_date"] >= pd.Timestamp(split_date)]
    return (train[FEATURE_INPUT_COLS], train[TARGET],
            test[FEATURE_INPUT_COLS], test[TARGET])


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    pct_err = np.abs(y_pred - y_true) / y_true
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": 100 * float(np.mean(pct_err)),
        "within_10pct_%": 100 * float(np.mean(pct_err <= 0.10)),
    }


def compare_models(models, X_train, y_train, X_test, y_test, cv_splits=5, seed=SEED):
    """5-fold CV on the training months, then fit on all training and score the test months."""
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    rows, fitted, preds = [], {}, {}
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=-1,
                             scoring={"rmse": "neg_root_mean_squared_error",
                                      "r2": "r2", "mae": "neg_mean_absolute_error"})
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name], preds[name] = pipe, pred
        row = {"model": name,
               "cv_RMSE": -res["test_rmse"].mean(),
               "cv_RMSE_std": res["test_rmse"].std(),
               "cv_R2": res["test_r2"].mean()}
        row.update({f"test_{k}": v for k, v in regression_metrics(y_test, pred).items()})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    return table, fitted, preds


def error_by_age_band(X_test, y_test, pred):
    bands = pd.cut(X_test["eldest_member_age"], [17, 30, 40, 50, 60, 120],
                   labels=["18-30", "31-40", "41-50", "51-60", "61+"])
    out = pd.DataFrame({"age_band": bands, "pct_error": 100 * (pred - y_test) / y_test})
    summary = out.groupby("age_band", observed=True)["pct_error"].agg(
        mean_pct_error="mean", median_pct_error="median",
        mean_abs_pct_error=lambda s: s.abs().mean())
    return out, summary
