"""Run the whole project end to end.

    python run_pipeline.py

Creates (if missing) the raw CSV, cleans it, draws all 14 charts, compares the five
models, explains Gradient Boosting with SHAP and saves metrics, summary and models.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap

from src import plots
from src.cleaning import clean_proposals, data_quality_report
from src.config import (CLEAN_CSV, MODEL_DIR, RAW_CSV, REPORT_DIR, SEED, SPLIT_DATE,
                        ensure_dirs)
from src.data_generation import write_raw_csv
from src.features import FeatureEngineer, build_models, transformed_frame
from src.modeling import compare_models, error_by_age_band, regression_metrics, time_split


def main():
    ensure_dirs()

    # ---- Milestone 1: data understanding & quality
    if not RAW_CSV.exists():
        print("Raw CSV not found - generating the synthetic dataset.")
        write_raw_csv()
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    quality = data_quality_report(raw)
    quality.to_csv(REPORT_DIR / "data_quality_report.csv")
    print(quality.to_string())

    # ---- Milestone 2: cleaning & EDA
    df = clean_proposals(raw, verbose=True)
    df.to_csv(CLEAN_CSV, index=False)
    eda = FeatureEngineer().transform(df)
    for fn, name in [
        (plots.chart01_premium_distribution, "01_premium_distribution.png"),
        (plots.chart02_monthly_volume, "02_proposals_per_month.png"),
        (plots.chart03_premium_age_smoker, "03_premium_by_age_smoking.png"),
        (plots.chart04_premium_zone_policy, "04_premium_by_zone_policy.png"),
        (plots.chart05_premium_conditions, "05_premium_by_conditions.png"),
        (plots.chart06_premium_sum_insured, "06_premium_by_sum_insured.png"),
        (plots.chart07_correlation_heatmap, "07_correlation_heatmap.png"),
    ]:
        plots.save_fig(fn(eda), name)

    # ---- Milestone 3: split + models
    X_train, y_train, X_test, y_test = time_split(df)
    print(f"\nTrain: {len(X_train):,} rows (before {SPLIT_DATE}) | Test: {len(X_test):,} rows")
    table, fitted, preds = compare_models(build_models(), X_train, y_train, X_test, y_test)
    best_name = table["cv_RMSE"].idxmin()           # chosen on CV RMSE, not on the test set
    print("\n", table.round(3).to_string())
    print(f"\nBest model by CV RMSE: {best_name}")

    # ---- Milestone 4: evaluation
    best_pred = preds[best_name]
    best_m = regression_metrics(y_test, best_pred)
    plots.save_fig(plots.chart08_model_comparison(table), "08_model_comparison.png")
    plots.save_fig(plots.chart09_actual_vs_predicted(
        y_test, best_pred, best_name, best_m["R2"], best_m["MAE"]), "09_actual_vs_predicted.png")
    err_df, err_summary = error_by_age_band(X_test, y_test, best_pred)
    plots.save_fig(plots.chart10_error_by_age(err_df), "10_error_by_age_band.png")
    err_summary.to_csv(REPORT_DIR / "error_by_age_band.csv")
    fig11, loadings = plots.chart11_loadings(fitted["Linear Regression (log target)"])
    plots.save_fig(fig11, "11_log_linear_loadings.png")
    loadings.rename("premium_change").to_csv(REPORT_DIR / "log_linear_loadings.csv")

    # SHAP on Gradient Boosting (explains the tree model, as per the brief)
    gb = fitted["Gradient Boosting"]
    Xt = transformed_frame(gb, X_test)
    explainer = shap.TreeExplainer(gb.named_steps["model"])
    sv = explainer.shap_values(Xt)
    base = float(np.ravel(explainer.expected_value)[0])
    plots.save_fig(plots.chart12_shap_summary(sv, Xt), "12_shap_summary.png")
    plots.save_fig(plots.chart13_shap_age_smoker(sv, Xt), "13_shap_age_smoker_dependence.png")
    idx = plots.pick_example_quote(X_test, y_test)
    pos = X_test.index.get_loc(idx)
    row = X_test.loc[idx]
    size = int(row["adults_covered"] + row["children_covered"])
    desc = (f"{int(row['eldest_member_age'])}-year-old {'smoker' if row['smoker']=='Yes' else 'non-smoker'}, "
            f"{row['policy_type']}, {size} members, ₹{row['sum_insured_inr']/100000:g}L cover")
    shap_row = pd.Series(sv[pos], index=Xt.columns)
    plots.save_fig(plots.chart14_one_quote(
        shap_row, base, float(preds["Gradient Boosting"][pos]), float(y_test.loc[idx]), desc),
        "14_one_quote_explained.png")
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns).sort_values(ascending=False)
    mean_abs.rename("mean_abs_shap_inr").to_csv(REPORT_DIR / "shap_importance.csv")

    # ---- Milestone 5: save everything
    table.round(4).to_csv(REPORT_DIR / "metrics.csv")
    summary = {
        "dataset": {"raw_rows": int(len(raw)), "clean_rows": int(len(df)),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                    "split_date": SPLIT_DATE, "seed": SEED},
        "best_model_by_cv_rmse": best_name,
        "best_model_test_metrics": {k: round(float(v), 4) for k, v in best_m.items()},
        "gradient_boosting_test_metrics": {
            k: round(float(v), 4) for k, v in regression_metrics(y_test, preds["Gradient Boosting"]).items()},
        "success_criteria": {
            "test_R2_>=_0.95": bool(best_m["R2"] >= 0.95),
            "test_MAPE_<_8%": bool(best_m["MAPE_%"] < 8),
            "share_within_10pct_>=_75%": bool(best_m["within_10pct_%"] >= 75),
        },
        "top_shap_features": [str(f) for f in mean_abs.index[:8]],
        "example_quote": {"description": desc,
                          "predicted_gb": round(float(preds["Gradient Boosting"][pos])),
                          "underwriter": round(float(y_test.loc[idx]))},
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
    joblib.dump(fitted["Gradient Boosting"], MODEL_DIR / "gradient_boosting.joblib")
    print("\nSaved metrics.csv, summary.json, 14 figures and models.")
    print(json.dumps(summary["success_criteria"], indent=2))


if __name__ == "__main__":
    main()
