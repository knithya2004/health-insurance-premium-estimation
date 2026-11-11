"""The 14 required charts. Each function returns a matplotlib Figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIG_DIR, TARGET
from .features import COND_FLAGS, NUM_COLS

sns.set_theme(style="whitegrid", context="notebook")
AGE_LABELS = ["18-30", "31-40", "41-50", "51-60", "61+"]
FIGSIZE = (9, 5.5)


def _add_age_band(df):
    df = df.copy()
    df["age_band"] = pd.cut(df["eldest_member_age"], [17, 30, 40, 50, 60, 120], labels=AGE_LABELS)
    return df


def _rupee_axis(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v/1000:,.0f}K")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ------------------------------------------------------------------ EDA 1-7
def chart01_premium_distribution(df):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(df[TARGET], bins=60, kde=True, ax=ax, color="#2a6fbb")
    med = df[TARGET].median()
    ax.axvline(med, color="crimson", ls="--", label=f"Median ₹{med:,.0f}")
    ax.axvline(60000, color="grey", ls=":", label="₹60K")
    ax.set(title="Distribution of annual premiums (right-skewed)", xlabel="Annual premium (₹)",
           ylabel="Proposals")
    ax.legend()
    return fig


def chart02_monthly_volume(df):
    m = df.set_index("proposal_date").resample("MS").size()
    colors = ["#e07b39" if d.month <= 3 else "#8aa6c1" for d in m.index]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(m.index.strftime("%b %y"), m.values, color=colors)
    share = 100 * m[m.index.month <= 3].sum() / m.sum()
    peak = m.idxmax()
    ax.annotate(f"Peak: {m.max()} ({peak.strftime('%b %Y')})", (m.index.get_loc(peak), m.max()),
                xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set(title=f"Proposals per month (Jan-Mar = {share:.0f}% of all proposals, Section 80D season)",
           ylabel="Proposals")
    ax.tick_params(axis="x", rotation=60)
    return fig


def chart03_premium_age_smoker(df):
    d = _add_age_band(df)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=d, x="age_band", y=TARGET, hue="smoker", hue_order=["No", "Yes"],
                showfliers=False, ax=ax, palette=["#5b9bd5", "#e4572e"])
    ax.set(title="Premium by age band and smoking status", xlabel="Eldest member age band",
           ylabel="Annual premium (₹)")
    _rupee_axis(ax)
    return fig


def chart04_premium_zone_policy(df):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=df, x="zone", y=TARGET, hue="policy_type", order=["Zone A", "Zone B", "Zone C"],
                showfliers=False, ax=ax, palette="Set2")
    ax.set(title="Premium by pricing zone and policy type", xlabel="", ylabel="Annual premium (₹)")
    _rupee_axis(ax)
    return fig


def chart05_premium_conditions(df):
    d = df[df["eldest_member_age"].between(41, 50) & (df["sum_insured_inr"] == 1_000_000)].copy()
    d["conditions"] = pd.cut(d["n_conditions"], [-1, 0, 1, 10], labels=["None", "One", "Two or more"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=d, x="conditions", y=TARGET, hue="conditions", legend=False, showfliers=False, ax=ax, palette="Blues")
    ax.set(title="Premium by declared conditions (age 41-50, ₹10L cover)",
           xlabel="Declared pre-existing conditions", ylabel="Annual premium (₹)")
    _rupee_axis(ax)
    return fig


def chart06_premium_sum_insured(df):
    d = df.copy()
    d["cover"] = (d["sum_insured_inr"] / 100000).map(lambda v: f"₹{v:g}L" if v < 100 else "₹1Cr")
    order = [f"₹{v:g}L" for v in (3, 5, 7.5, 10, 15, 20, 25, 50)] + ["₹1Cr"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=d, x="cover", y=TARGET, order=order, hue="cover", legend=False, showfliers=False, ax=ax, palette="viridis")
    ax.set(title="Premium by sum insured (cost rises far less than proportionally)",
           xlabel="Sum insured", ylabel="Annual premium (₹)")
    _rupee_axis(ax)
    return fig


def chart07_correlation_heatmap(df):
    cols = ["eldest_member_age", "n_conditions", "log_sum_insured", "family_size", "is_floater",
            "smoker_flag", "bmi", "claims_last_3y", "co_pay_pct", "policy_tenure_years",
            "annual_income_lakh", "rate_revision", TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap")
    return fig


# ------------------------------------------------------------ Modelling 8-11
def chart08_model_comparison(table):
    t = table.sort_values("test_RMSE")
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 0.2, t["cv_RMSE"], 0.4, label="5-fold CV RMSE (train months)", color="#8aa6c1")
    ax.bar(x + 0.2, t["test_RMSE"], 0.4, label="Test RMSE (Jul-Dec 2025)", color="#2a6fbb")
    for i, v in enumerate(t["test_RMSE"]):
        ax.text(i + 0.2, v, f"₹{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in t.index])
    ax.set(title="Model comparison (lower RMSE is better)", ylabel="RMSE (₹)")
    ax.legend()
    return fig


def chart09_actual_vs_predicted(y_test, pred, model_name, r2, mae):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(y_test, pred, s=10, alpha=0.5, color="#2a6fbb")
    lim = [0, max(y_test.max(), pred.max()) * 1.03]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set(xlim=lim, ylim=lim, xlabel="Underwriter premium (₹)", ylabel="Predicted premium (₹)",
           title=f"Actual vs predicted - {model_name}\nR² = {r2:.3f}, MAE = ₹{mae:,.0f}")
    ax.legend()
    return fig


def chart10_error_by_age(err_df):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=err_df, x="age_band", y="pct_error", showfliers=False, ax=ax, color="#9ecae1")
    ax.axhline(0, color="crimson", ls="--")
    ax.set(title="Percentage error by age band (test months)", xlabel="Eldest member age band",
           ylabel="(Predicted - actual) / actual, %")
    return fig


def chart11_loadings(log_pipe):
    """Turn log-linear coefficients (fitted on scaled inputs) into pricing multipliers."""
    prep = log_pipe.named_steps["prep"]
    reg = log_pipe.named_steps["model"].regressor_
    scaler = prep.named_transformers_["num"]
    coef = pd.Series(reg.coef_[: len(NUM_COLS)] / scaler.scale_, index=NUM_COLS)
    items = {
        "Heart disease": np.exp(coef["cond_heart_disease"]) - 1,
        "Diabetes": np.exp(coef["cond_diabetes"]) - 1,
        "Hypertension": np.exp(coef["cond_hypertension"]) - 1,
        "Asthma": np.exp(coef["cond_asthma"]) - 1,
        "Thyroid": np.exp(coef["cond_thyroid"]) - 1,
        "Obese (BMI ≥ 30)": np.exp(coef["obese_flag"]) - 1,
        "Each past claim": np.exp(coef["claims_last_3y"]) - 1,
        "Each extra tenure year": np.exp(coef["policy_tenure_years"]) - 1,
        "Each 10 pts of co-pay": np.exp(10 * coef["co_pay_pct"]) - 1,
        "Rate revision (Apr 2025)": np.exp(coef["rate_revision"]) - 1,
    }
    s = pd.Series(items).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(s.index, 100 * s.values, color=["#e4572e" if v > 0 else "#2a9d8f" for v in s.values])
    for i, v in enumerate(s.values):
        ax.text(100 * v + (0.7 if v > 0 else -0.7), i, f"{100*v:+.0f}%",
                va="center", ha="left" if v > 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Loadings learned by the log-linear model (as premium multipliers)",
           xlabel="Change in premium, %")
    return fig, s


# ----------------------------------------------------------------- SHAP 12-14
def chart12_shap_summary(shap_values, Xt, max_display=15):
    import shap
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(shap_values, Xt, max_display=max_display, show=False)
    plt.title("SHAP summary - Gradient Boosting")
    plt.tight_layout()
    return plt.gcf()


def chart13_shap_age_smoker(shap_values, Xt):
    cols = list(Xt.columns)
    age = Xt["eldest_member_age"].to_numpy()
    smoker = Xt["smoker_flag"].to_numpy()
    s_age = shap_values[:, cols.index("eldest_member_age")]
    s_smoke = shap_values[:, cols.index("smoker_flag")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for val, lab, col in [(0, "Non-smoker", "#5b9bd5"), (1, "Smoker", "#e4572e")]:
        m = smoker == val
        axes[0].scatter(age[m], s_age[m], s=9, alpha=0.5, color=col, label=lab)
    axes[0].set(title="SHAP of age: flat-ish to ~40, then steep", xlabel="Eldest member age",
                ylabel="SHAP value (₹)")
    axes[0].legend()
    m = smoker == 1
    axes[1].scatter(age[m], s_smoke[m], s=11, alpha=0.6, color="#e4572e")
    axes[1].set(title="Smoking loading grows with age (smokers only)", xlabel="Eldest member age",
                ylabel="SHAP value of smoker flag (₹)")
    fig.tight_layout()
    return fig


def pick_example_quote(X_test, y_test):
    """A 47-year-old smoker on a four-member family floater (closest match in the test months)."""
    cand = X_test[(X_test["smoker"] == "Yes") & (X_test["policy_type"] == "Family Floater")].copy()
    cand["size"] = cand["adults_covered"] + cand["children_covered"]
    cand["dist"] = (cand["eldest_member_age"] - 47).abs() + 3 * (cand["size"] - 4).abs()
    return cand["dist"].idxmin()


def chart14_one_quote(shap_row, base_value, predicted, actual, title_extra, top=9):
    s = shap_row.copy()
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    shown = s.iloc[:top]
    rest = s.iloc[top:].sum()
    shown = pd.concat([shown, pd.Series({"all other features": rest})]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(shown.index, shown.values, color=["#e4572e" if v > 0 else "#2a9d8f" for v in shown.values])
    for i, v in enumerate(shown.values):
        ax.text(v, i, f" {v:+,.0f}", va="center", ha="left" if v > 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Contribution to premium (₹)",
           title=(f"One quote explained: {title_extra}\n"
                  f"Average quote ₹{base_value:,.0f} → predicted ₹{predicted:,.0f} "
                  f"(underwriter ₹{actual:,.0f})"))
    fig.tight_layout()
    return fig


def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path
