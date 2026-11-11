"""Build and execute notebooks/health_insurance_premium_estimation.ipynb.

    python notebooks/build_notebook.py

The notebook is generated from the cells below, then executed so that every output
(tables and charts) is stored inside the .ipynb file.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
OUT = HERE / "health_insurance_premium_estimation.ipynb"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Health Insurance Premium Estimation
**TERV AIML · Track A - Regression · TFTPL310 (Trainer: Nithya)**

Predict the final annual premium (₹, before GST) an underwriter would set from the proposal form,
tested on the most recent six months (Jul - Dec 2025) and explained factor by factor with SHAP.

| Milestone | Day | Sections below |
|---|---|---|
| 1 Data understanding & quality | 1 | 1 |
| 2 Cleaning & EDA | 2-3 | 2, 3 |
| 3 Feature engineering & modelling | 4-5 | 4, 5 |
| 4 Evaluation & explanation | 6 | 6, 7 |
| 5 Insights & GitHub | 7 | 8 |
""")

code("""
import os, sys, json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from IPython.display import display

%matplotlib inline
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

from src import plots
from src.config import RAW_CSV, CLEAN_CSV, REPORT_DIR, MODEL_DIR, SPLIT_DATE, TARGET, ensure_dirs
from src.data_generation import write_raw_csv
from src.cleaning import data_quality_report, clean_proposals
from src.features import FeatureEngineer, build_models, transformed_frame
from src.modeling import time_split, compare_models, regression_metrics, error_by_age_band

ensure_dirs()


def show(fig, name):
    plots.save_fig(fig, name)      # every chart is also saved to outputs/figures/
    display(fig)
    plt.close(fig)
""")

# ------------------------------------------------------------------ 1
md("""
## 1. Data understanding & quality (Milestone 1)
If `data/raw/health_insurance_proposals.csv` is missing, a synthetic dataset with the same schema, size
and data-quality problems as the brief is generated (see `src/data_generation.py`). Drop the real file in
that location and re-run to use it instead.
""")
code("""
if not RAW_CSV.exists():
    write_raw_csv()
raw = pd.read_csv(RAW_CSV)
print("Shape:", raw.shape)
raw.head()
""")
code("""
display(raw.dtypes.to_frame("dtype").T)
display(raw.describe().T.round(2))
""")
code("""
quality = data_quality_report(raw)
quality
""")

# ------------------------------------------------------------------ 2
md("""
## 2. Cleaning (Milestone 2)
Each choice and why:

| Issue | Treatment | Why |
|---|---|---|
| Exact duplicate rows (re-submitted forms) | Drop | The same proposal would otherwise be counted (and priced) twice and could leak between train and validation folds. |
| Invalid ages (0, 2, 150, 199) | Drop rows outside 18-100 | The eldest insured member is an adult; the true age is unknowable, and the rating age drives the premium, so guessing would inject large errors. Only 14 rows. |
| Zero / negative premiums | Drop | System placeholders, not real underwriter decisions; the target must be a real price. |
| BMI with missed decimal (274) | Divide by 10; anything still outside 10-60 becomes missing | Clear unit slip: 274 → 27.4. |
| Sum insured entered in lakh (10) | Multiply by 100,000 | Cover is ₹3L - ₹1Cr; values below ₹1,000 can only be lakh. |
| Spelling variants in `smoker`, `policy_type`, `zone` | Map to canonical labels | One category must have one label or the encoder would split it. |
| Old / upper-case city names (Bombay, Madras, Bangalore, Trichy, PUNE) | Map to current names | Same city, same pricing zone. |
| Blank `pre_existing_conditions` | Fill with `'Nil'` | The brief says blank means none declared. |
| Missing `smoker` (1.5%) | Fill with `'No'` | The majority class (about 85-90% of proposers); a constant, so it needs nothing learned from the data. |
| Missing `bmi` (4.5%), `annual_income_lakh` (6%) | **Impute inside the model pipeline**: BMI by gender × age band, income by occupation (medians) | Medians are learned on the training months only, so no information from validation or test rows leaks in. |
""")
code("""
df = clean_proposals(raw, verbose=True)
df.to_csv(CLEAN_CSV, index=False)
print()
print("Still missing after deterministic cleaning (imputed later, in the pipeline):")
print(df[["bmi", "annual_income_lakh"]].isna().sum().to_string())
print("Cities:", df["city"].nunique(), "| zones:", sorted(df["zone"].unique()))
""")

# ------------------------------------------------------------------ 3
md("""
## 3. Exploratory data analysis (charts 1-7)
Rating features are added first (condition flags, condition count, smoker × age, family size, log sum insured,
rate-revision flag) so the charts can use them.
""")
code("""
eda = FeatureEngineer().transform(df)
eda[["proposal_date", "eldest_member_age", "pre_existing_conditions", "n_conditions",
     "smoker_x_age", "family_size", "log_sum_insured", "rate_revision", TARGET]].head()
""")
code("""
show(plots.chart01_premium_distribution(eda), "01_premium_distribution.png")
print(f"Median premium: ₹{eda[TARGET].median():,.0f} | share above ₹60K: {(eda[TARGET] > 60000).mean():.1%}")
""")
md("**Insight 1.** Premiums are right-skewed: the median is about ₹17.7K, but older families with high cover and declared conditions form a long tail above ₹60K (about 9% of proposals). A model on raw rupees will be dominated by that tail, which is why the log premium is worth trying.")
code("""
fig = plots.chart02_monthly_volume(eda)
show(fig, "02_proposals_per_month.png")
""")
md("**Insight 2.** Volumes jump every January - March as customers buy for the Section 80D tax deduction: these three months bring about 39% of all proposals, peaking at 421 in March 2025. Underwriting capacity and the instant-quote tool matter most in this window.")
code("""
show(plots.chart03_premium_age_smoker(eda), "03_premium_by_age_smoking.png")
eda.assign(band=pd.cut(eda.eldest_member_age, [17, 30, 40, 50, 60, 120],
                       labels=["18-30", "31-40", "41-50", "51-60", "61+"])) \\
   .groupby(["band", "smoker"], observed=True)[TARGET].median().unstack().round(0)
""")
md("**Insight 3.** The median premium for non-smokers rises from about ₹9.7K (age 18-30) to ₹51K (61+). The smoker gap widens with age: at 61+ smokers pay a median ₹81.7K, about 60% more, while at 18-30 the gap is under 10%.")
code("""
show(plots.chart04_premium_zone_policy(eda), "04_premium_by_zone_policy.png")
eda.groupby(["zone", "policy_type"])[TARGET].median().unstack().round(0)
""")
md("**Insight 4.** A family floater in metro Zone A has a median premium of ₹22.2K against ₹16.8K in Zone C, and individual policies cost roughly 60% of floaters in every zone.")
code("""
show(plots.chart05_premium_conditions(eda), "05_premium_by_conditions.png")
""")
md("**Insight 5.** For proposers aged 41-50 with ₹10L cover, the median premium rises from about ₹16.3K (no condition) to ₹16.8K (one) and ₹21.1K (two or more, a small group). Family size and smoking explain the wide spread inside each box, so conditions alone are a modest driver compared with age.")
code("""
show(plots.chart06_premium_sum_insured(eda), "06_premium_by_sum_insured.png")
med = eda.groupby("sum_insured_inr")[TARGET].median()
print(f"₹1 Cr cover costs {med[10_000_000] / med[500_000]:.1f}x a ₹5L cover (20x the cover)")
""")
md("**Insight 6.** Higher cover costs more, but far less than proportionally: a ₹1 Cr cover costs only about 2.8 times a ₹5L cover. Customers are being offered a strong price incentive to buy more cover.")
code("""
show(plots.chart07_correlation_heatmap(eda), "07_correlation_heatmap.png")
""")
md("**Insight 7.** Age has by far the strongest correlation with the premium (about 0.66), followed by the number of declared conditions and sum insured (about 0.31 each). Co-pay and tenure are weak, negative factors, as expected for discounts.")

# ------------------------------------------------------------------ 4
md("""
## 4. Feature engineering & time-ordered split (Milestone 3)
Rating features built by `FeatureEngineer` (inside the pipeline): one flag per declared condition (5), condition count,
obesity flag (BMI ≥ 30), smoker flag, **smoker × age**, family size, floater flag, **log sum insured**, an age-over-40 hinge
(the age effect is flat-ish then steep), check-up flag and the **rate-revision flag** (proposals from 1 Apr 2025).

`adults_covered` / `children_covered` (they add up to family size) and the condition count (sum of the five flags) stay in the
data but are left out of the model matrix: they are exact linear combinations of other columns and would make the
linear-model loadings meaningless.

**Split by date, not at random:** train on Jan 2024 - Jun 2025, test on Jul - Dec 2025. This mimics real use (price future
proposals) and tests the model across the rate revision.
""")
code("""
X_train, y_train, X_test, y_test = time_split(df)
print(f"Train: {len(X_train):,} rows  {X_train.proposal_date.min():%b %Y} - {X_train.proposal_date.max():%b %Y}")
print(f"Test : {len(X_test):,} rows  {X_test.proposal_date.min():%b %Y} - {X_test.proposal_date.max():%b %Y}")
""")

# ------------------------------------------------------------------ 5
md("""
## 5. Models, pipelines and 5-fold cross-validation
Five models, each in a `Pipeline`: `ClinicalImputer → FeatureEngineer → ColumnTransformer → model`.
Scaled inputs and one-hot encoding (first level dropped) for the linear models, unscaled for the trees. The imputer, scaler
and encoder are fitted inside each CV fold, so there is no leakage. The best model is chosen on **CV RMSE** (training months
only), never on the test set.
""")
code("""
models = build_models()
models["Linear Regression (log target)"]
""")
code("""
table, fitted, preds = compare_models(models, X_train, y_train, X_test, y_test)
best_name = table["cv_RMSE"].idxmin()
print("Best model by 5-fold CV RMSE:", best_name)
table.round(3)
""")

# ------------------------------------------------------------------ 6
md("""
## 6. Test evaluation (Milestone 4)
MAE, MSE, RMSE, R² and MAPE on the unseen Jul - Dec 2025 proposals, plus the share of quotes within ±10% of the underwriter.
""")
code("""
test_cols = ["test_MAE", "test_MSE", "test_RMSE", "test_R2", "test_MAPE_%", "test_within_10pct_%"]
table[test_cols].sort_values("test_RMSE").round(3)
""")
code("""
show(plots.chart08_model_comparison(table), "08_model_comparison.png")
""")
md("**Insight 8.** Linear Regression on the log premium is the most accurate model, well ahead of Gradient Boosting. Plain Linear Regression and a single Decision Tree are the weakest. Premiums are built by multiplying rating factors, which is a straight line on the log scale; trees have to approximate this with steps and cannot extrapolate above the premiums they saw in training (relevant after the rate revision).")
code("""
best_pred = preds[best_name]
m = regression_metrics(y_test, best_pred)
show(plots.chart09_actual_vs_predicted(y_test, best_pred, best_name, m["R2"], m["MAE"]),
     "09_actual_vs_predicted.png")
print({k: round(v, 3) for k, v in m.items()})
""")
code("""
err_df, err_summary = error_by_age_band(X_test, y_test, best_pred)
show(plots.chart10_error_by_age(err_df), "10_error_by_age_band.png")
err_summary.round(2)
""")
md("**Insight 9.** Percentage errors are centred on zero in every age band (mean error within ±1%, mean absolute error about 2.5-3%), so the model is not biased against young or senior proposers.")
code("""
fig11, loadings = plots.chart11_loadings(fitted["Linear Regression (log target)"])
show(fig11, "11_log_linear_loadings.png")
(loadings * 100).round(1).rename("premium change %").to_frame()
""")
md("**Insight 10.** Read as multipliers, the log-linear coefficients match underwriting practice: heart disease about +34%, diabetes +18%, hypertension +12%, each past claim +7%, and each 10 points of co-pay about -9%. The Appointed Actuary can compare these directly with the filed rate card.")

# ------------------------------------------------------------------ 7
md("""
## 7. Explaining the pricing with SHAP (Gradient Boosting)
`TreeExplainer` explains the tree model on the test months: global drivers, the age dependence split by smoking
status, and one individual quote.
""")
code("""
gb = fitted["Gradient Boosting"]
Xt = transformed_frame(gb, X_test)
explainer = shap.TreeExplainer(gb.named_steps["model"])
sv = explainer.shap_values(Xt)
base = float(np.ravel(explainer.expected_value)[0])
show(plots.chart12_shap_summary(sv, Xt), "12_shap_summary.png")
pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns).sort_values(ascending=False).head(10).round(0)
""")
md("**Insight 11.** SHAP ranks age (the two age features together) far above everything else, followed by family size and sum insured; floater status, smoker × age, the rate revision and Zone A come next. This agrees with the EDA and with underwriting practice.")
code("""
show(plots.chart13_shap_age_smoker(sv, Xt), "13_shap_age_smoker_dependence.png")
""")
md("**Insight 12.** The age effect is flat-ish until about 40 and then climbs steeply. For smokers, the smoking loading grows steadily with age.")
code("""
idx = plots.pick_example_quote(X_test, y_test)
pos = X_test.index.get_loc(idx)
row = X_test.loc[idx]
size = int(row.adults_covered + row.children_covered)
desc = (f"{int(row.eldest_member_age)}-year-old smoker, {row.policy_type}, {size} members, "
        f"₹{row.sum_insured_inr / 100000:g}L cover")
pred_gb = float(preds["Gradient Boosting"][pos])
show(plots.chart14_one_quote(pd.Series(sv[pos], index=Xt.columns), base, pred_gb,
                             float(y_test.loc[idx]), desc), "14_one_quote_explained.png")
""")
md("**Insight 13.** For a single quote the bars add up exactly from the average quote to the prediction, so an agent can tell the customer: 'the four family members and your smoking at this age add the most; the age pattern and cover level are priced in too'. Underwriters can use the same view to review a quote that looks odd.")

# ------------------------------------------------------------------ 8
md("## 8. Save outputs and conclusions (Milestone 5)")
code("""
table.round(4).to_csv(REPORT_DIR / "metrics.csv")
joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
joblib.dump(fitted["Gradient Boosting"], MODEL_DIR / "gradient_boosting.joblib")
summary_path = REPORT_DIR / "summary.json"
print(summary_path.read_text() if summary_path.exists() else "run run_pipeline.py to create summary.json")
""")
md("""
### Five insights
1. **Age drives the premium** (correlation about 0.66, top SHAP feature): flat-ish to about 40, then a steep climb.
2. **Smoking is an age-dependent loading**: small at 18-30, about +60% on the median at 61+.
3. **Cover is priced sub-proportionally**: 20x the cover costs about 2.8x the premium.
4. **Pricing is multiplicative**, so a log-linear model beats the tree models on accuracy *and* is the easiest to audit against the rate card.
5. **Demand is seasonal**: Jan - Mar brings about 39% of proposals (Section 80D), so quote speed matters most then.

### Three recommendations
1. **Instant indicative quotes.** Put the log-linear model behind the agent / bank-partner / web quote app; about 98% of test quotes were within ±10% of the underwriter.
2. **Underwriting review shortlist.** Auto-quote the rest and send to manual review only quotes where the log-linear and Gradient Boosting predictions disagree, or where the customer is outside the training range (very high cover, age 61+ with several conditions).
3. **Consistency and governance.** Give underwriters the factor-by-factor breakdown to cut the up-to-15% gap between underwriters, retrain after every rate revision or at least every six months, and monitor error by age band.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}

if __name__ == "__main__":
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbf.write(nb, OUT)
    print("Wrote executed notebook:", OUT)
