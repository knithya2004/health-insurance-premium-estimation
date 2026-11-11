# Health Insurance Premium Estimation

**TERV AIML · Track A - Regression · TFTPL310 (Trainer: Nithya)**

Estimate the annual premium (₹, before GST) an underwriter would set for an individual or family-floater health policy from the
proposal form, so agents get instant, consistent quotes and underwriters review only the unusual cases.

> **About the data.** The insurer's real file is not part of this repository, so `src/data_generation.py` builds a synthetic
> dataset with the same 21 columns, 6,242 rows, 21 cities / 3 zones, Jan 2024 - Dec 2025 dates, an April 2025 rate revision,
> and every data-quality problem in the brief. All numbers below come from that synthetic data, so they will differ from the
> reference solution in the handbook. To use the real data, put it at `data/raw/health_insurance_proposals.csv`; nothing else changes.

## Quick start

```bash
git clone https://github.com/<your-username>/terv-aiml-tftpl310-health-insurance-premium-estimation.git
cd terv-aiml-tftpl310-health-insurance-premium-estimation

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py               # raw CSV -> clean -> 14 charts -> models -> SHAP -> reports
python notebooks/build_notebook.py   # rebuilds and executes the notebook with all outputs
python predict.py                    # indicative quote for an example proposal
streamlit run app.py                 # interactive quote estimator & underwriting web application
python -m pytest -q                  # tests
```

## Project structure

```
app.py                                       interactive Streamlit web application (quote tool, governance, charts)
data/raw/health_insurance_proposals.csv      raw proposals (6,242 x 21)
data/processed/health_insurance_clean.csv    cleaned data
notebooks/health_insurance_premium_estimation.ipynb   executed notebook, all outputs
notebooks/build_notebook.py                  builds + executes the notebook
src/config.py                                paths and constants
src/data_generation.py                       synthetic raw data with the brief's data-quality issues
src/cleaning.py                              data-quality report and deterministic cleaning
src/features.py                              imputer, feature engineering, five model pipelines
src/modeling.py                              time-ordered split, 5-fold CV, metrics
src/plots.py                                 the 14 charts
run_pipeline.py                              runs everything end to end
predict.py                                   quote a single proposal with the saved model
outputs/figures/                             all 14 charts
outputs/reports/                             metrics.csv, summary.json and supporting tables
models/                                      best_model.joblib, gradient_boosting.joblib
tests/                                       pytest checks
```

## Method

1. **Data-quality audit**: missing values, duplicates, invalid ages and premiums, BMI / sum-insured unit typos, spelling variants.
2. **Cleaning**: drop duplicates and invalid rows, fix unit typos, map spellings and old city names, blank conditions → `Nil`.
   BMI (by gender × age band) and income (by occupation) are imputed **inside the pipeline**, using training months only.
3. **EDA**: charts 1-7, with one insight under each in the notebook.
4. **Features**: a flag per declared condition, obesity flag, smoker × age, family size, floater flag, log sum insured, age-over-40
   hinge, check-up flag, rate-revision flag. (Adults/children and the condition count are kept in the data but left out of the model
   matrix because they are exact sums of other columns, which would break linear-model coefficients.)
5. **Time-ordered split**: train Jan 2024 - Jun 2025 (4,807 rows), test Jul - Dec 2025 (1,318 rows).
6. **Models** (each in `Imputer → FeatureEngineer → ColumnTransformer → model`): Linear Regression (raw and log target), Decision
   Tree, Random Forest, Gradient Boosting; 5-fold CV on training months; best model picked on CV RMSE, never on the test set.
7. **Test evaluation**: MAE, MSE, RMSE, R², MAPE, share of quotes within ±10%, error by age band.
8. **SHAP `TreeExplainer`** on Gradient Boosting: global drivers, age × smoking dependence, one quote.

## Results (test period Jul - Dec 2025)

| Model | CV RMSE (₹) | Test MAE (₹) | Test RMSE (₹) | Test R² | Test MAPE | Within ±10% |
|---|---:|---:|---:|---:|---:|---:|
| **Linear Regression (log target)** | **1,430** | **842** | **2,091** | **0.995** | **2.7%** | **98.3%** |
| Gradient Boosting | 4,978 | 2,543 | 7,578 | 0.938 | 7.7% | 73.1% |
| Random Forest | 8,282 | 4,311 | 11,645 | 0.855 | 11.3% | 54.0% |
| Decision Tree | 11,330 | 6,166 | 13,983 | 0.790 | 18.2% | 34.1% |
| Linear Regression | 11,852 | 7,826 | 15,617 | 0.738 | 40.9% | 21.9% |

Full table (including MSE and CV spread): `outputs/reports/metrics.csv`; machine-readable summary: `outputs/reports/summary.json`.

**Success criteria** (for the best model, log-linear): R² ≥ 0.95 ✅ (0.995) · MAPE < 8% ✅ (2.7%) · ≥ 75% within ±10% ✅ (98.3%) ·
SHAP drivers consistent with underwriting practice ✅ · at least three recommendations ✅.

## Five insights

1. **Age drives the premium.** Correlation about 0.66 and the top SHAP feature; the effect is flat-ish to about 40, then climbs steeply.
2. **Smoking is an age-dependent loading.** Median premium for 18-30 smokers is within 10% of non-smokers, but at 61+ smokers pay about 60% more (₹81.7K vs ₹51.0K).
3. **Cover is priced sub-proportionally.** ₹1 Cr costs about 2.8x a ₹5L cover (20x the cover).
4. **Pricing is multiplicative**, so the log-linear model beats the tree models on accuracy and is also the easiest to audit. Its coefficients read as loadings: heart disease +34%, diabetes +18%, hypertension +12%, each past claim +7%, each 10 points of co-pay -9%, rate revision +10%.
5. **Demand is seasonal.** January - March brings about 39% of proposals (Section 80D tax deduction), peaking at 421 in March 2025.

## Three recommendations

1. **Instant indicative quotes.** Run the log-linear model behind the agent, bank-partner and web quote app, with the factor-by-factor breakdown shown to the customer. About 98% of test quotes were within ±10% of the underwriter.
2. **Underwriting review shortlist.** Auto-quote the rest and send to manual review only quotes where the log-linear and Gradient Boosting predictions disagree, or where the proposal is outside the training range (very high cover, age 61+ with several conditions). Underwriters then spend time only on the unusual cases.
3. **Consistency and governance.** Use the breakdowns to narrow the up-to-15% gap between underwriters, give the Appointed Actuary the loadings table to check against the filed rate card, and retrain after every rate revision (or at least every six months) while monitoring error by age band.

## Charts

All 14 are in `outputs/figures/`: 01 premium distribution · 02 proposals per month · 03 premium by age band and smoking · 04 premium by zone and policy type · 05 premium by declared conditions · 06 premium by sum insured · 07 correlation heatmap · 08 model comparison · 09 actual vs predicted · 10 error by age band · 11 log-linear loadings · 12 SHAP summary · 13 SHAP age and smoking dependence · 14 one quote explained.

## Limitations

- Results are from synthetic data generated with known pricing rules, which makes the log-linear model look especially good. Real underwriter decisions are noisier, so expect lower accuracy and re-check which model wins.
- The model is trained on past underwriter prices, so it reproduces their habits, including any bias. It gives an indicative quote, not a filed premium.
- GST is excluded, as in the target.
