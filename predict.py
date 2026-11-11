"""Get an indicative premium for one proposal using the saved model.

    python predict.py                      # runs the built-in example proposal
    python predict.py proposal.json        # your own proposal (see EXAMPLE below)
"""
import json
import sys

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.features import FEATURE_INPUT_COLS

EXAMPLE = {
    "proposal_date": "2026-01-15",
    "zone": "Zone A",
    "policy_type": "Family Floater",
    "eldest_member_age": 47,
    "gender": "Male",
    "adults_covered": 2,
    "children_covered": 2,
    "bmi": 27.4,
    "smoker": "Yes",
    "pre_existing_conditions": "Hypertension",
    "sum_insured_inr": 1000000,
    "co_pay_pct": 0,
    "policy_tenure_years": 1,
    "claims_last_3y": 0,
    "sales_channel": "Agent",
    "annual_health_checkup": "No",
    "occupation": "Salaried",
    "annual_income_lakh": 14.0,
}


def predict_premium(proposal, model_path=MODEL_DIR / "best_model.joblib"):
    """proposal: dict (one quote) or list of dicts. Returns annual premium(s) in rupees."""
    model = joblib.load(model_path)
    frame = pd.DataFrame([proposal] if isinstance(proposal, dict) else proposal)
    frame["proposal_date"] = pd.to_datetime(frame["proposal_date"])
    return model.predict(frame[FEATURE_INPUT_COLS])


if __name__ == "__main__":
    data = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else EXAMPLE
    for p in predict_premium(data):
        print(f"Indicative annual premium (before GST): ₹{p:,.0f}")
