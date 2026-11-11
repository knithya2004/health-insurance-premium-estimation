"""Central configuration: paths and constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "data" / "raw" / "health_insurance_proposals.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "health_insurance_clean.csv"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR = ROOT / "models"

SEED = 42
TARGET = "annual_premium_inr"

# Time-ordered split: train Jan 2024 - Jun 2025, test Jul - Dec 2025
SPLIT_DATE = "2025-07-01"
# Rate revision of about 10% came into force in April 2025
REVISION_DATE = "2025-04-01"

CONDITIONS = ["Diabetes", "Hypertension", "Heart Disease", "Thyroid", "Asthma"]


def ensure_dirs():
    for d in (RAW_CSV.parent, CLEAN_CSV.parent, FIG_DIR, REPORT_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
