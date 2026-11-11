"""Data-quality audit and deterministic cleaning.

Only steps that need no statistics from the data live here. Anything that learns
a value from the data (BMI and income imputation) happens inside the modelling
pipeline so it is fitted on the training months only (no leakage).
"""
import re

import numpy as np
import pandas as pd

from .config import TARGET

VALID_AGE = (18, 100)
OLD_CITY_NAMES = {
    "Bombay": "Mumbai", "Madras": "Chennai", "Bangalore": "Bengaluru",
    "Calcutta": "Kolkata", "Trichy": "Tiruchirappalli",
}


def data_quality_report(df):
    """Return a one-column DataFrame summarising every issue listed in the brief."""
    n = len(df)
    ages = pd.to_numeric(df["eldest_member_age"], errors="coerce")
    rows = {
        "rows": n,
        "columns": df.shape[1],
        "exact duplicate rows": int(df.duplicated().sum()),
        "duplicate % ": round(100 * df.duplicated().mean(), 2),
        "missing bmi %": round(100 * df["bmi"].isna().mean(), 2),
        "missing annual_income_lakh %": round(100 * df["annual_income_lakh"].isna().mean(), 2),
        "blank pre_existing_conditions %": round(100 * df["pre_existing_conditions"].isna().mean(), 2),
        "missing smoker %": round(100 * df["smoker"].isna().mean(), 2),
        f"invalid ages (<{VALID_AGE[0]} or >{VALID_AGE[1]})":
            int(((ages < VALID_AGE[0]) | (ages > VALID_AGE[1])).sum()),
        "zero or negative premiums": int((df[TARGET] <= 0).sum()),
        "BMI with missed decimal (>100)": int((df["bmi"] > 100).sum()),
        "sum insured entered in lakh (<1000)": int((df["sum_insured_inr"] < 1000).sum()),
        "distinct smoker spellings": df["smoker"].nunique(),
        "distinct policy_type spellings": df["policy_type"].nunique(),
        "distinct zone spellings": df["zone"].nunique(),
        "distinct city spellings": df["city"].nunique(),
    }
    return pd.DataFrame({"value": rows})


def _norm_smoker(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower().replace(" ", "").replace("-", "")
    if s in {"y", "yes", "smoker"}:
        return "Yes"
    if s in {"n", "no", "nonsmoker"}:
        return "No"
    return np.nan


def _norm_policy(v):
    s = str(v).strip().lower()
    if "floater" in s:
        return "Family Floater"
    if s.startswith("indiv") or s.startswith("single"):
        return "Individual"
    return np.nan


def _norm_zone(v):
    m = re.search(r"([abc])\s*$", str(v).strip().lower())
    return f"Zone {m.group(1).upper()}" if m else np.nan


def _norm_city(v):
    s = str(v).strip().title()
    return OLD_CITY_NAMES.get(s, s)


def clean_proposals(df_raw, verbose=False):
    """Apply every cleaning step from the brief and return the cleaned frame."""
    df = df_raw.copy()
    log = {"start": len(df)}

    df = df.drop_duplicates()                                     # 1 re-submitted forms
    log["after dropping duplicates"] = len(df)

    df["eldest_member_age"] = pd.to_numeric(df["eldest_member_age"], errors="coerce")
    ok_age = df["eldest_member_age"].between(*VALID_AGE)           # 2 invalid ages
    df = df[ok_age]
    log["after dropping invalid ages"] = len(df)

    df = df[df[TARGET] > 0]                                        # 3 placeholder premiums
    log["after dropping non-positive premiums"] = len(df)

    df["bmi"] = np.where(df["bmi"] > 100, df["bmi"] / 10, df["bmi"])      # 4 BMI 274 -> 27.4
    df.loc[~df["bmi"].between(10, 60), "bmi"] = np.nan                    # impossible -> impute later
    df["sum_insured_inr"] = np.where(df["sum_insured_inr"] < 1000,        # 5 lakh -> rupees
                                     df["sum_insured_inr"] * 100_000, df["sum_insured_inr"])

    df["smoker"] = df["smoker"].map(_norm_smoker).fillna("No")     # 6 spellings; missing -> No
    df["policy_type"] = df["policy_type"].map(_norm_policy)
    df["zone"] = df["zone"].map(_norm_zone)
    df["city"] = df["city"].map(_norm_city)
    df["pre_existing_conditions"] = (df["pre_existing_conditions"].fillna("Nil")
                                     .astype(str).str.strip().replace("", "Nil"))

    df["proposal_date"] = pd.to_datetime(df["proposal_date"])
    df = df.sort_values("proposal_date").reset_index(drop=True)
    df["sum_insured_inr"] = df["sum_insured_inr"].astype(int)
    df["eldest_member_age"] = df["eldest_member_age"].astype(int)
    log["final"] = len(df)

    if verbose:
        for k, v in log.items():
            print(f"{k:42s} {v:>7,}")
    df.attrs["cleaning_log"] = log
    return df
