"""Generate a synthetic proposals dataset that mirrors the project brief.

The real insurer file is not available, so this module builds a dataset with the
same 21 columns, 6,242 rows, 21 cities / 3 zones, Jan 2024 - Dec 2025 dates, an
April 2025 rate revision, and every data-quality problem listed in the brief
(missing values, duplicates, invalid ages, non-positive premiums, BMI and
sum-insured unit typos, inconsistent spellings and old city names).

If you have the real CSV, put it at data/raw/health_insurance_proposals.csv and
the rest of the pipeline works unchanged.
"""
import numpy as np
import pandas as pd

from .config import CONDITIONS, RAW_CSV, REVISION_DATE, SEED, ensure_dirs

CITY_ZONE = {
    # Zone A (metros)
    "Mumbai": "A", "Delhi": "A", "Bengaluru": "A", "Chennai": "A",
    "Kolkata": "A", "Hyderabad": "A", "Pune": "A",
    # Zone B
    "Ahmedabad": "B", "Jaipur": "B", "Lucknow": "B", "Chandigarh": "B",
    "Kochi": "B", "Coimbatore": "B", "Surat": "B",
    # Zone C
    "Nagpur": "C", "Indore": "C", "Patna": "C", "Bhopal": "C",
    "Tiruchirappalli": "C", "Madurai": "C", "Visakhapatnam": "C",
}
ZONE_SHARE = {"A": 0.55, "B": 0.27, "C": 0.18}

# Multipliers used by the simulated "underwriter"
ZONE_FACTOR = {"A": 1.00, "B": 0.87, "C": 0.75}
CONDITION_LOADING = {
    "Heart Disease": 1.34, "Diabetes": 1.18, "Hypertension": 1.12,
    "Thyroid": 1.05, "Asthma": 1.08,
}
CHANNEL_FACTOR = {"Agent": 1.00, "Bancassurance": 0.98, "Online": 0.95, "Broker": 1.00}
TARGET_MEDIAN = 17736.0


def _city_probs():
    cities = list(CITY_ZONE)
    p = np.array([ZONE_SHARE[CITY_ZONE[c]] /
                  sum(1 for z in CITY_ZONE.values() if z == CITY_ZONE[c]) for c in cities])
    return cities, p / p.sum()


def generate_clean_proposals(n, seed=SEED):
    """Return n clean, priced proposals (no data-quality issues yet)."""
    rng = np.random.default_rng(seed)

    # ---- dates: Jan-Mar season (Section 80D) ~38% of volume, 2025 slightly busier
    days = pd.date_range("2024-01-01", "2025-12-31")
    w = np.where(days.month <= 3, 1.87, 1.0) * np.where(days.year == 2025, 1.12, 1.0)
    dates = pd.to_datetime(np.sort(rng.choice(days, size=n, p=w / w.sum())))

    cities, p_city = _city_probs()
    city = rng.choice(cities, size=n, p=p_city)
    zone = np.array([CITY_ZONE[c] for c in city])

    # ---- eldest member age by band
    bands = [(18, 30), (31, 40), (41, 50), (51, 60), (61, 75)]
    band_idx = rng.choice(len(bands), size=n, p=[0.18, 0.27, 0.22, 0.17, 0.16])
    lo = np.array([bands[i][0] for i in band_idx])
    hi = np.array([bands[i][1] for i in band_idx])
    age = rng.integers(lo, hi + 1)

    # ---- policy structure
    floater = rng.random(n) < np.where(age < 60, 0.60, 0.35)
    adults = np.ones(n, dtype=int)
    children = np.zeros(n, dtype=int)
    two_adults = rng.random(n) < 0.70
    n_child = rng.choice([0, 1, 2, 3], size=n, p=[0.20, 0.30, 0.35, 0.15])
    old = age >= 55
    n_child = np.where(old & (rng.random(n) < 0.8), 0, n_child)
    two_adults = np.where(old, rng.random(n) < 0.9, two_adults)
    adults = np.where(floater, np.where(two_adults, 2, 1), 1)
    children = np.where(floater, n_child, 0)
    single_floater = floater & (adults == 1) & (children == 0)
    children = np.where(single_floater, 1, children)
    family_size = adults + children

    # ---- personal / health
    gender = rng.choice(["Male", "Female"], size=n, p=[0.62, 0.38])
    male = gender == "Male"
    bmi = np.round(np.clip(rng.normal(25.3 + 0.03 * (age - 40) - 0.4 * (~male), 4.3), 15, 45), 1)
    smoker = rng.random(n) < np.where(male, 0.18, 0.05)

    a = age - 18
    p_cond = {
        "Diabetes": 0.03 + 0.004 * a,
        "Hypertension": 0.04 + 0.0055 * a,
        "Heart Disease": 0.01 + 0.0017 * a,
        "Thyroid": 0.05 + 0.0008 * a + 0.05 * (~male),
        "Asthma": np.full(n, 0.04),
    }
    cond_flags = {c: rng.random(n) < np.clip(p_cond[c], 0, 0.9) for c in CONDITIONS}
    cond_text = np.array([
        ", ".join(c for c in CONDITIONS if cond_flags[c][i]) or "Nil" for i in range(n)
    ])

    # ---- cover and customer choices
    si_lakh = rng.choice([3, 5, 7.5, 10, 15, 20, 25, 50, 100], size=n,
                         p=[0.10, 0.20, 0.07, 0.25, 0.12, 0.10, 0.06, 0.07, 0.03])
    sum_insured = np.round(si_lakh * 100000).astype(int)
    co_pay = rng.choice([0, 10, 20], size=n, p=[0.60, 0.25, 0.15])
    tenure = rng.choice([1, 2, 3], size=n, p=[0.60, 0.20, 0.20])
    claims = np.minimum(rng.poisson(0.15 + 0.004 * a), 4)
    channel = rng.choice(list(CHANNEL_FACTOR), size=n, p=[0.45, 0.20, 0.25, 0.10])
    checkup = rng.random(n) < 0.35

    retired = (age >= 60) & (rng.random(n) < 0.6)
    occupation = np.where(
        retired, "Retired",
        rng.choice(["Salaried", "Self-employed", "Business"], size=n, p=[0.50, 0.25, 0.25]),
    )
    inc_median = pd.Series(occupation).map(
        {"Salaried": 9, "Self-employed": 12, "Business": 18, "Retired": 5}).to_numpy()
    income = np.round(np.clip(inc_median * rng.lognormal(0, 0.5, n), 2, 200), 1)

    # ---- the simulated underwriter: rate card x loadings x discounts
    age_f = np.exp(0.011 * (np.minimum(age, 40) - 18) + 0.06 * np.maximum(age - 40, 0))
    si_f = (sum_insured / 1_000_000) ** 0.33
    fam_f = np.where(floater, 1 + 0.5 * (family_size - 1), 1.0)
    zone_f = pd.Series(zone).map(ZONE_FACTOR).to_numpy()
    cond_f = np.ones(n)
    for c in CONDITIONS:
        cond_f = cond_f * np.where(cond_flags[c], CONDITION_LOADING[c], 1.0)
    smoke_f = 1 + smoker * (0.12 + 0.0075 * (age - 18))
    bmi_f = 1 + 0.06 * (bmi >= 30) + 0.06 * (bmi >= 35)
    claim_f = 1.07 ** claims
    copay_f = np.exp(-0.0094 * co_pay)
    tenure_f = 0.97 ** (tenure - 1)
    chan_f = pd.Series(channel).map(CHANNEL_FACTOR).to_numpy()
    check_f = np.where(checkup, 0.97, 1.0)
    rev_f = np.where(dates >= pd.Timestamp(REVISION_DATE), 1.10, 1.0)
    noise = rng.lognormal(0, 0.025, n)

    premium = (age_f * si_f * fam_f * zone_f * cond_f * smoke_f * bmi_f * claim_f
               * copay_f * tenure_f * chan_f * check_f * rev_f * noise)
    premium = premium * TARGET_MEDIAN / np.median(premium)

    df = pd.DataFrame({
        "proposal_id": [f"HI{i:06d}" for i in range(1, n + 1)],
        "proposal_date": dates.strftime("%Y-%m-%d"),
        "city": city,
        "zone": ["Zone " + z for z in zone],
        "policy_type": np.where(floater, "Family Floater", "Individual"),
        "eldest_member_age": age,
        "gender": gender,
        "adults_covered": adults,
        "children_covered": children,
        "bmi": bmi,
        "smoker": np.where(smoker, "Yes", "No"),
        "pre_existing_conditions": cond_text,
        "sum_insured_inr": sum_insured,
        "co_pay_pct": co_pay,
        "policy_tenure_years": tenure,
        "claims_last_3y": claims,
        "sales_channel": channel,
        "annual_health_checkup": np.where(checkup, "Yes", "No"),
        "occupation": occupation,
        "annual_income_lakh": income,
        "annual_premium_inr": np.round(premium, 0),
    })
    return df


def _variants(rng, series, mapping, p_canonical=0.70):
    """Replace canonical labels by messy spellings (canonical kept ~70% of the time)."""
    out = series.copy().astype(object)
    for canon, variants in mapping.items():
        mask = (series == canon).to_numpy()
        k = mask.sum()
        if k == 0:
            continue
        options = [canon] + variants
        probs = [p_canonical] + [(1 - p_canonical) / len(variants)] * len(variants)
        out[mask] = rng.choice(options, size=k, p=probs)
    return out


def make_raw_dataset(n_total=6242, seed=SEED):
    """Clean data + all the corruptions described in the brief."""
    n_dup = int(round(0.015 * n_total))          # ~1.5% duplicate rows
    n = n_total - n_dup
    df = generate_clean_proposals(n, seed)
    rng = np.random.default_rng(seed + 1)

    perm = rng.permutation(n)
    idx_age = perm[:14]                           # 14 invalid ages
    idx_prem = perm[14:23]                        # 9 zero/negative premiums
    idx_bmi = perm[23:53]                         # 30 BMI decimal typos
    used = set(perm[:53])
    eligible_si = [i for i in range(n)
                   if i not in used and df.at[i, "sum_insured_inr"] % 100000 == 0]
    idx_si = rng.choice(eligible_si, size=40, replace=False)   # 40 sums entered in lakh
    special = used | set(idx_si)
    pool = np.array([i for i in range(n) if i not in special])

    df.loc[idx_age, "eldest_member_age"] = np.resize([0, 2, 150, 199], 14)
    df.loc[idx_prem, "annual_premium_inr"] = np.resize([0, 0, 0, 0, 0, -999, -999, -999, -999], 9)
    df.loc[idx_bmi, "bmi"] = (df.loc[idx_bmi, "bmi"] * 10).round(0)
    df.loc[idx_si, "sum_insured_inr"] = (df.loc[idx_si, "sum_insured_inr"] // 100000).astype(int)

    # missing values (taken from rows not used for the typo cases)
    for col, share in [("bmi", 0.045), ("annual_income_lakh", 0.06),
                       ("pre_existing_conditions", 0.03), ("smoker", 0.015)]:
        rows = rng.choice(pool, size=int(round(share * n)), replace=False)
        df.loc[rows, col] = np.nan

    # inconsistent spellings
    df["smoker"] = _variants(rng, df["smoker"], {
        "Yes": ["Y", "yes", "YES", "Smoker"],
        "No": ["NO", "N", "no", "Non-smoker"],
    })
    df["policy_type"] = _variants(rng, df["policy_type"], {
        "Individual": ["Indiv.", "individual", "INDIVIDUAL"],
        "Family Floater": ["Floater", "family floater", "FAMILY FLOATER"],
    })
    df["zone"] = _variants(rng, df["zone"], {
        "Zone A": ["A", "zone a", "Zone-A"],
        "Zone B": ["B", "zone b", "Zone-B"],
        "Zone C": ["C", "zone c", "Zone-C"],
    })

    # old / upper-case city names
    old_names = {"Mumbai": "Bombay", "Chennai": "Madras", "Bengaluru": "Bangalore",
                 "Kolkata": "Calcutta", "Tiruchirappalli": "Trichy"}
    city = df["city"].astype(object)
    for new, old in old_names.items():
        m = (df["city"] == new).to_numpy() & (rng.random(n) < 0.35)
        city[m] = old
    upper = rng.random(n) < 0.03
    city[upper] = city[upper].str.upper()
    df["city"] = city

    # exact duplicates (re-submitted forms), then shuffle
    dup_rows = df.iloc[rng.choice(n, size=n_dup, replace=False)]
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def write_raw_csv(path=RAW_CSV, seed=SEED):
    ensure_dirs()
    df = make_raw_dataset(seed=seed)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = write_raw_csv()
    print(f"Wrote {len(out):,} rows x {out.shape[1]} columns to {RAW_CSV}")
