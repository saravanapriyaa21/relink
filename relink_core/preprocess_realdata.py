# ==============================================================
# ReLink — Real Data Preprocessing (Robust NCRB + Census Merger)
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz import process, fuzz
import warnings, os
warnings.filterwarnings("ignore")

print("🚀 Starting ReLink Real Data Preprocessing (Stable + Hybrid Risk Model)...")

# -------------------- 1. Load --------------------
ncrb_path = "../data/NCRB_District_Table_1.10.csv"
census_path = "../data/india_districts_census_2011.csv"
out_path = "../data/district_risk_2022.csv"
unmatched_path = "../data/unmatched_districts.csv"

ncrb = pd.read_csv(ncrb_path)
census = pd.read_csv(census_path)

print(f"📊 NCRB rows: {len(ncrb)}, Census rows: {len(census)}")

# -------------------- 2. Clean column names --------------------
ncrb.columns = ncrb.columns.str.strip().str.lower()
census.columns = census.columns.str.strip().str.lower()

if not {"district name", "state name"}.issubset(census.columns):
    raise SystemExit("❌ Your census file must have 'District name' and 'State name' columns.")

# -------------------- 3. Clean names --------------------
def clean_name(x):
    if pd.isna(x): return ""
    x = str(x).lower().strip()
    for rem in ["district", "dt.", "(urban)", "(rural)", "&", "and"]:
        x = x.replace(rem, "")
    return " ".join(x.split())

ncrb["district"] = ncrb["district"].astype(str).apply(clean_name)
ncrb["state"] = ncrb["state/ut"].astype(str).apply(clean_name)
census["district"] = census["district name"].astype(str).apply(clean_name)
census["state"] = census["state name"].astype(str).apply(clean_name)

# -------------------- 4. Compute gender/child totals --------------------
def safe_sum(df, keyword):
    cols = [c for c in df.columns if keyword in c]
    return df[cols].fillna(0).sum(axis=1) if cols else 0

ncrb["male_total"] = safe_sum(ncrb, "male -")
ncrb["female_total"] = safe_sum(ncrb, "female -")
child_cols = [c for c in ncrb.columns if any(k in c for k in ["below 12", "below 16", "below 18", "children"])]
ncrb["child_total"] = ncrb[child_cols].fillna(0).sum(axis=1) if child_cols else 0
ncrb["total_missing"] = ncrb["male_total"] + ncrb["female_total"]

print("🧮 Computed gender & child totals successfully.")

# -------------------- 5. Census ratios --------------------
for col in ["population", "female", "literate", "workers"]:
    census[col] = census[col].replace(0, np.nan).fillna(census[col].median())

census["female_ratio"] = census["female"] / census["population"]
census["literacy_rate"] = census["literate"] / census["population"]
census["workers_ratio"] = census["workers"] / census["population"]

# -------------------- 6. Fuzzy Matching --------------------
def normalize_name(name):
    if pd.isna(name): return ""
    name = str(name).lower().strip()
    replace_map = {
        "bengaluru": "bangalore", "bengaluru urban": "bangalore urban",
        "bengaluru rural": "bangalore rural",
        "south twenty four": "south 24", "north twenty four": "north 24",
        "gurugram": "gurgaon", "ahmadabad": "ahmedabad",
        "24 pgs": "24 parganas", "andaman & nicobar": "andaman and nicobar",
        "delhi": "nct of delhi"
    }
    for k, v in replace_map.items():
        if k in name:
            name = name.replace(k, v)
    for rem in ["district", "dt", "(urban)", "(rural)", "city"]:
        name = name.replace(rem, "")
    name = name.replace("&", "and")
    return " ".join(name.split())

ncrb["district"] = ncrb["district"].apply(normalize_name)
census["district"] = census["district"].apply(normalize_name)

def best_match_adv(district, state, census_df):
    if not isinstance(district, str) or not district.strip():
        return None

    subset = census_df[census_df["state"].str.contains(state[:6], na=False)]
    choices = subset["district"].tolist() if len(subset) else census_df["district"].tolist()

    def safe_extract_one(q, choices, scorer):
        try:
            res = process.extractOne(q, choices, scorer=scorer)
            if res is None:
                return (None, 0)
            elif len(res) == 3:
                match, score, _ = res
            else:
                match, score = res
            return (match, score)
        except Exception:
            return (None, 0)

    best, score = safe_extract_one(district, choices, fuzz.token_sort_ratio)
    if score < 80:
        alt, alt_score = safe_extract_one(district, choices, fuzz.partial_ratio)
        if alt_score > score:
            best, score = alt, alt_score
    if score < 70:
        alt, alt_score = safe_extract_one(district, choices, fuzz.ratio)
        if alt_score > score:
            best, score = alt, alt_score

    return best if best and score >= 65 else None

ncrb["matched_district"] = ncrb.apply(
    lambda r: best_match_adv(r["district"], r["state"], census), axis=1
)

matched = ncrb["matched_district"].notna().sum()
print(f"🔍 Fuzzy matched {matched}/{len(ncrb)} districts successfully (adaptive mode).")

# Log unmatched
unmatched = ncrb[ncrb["matched_district"].isna()][["state", "district"]]
unmatched.to_csv(unmatched_path, index=False)
print(f"⚠️ Unmatched districts logged → {unmatched_path}")

# -------------------- 7. Merge --------------------
merged = pd.merge(ncrb, census, left_on="matched_district", right_on="district", how="left")

# -------------------- 8. Compute Hybrid Risk --------------------
merged["missing_rate"] = (merged["total_missing"] / merged["population"]).fillna(0)
merged["missing_total_norm"] = MinMaxScaler().fit_transform(merged[["total_missing"]])

merged["female_risk"] = merged["female_ratio"]
merged["literacy_risk"] = 1 - merged["literacy_rate"]
merged["worker_risk"] = 1 - merged["workers_ratio"]

for col in ["missing_rate", "female_risk", "literacy_risk", "worker_risk"]:
    merged[col] = merged[col].fillna(0)
    merged[col] = MinMaxScaler().fit_transform(merged[[col]])

merged["risk_score"] = (
    0.5 * merged["missing_rate"] +
    0.25 * merged["missing_total_norm"] +
    0.1 * merged["female_risk"] +
    0.1 * merged["literacy_risk"] +
    0.05 * merged["worker_risk"]
)
merged["risk_score"] = MinMaxScaler().fit_transform(merged[["risk_score"]])

# --- Per-State Normalization (so metros don’t appear unrealistically low) ---
merged["risk_score"] = merged.groupby("state_x")["risk_score"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)
merged["risk_score"] = merged["risk_score"].fillna(0)

# -------------------- 9. Final Output --------------------
keep_cols = [
    "state_x", "matched_district", "missing_rate", "female_ratio",
    "literacy_rate", "workers_ratio", "male_total", "female_total",
    "child_total", "risk_score"
]
final_df = merged[keep_cols].rename(columns={"state_x": "state", "matched_district": "district"})
final_df["state"] = final_df["state"].str.title().fillna("Unknown")
final_df["district"] = final_df["district"].str.lower().str.strip()

# --- Fill NaNs ---
final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# -------------------- 10. Deduplicate + Save --------------------
# --- Deduplicate duplicate districts safely ---
numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
final_df = (
    final_df.groupby("district", as_index=False)[numeric_cols].mean()
    .merge(
        final_df.drop(columns=numeric_cols).drop_duplicates(subset=["district"]),
        on="district",
        how="left"
    )
)

# --- Fill NaNs & clean ---
final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# --- Verify uniqueness ---
unique_districts = final_df["district"].nunique()
print(f"🧹 Deduplicated to {unique_districts} unique districts.")

# --- Save ---
os.makedirs("../data", exist_ok=True)
final_df.to_csv(out_path, index=False)
print(f"✅ Preprocessing complete — saved to {out_path}")
print(f"🏙️ Total districts processed: {len(final_df)}")


# -------------------- 11. Summary --------------------
print("\n🔥 Top 10 High-Risk Districts:")
print(final_df.sort_values("risk_score", ascending=False).head(10))


