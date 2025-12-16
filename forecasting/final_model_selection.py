import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths (edit for your environment)
# Tip: In GitHub, prefer relative paths like: data/processed/...
# ---------------------------------------------------------------------
PATH_MASE = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\MASE.xlsx"
PATH_WMAPE = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\WMAPE.xlsx"
PATH_BIAS = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\BIAS.xlsx"
PATH_TARGETS = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\HEDEF_TABLOSU.xlsx"
PATH_SEGMENT = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\Bayi_ABC_Segment_Analizi.xlsx"
PATH_WEEKLY_DATA = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\Bayi_Haftalik_DETAY_SERISI.xlsx"
PATH_OUTPUT = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\FINAL_DECISION_AND_MODEL_SELECTION_NO_NAIVE.xlsx"

print("Running Final Model Selection (Cost-Sensitive Logic, XGBoost-safe parsing)...")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_and_melt(path: str, value_name: str, metric_prefix: str) -> pd.DataFrame:
    """
    Reads a wide metric Excel file (columns: Bayi Kodu + model columns),
    converts it to long format, standardizes model names, and removes Naive variants.
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    model_cols = [c for c in df.columns if c != "Bayi Kodu"]
    df_long = df.melt(
        id_vars="Bayi Kodu",
        value_vars=model_cols,
        var_name="Model_Raw",
        value_name=value_name,
    )

    # Remove metric prefix from column headers (handles both spaces and underscores)
    base_metric = metric_prefix.replace("_", "")
    df_long["Model"] = (
        df_long["Model_Raw"]
        .astype(str)
        .str.replace(f"^{base_metric}[\\s_]*", "", regex=True)
        .str.strip()
    )

    # Drop Naive / SNaive etc.
    df_long = df_long[~df_long["Model"].str.contains("Naive", case=False, na=False)]

    return df_long[["Bayi Kodu", "Model", value_name]]


def parse_percentage_low(val) -> float:
    """Parse the first percentage in a string like '%35' or '35%'. Returns 0-1 scale."""
    if pd.isna(val):
        return 999.0
    nums = re.findall(r"[\d\.]+", str(val))
    return float(nums[0]) / 100 if nums else 999.0


def parse_percentage_high(val) -> float:
    """Parse the last percentage in a string. Returns 0-1 scale."""
    if pd.isna(val):
        return 0.0
    nums = re.findall(r"[\d\.]+", str(val))
    return float(nums[-1]) / 100 if nums else 0.0


# ---------------------------------------------------------------------
# 1) Load metrics and merge
# ---------------------------------------------------------------------
print("- Loading metric files and standardizing model names...")
df_mase = load_and_melt(PATH_MASE, "MASE", "MASE_")
df_wmape = load_and_melt(PATH_WMAPE, "WMAPE", "WMAPE_")
df_bias = load_and_melt(PATH_BIAS, "BIAS", "BIAS_")

df_all = pd.merge(df_mase, df_wmape, on=["Bayi Kodu", "Model"], how="inner")
df_all = pd.merge(df_all, df_bias, on=["Bayi Kodu", "Model"], how="inner")

print(f"- Total merged rows: {len(df_all)}")
print(f"- Detected models: {df_all['Model'].unique()}")

# ---------------------------------------------------------------------
# 2) Selection logic (MASE -> WMAPE -> BIAS risk swap)
# ---------------------------------------------------------------------
selection_results = []

for dealer_id in df_all["Bayi Kodu"].unique():
    df_dealer = df_all[df_all["Bayi Kodu"] == dealer_id].copy()

    # Step 1: Quality filter (MASE <= 1.2)
    df_qualified = df_dealer[df_dealer["MASE"] <= 1.2].copy()
    quality_note = "High Quality"

    if df_qualified.empty:
        df_qualified = df_dealer.copy()
        quality_note = "Low Quality (MASE>1.2)"

    # Step 2: Accuracy ranking (min WMAPE)
    df_qualified = df_qualified.sort_values(by="WMAPE", ascending=True).reset_index(drop=True)
    if df_qualified.empty:
        continue

    best = df_qualified.iloc[0]
    final_choice = best
    reason = "Lowest WMAPE"

    # Step 3: Risk management (prefer positive bias if WMAPE difference is small)
    if len(df_qualified) > 1:
        runner_up = df_qualified.iloc[1]
        wmape_diff = runner_up["WMAPE"] - best["WMAPE"]

        if (best["BIAS"] < 0) and (runner_up["BIAS"] > 0) and (wmape_diff < 0.05):
            final_choice = runner_up
            reason = "Risk swap (prefer positive BIAS if WMAPE gap < 0.05)"

    selection_results.append(
        {
            "Bayi Kodu": dealer_id,
            "Selected_Model": final_choice["Model"],
            "MASE": final_choice["MASE"],
            "WMAPE": final_choice["WMAPE"],
            "BIAS": final_choice["BIAS"],
            "Decision_Reason": reason,
            "Model_Quality": quality_note,
        }
    )

df_selection = pd.DataFrame(selection_results)

# ---------------------------------------------------------------------
# 3) Add segment + difficulty (CV-based) and compare to target thresholds
# ---------------------------------------------------------------------
df_seg = pd.read_excel(PATH_SEGMENT)
seg_col = next((c for c in df_seg.columns if ("Segment" in c) or ("Grubu" in c)), None)
df_seg = df_seg[["Bayi Kodu", seg_col]].rename(columns={seg_col: "Volume_Group"})

df_targets = pd.read_excel(PATH_TARGETS)

df_final = pd.merge(df_selection, df_seg, on="Bayi Kodu", how="left")

df_weekly = pd.read_excel(PATH_WEEKLY_DATA)
numeric_cols = [c for c in df_weekly.columns if c not in ["Bayi Kodu", "Bayi Adı"]]

df_weekly["Mean"] = df_weekly[numeric_cols].mean(axis=1)
df_weekly["Std"] = df_weekly[numeric_cols].std(axis=1)
df_weekly["CV"] = df_weekly["Std"] / df_weekly["Mean"]

median_cv = df_weekly["CV"].median()
df_weekly["Difficulty"] = df_weekly["CV"].apply(
    lambda x: "Stable (Easy)" if x < median_cv else "Volatile (Hard)"
)

df_final = pd.merge(df_final, df_weekly[["Bayi Kodu", "Difficulty"]], on="Bayi Kodu", how="left")

df_final = pd.merge(
    df_final,
    df_targets,
    left_on=["Volume_Group", "Difficulty"],
    right_on=["Hacim Grubu", "Zorluk (Yapı)"],
    how="left",
)

def determine_status(row) -> str:
    wmape = row["WMAPE"]
    q1_str = row.get("MUKEMMEL_HEDEF (Q1)")
    q3_str = row.get("RISKLI_BOLGE (>Q3)")

    excellent_thr = parse_percentage_low(q1_str)
    risky_thr = parse_percentage_high(q3_str)

    if wmape < excellent_thr:
        return "EXCELLENT (Within Target)"
    elif wmape > risky_thr:
        return "CRITICAL (Out of Target)"
    else:
        return "ACCEPTABLE (Deviation)"

df_final["PERFORMANCE_STATUS"] = df_final.apply(determine_status, axis=1)

# ---------------------------------------------------------------------
# 4) Export
# ---------------------------------------------------------------------
output_columns = [
    "Bayi Kodu",
    "Volume_Group",
    "Difficulty",
    "Selected_Model",
    "WMAPE",
    "BIAS",
    "MASE",
    "PERFORMANCE_STATUS",
    "Decision_Reason",
    "Model_Quality",
    "MUKEMMEL_HEDEF (Q1)",
    "RISKLI_BOLGE (>Q3)",
]
final_cols = [c for c in output_columns if c in df_final.columns]
df_export = df_final[final_cols]

df_export.to_excel(PATH_OUTPUT, index=False)

print("\n" + "=" * 60)
print("DONE: Final model selection completed (Naive excluded, XGBoost included).")
print(f"Saved output to: {PATH_OUTPUT}")
print("=" * 60)
print(df_export[["Bayi Kodu", "Selected_Model", "WMAPE", "PERFORMANCE_STATUS"]].head())
