import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths (Mature dealers: 123)
# NOTE: For GitHub, prefer relative paths like:
#   data/raw/...
#   data/processed/...
#   results/...
# ---------------------------------------------------------------------
PATH_DATA = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\olgun bayiler için\Bayi_Haftalik_DETAY_SERISI.xlsx"
PATH_METRICS = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\olgun bayiler için\OLGUN_BAYILER_METRIK_SONUCLARI.xlsx"

PATH_FINAL_CHOICE = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\olgun bayiler için\MATURE_DEALERS_FINAL_SELECTION_WMAPE.xlsx"
PATH_TARGET_TABLE = r"C:\Users\PC\Desktop\Renault\renault veriler\Yeni klasör (2)\olgun bayiler için\MATURE_DEALERS_TARGET_TABLE_WMAPE.xlsx"

print("Starting methodological analysis (123 mature dealers)...")
print("-" * 60)


def clean_dealer_code(series: pd.Series) -> pd.Series:
    """Standardize dealer codes (string, strip, remove trailing .0)."""
    return series.astype(str).str.strip().str.replace(".0", "", regex=False)


def wide_to_long_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert weekly wide format to long format if needed."""
    if "Bayi Kodu" in df.columns:
        id_vars = ["Bayi Kodu", "Bayi Adı"] if "Bayi Adı" in df.columns else ["Bayi Kodu"]
        date_vars = [c for c in df.columns if c not in id_vars]
        df_long = df.melt(id_vars=id_vars, value_vars=date_vars, var_name="DATE", value_name="QTY")
    else:
        df_long = df.copy()

    df_long["QTY"] = pd.to_numeric(df_long["QTY"], errors="coerce").fillna(0)
    df_long["Bayi Kodu"] = clean_dealer_code(df_long["Bayi Kodu"])
    return df_long


def compute_dealer_stats(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dealer-level statistics:
    - Mean weekly volume (active period only)
    - Total volume
    - CV (Std/Mean) on active period
    """
    stats = []

    for dealer_id in df_long["Bayi Kodu"].unique():
        df_dealer = df_long[df_long["Bayi Kodu"] == dealer_id]

        # Active period cleaning (avoid cold-start zeros)
        if df_dealer["QTY"].sum() > 0:
            active_series = df_dealer[df_dealer["QTY"].cumsum() > 0]["QTY"]

            mean_val = active_series.mean()
            std_val = active_series.std()
            total_val = active_series.sum()
            cv = (std_val / mean_val) if mean_val > 0 else 0

            stats.append(
                {
                    "Bayi Kodu": dealer_id,
                    "Mean_Weekly_Volume": mean_val,
                    "Total_Volume": total_val,
                    "CV": cv,
                }
            )

    return pd.DataFrame(stats)


def assign_volume_segment(df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Non-parametric volume segmentation (quantile-based):
      A: top 10% (>= 90th percentile)
      B: 70-90%
      C: 40-70%
      D: bottom 40%
    """
    q90 = df_stats["Total_Volume"].quantile(0.90)
    q70 = df_stats["Total_Volume"].quantile(0.70)
    q40 = df_stats["Total_Volume"].quantile(0.40)

    def segment_rule(vol):
        if vol >= q90:
            return "A (VIP)"
        elif vol >= q70:
            return "B (High)"
        elif vol >= q40:
            return "C (Standard)"
        else:
            return "D (Micro)"

    df_stats["Volume_Segment"] = df_stats["Total_Volume"].apply(segment_rule)
    return df_stats, (q90, q70, q40)


def assign_difficulty(df_stats: pd.DataFrame) -> pd.DataFrame:
    """Difficulty segmentation using median CV threshold."""
    median_cv = df_stats["CV"].median()
    df_stats["Difficulty"] = df_stats["CV"].apply(
        lambda x: "Stable (Easy)" if x < median_cv else "Volatile (Hard)"
    )
    return df_stats, median_cv


# ---------------------------------------------------------------------
# Step 1: Distribution and CV analysis
# ---------------------------------------------------------------------
print("Step 1: Loading weekly data and computing volume/CV...")
df_raw = pd.read_excel(PATH_DATA)
df_long = wide_to_long_weekly(df_raw)

df_stats = compute_dealer_stats(df_long)
if df_stats.empty:
    raise ValueError("No dealers found for analysis (check input data).")

df_stats, (q90, q70, q40) = assign_volume_segment(df_stats)
df_stats, median_cv = assign_difficulty(df_stats)

print(f"  > Volume thresholds: A>= {q90:.1f}, B>= {q70:.1f}, C>= {q40:.1f}")
print(f"  > Median CV threshold: {median_cv:.3f}")


# ---------------------------------------------------------------------
# Step 2: Model selection by WMAPE (best model per dealer)
# ---------------------------------------------------------------------
print("Step 2: Loading metrics and selecting best model by WMAPE...")

try:
    xls = pd.ExcelFile(PATH_METRICS)
    sheet = "Tum_Detaylar" if "Tum_Detaylar" in xls.sheet_names else 0
    df_metrics = pd.read_excel(PATH_METRICS, sheet_name=sheet)
    df_metrics["Bayi Kodu"] = clean_dealer_code(df_metrics["Bayi Kodu"])
except Exception as e:
    raise RuntimeError(f"Metrics file could not be read: {e}")

final_rows = []

for dealer_id in df_metrics["Bayi Kodu"].unique():
    df_dealer_models = df_metrics[df_metrics["Bayi Kodu"] == dealer_id].copy()
    if df_dealer_models.empty:
        continue

    df_best = df_dealer_models.sort_values(by="WMAPE", ascending=True).iloc[0]

    stat_match = df_stats[df_stats["Bayi Kodu"] == dealer_id]
    if stat_match.empty:
        continue

    st = stat_match.iloc[0]
    final_rows.append(
        {
            "Bayi Kodu": dealer_id,
            "Volume_Segment": st["Volume_Segment"],
            "Difficulty": st["Difficulty"],
            "Mean_Weekly_Volume": st["Mean_Weekly_Volume"],
            "CV": st["CV"],
            "Selected_Model": df_best["Model"],
            "Best_WMAPE": df_best["WMAPE"],  # used for targets (internal benchmarking)
            "SMAPE": df_best.get("SMAPE", np.nan),
            "MASE": df_best.get("MASE", np.nan),
            "BIAS": df_best.get("BIAS", np.nan),
        }
    )

df_final = pd.DataFrame(final_rows)

if df_final.empty:
    raise ValueError("No final selections created (check code matching and metrics file).")


# ---------------------------------------------------------------------
# Step 3: Target table (Internal benchmarking using Q1-Q3 of Best_WMAPE)
# ---------------------------------------------------------------------
print("Step 3: Building target table (Q1-Q3) by segment & difficulty...")

target_rows = []
grouped = df_final.groupby(["Volume_Segment", "Difficulty"])

for (seg, diff), group in grouped:
    n = len(group)
    q1 = group["Best_WMAPE"].quantile(0.25)
    q3 = group["Best_WMAPE"].quantile(0.75)

    target_rows.append(
        {
            "Volume_Group": seg,
            "Difficulty": diff,
            "Dealer_Count": n,
            "Avg_Mean_Weekly_Volume": round(group["Mean_Weekly_Volume"].mean(), 2),
            "Avg_CV": round(group["CV"].mean(), 3),
            "EXCELLENT_TARGET (Q1)": f"<%{q1:.1f}",
            "NORMAL_RANGE (Q1-Q3)": f"%{q1:.1f} - %{q3:.1f}",
            "RISK_ZONE (>Q3)": f">%{q3:.1f}",
        }
    )

df_targets = pd.DataFrame(target_rows).sort_values("Volume_Group")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
df_final.to_excel(PATH_FINAL_CHOICE, index=False)
df_targets.to_excel(PATH_TARGET_TABLE, index=False)

print("-" * 60)
print("ANALYSIS COMPLETED")
print(f"1) Dealer-level final selections saved to: {PATH_FINAL_CHOICE}")
print(f"2) Target table saved to: {PATH_TARGET_TABLE}")
print("-" * 60)
print("Sample target table:")
print(df_targets[["Volume_Group", "Difficulty", "EXCELLENT_TARGET (Q1)", "RISK_ZONE (>Q3)"]].to_string(index=False))
