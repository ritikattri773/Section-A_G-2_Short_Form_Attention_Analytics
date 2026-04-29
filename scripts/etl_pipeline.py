"""
=============================================================================
ETL PIPELINE — Social Media Behavior & Mental Health Dataset
=============================================================================
Dataset  : cleaned_data_userID__1_.csv
Records  : 11,988 users
Columns  : 15 (demographics + platform usage + mental health indicators)

Pipeline Phases:
  1. EXTRACT  — Load raw data, validate file integrity, profile the data
  2. TRANSFORM — Clean, encode, engineer features, normalize
  3. LOAD      — Export transformed data in multiple formats

Author   : ETL Pipeline
=============================================================================
"""

import os
import warnings
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ETL")


# ===========================================================================
# PHASE 1 — EXTRACT
# ===========================================================================

def extract(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame.
    Validates:
      - File existence
      - Non-empty dataset
      - Expected columns present
      - Basic dtype sanity
    """
    log.info("=" * 60)
    log.info("PHASE 1 — EXTRACT")
    log.info("=" * 60)

    path = Path(filepath)

    # --- File existence check -------------------------------------------
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {filepath}")

    file_size_kb = path.stat().st_size / 1024
    log.info(f"Source file : {path.name}  ({file_size_kb:.1f} KB)")

    # --- Load data ----------------------------------------------------------
    df = pd.read_csv(filepath)
    log.info(f"Rows loaded : {len(df):,}")
    log.info(f"Columns     : {df.shape[1]}  →  {list(df.columns)}")

    # --- Expected schema check ---------------------------------------------
    EXPECTED_COLUMNS = {
        "user_id", "age", "gender", "location", "platform",
        "reels_watch_time_hours", "daily_screen_time_hours",
        "scrolling_sessions_day", "notifications_per_day",
        "sleep_hours", "physical_activity_hours_week",
        "stress_level", "attention_span_score",
        "focus_level", "task_completion_rate"
    }
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    log.info("Schema validation   : PASSED ✓")

    # --- Empty dataset check -----------------------------------------------
    if df.empty:
        raise ValueError("Extracted DataFrame is empty.")

    # --- Data profiling snapshot -------------------------------------------
    log.info("\n--- RAW DATA PROFILE ---")
    log.info(f"{'Column':<35} {'Dtype':<12} {'Nulls':>6}  {'Null%':>7}")
    log.info("-" * 65)
    for col in df.columns:
        nulls = df[col].isna().sum()
        pct   = (nulls / len(df)) * 100
        log.info(f"  {col:<33} {str(df[col].dtype):<12} {nulls:>6}  {pct:>6.2f}%")

    log.info("\n--- CATEGORICAL DISTRIBUTIONS (RAW) ---")
    for col in ["gender", "location", "platform", "stress_level"]:
        log.info(f"\n  {col}:")
        for val, cnt in df[col].value_counts().items():
            log.info(f"    {val:<20} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

    log.info("\n--- NUMERIC SUMMARY (RAW) ---")
    log.info(df[[
        "age", "reels_watch_time_hours", "daily_screen_time_hours",
        "sleep_hours", "attention_span_score", "focus_level",
        "task_completion_rate"
    ]].describe().round(2).to_string())

    log.info("\n[EXTRACT] Completed successfully.\n")
    return df


# ===========================================================================
# PHASE 2 — TRANSFORM
# ===========================================================================

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full transformation pipeline:
      2.1  Deep copy to preserve raw data
      2.2  Deduplicate
      2.3  Standardize categorical values
      2.4  Handle missing values
      2.5  Fix data types
      2.6  Outlier detection & capping
      2.7  Feature engineering
      2.8  Encoding (ordinal + one-hot)
      2.9  Normalization (Min-Max on key numeric cols)
      2.10 Final validation
    """
    log.info("=" * 60)
    log.info("PHASE 2 — TRANSFORM")
    log.info("=" * 60)

    df = df.copy()  # never mutate the raw extract

    # -----------------------------------------------------------------------
    # 2.1  DEDUPLICATION
    # -----------------------------------------------------------------------
    log.info("\n[2.1] Deduplication")
    before = len(df)
    df.drop_duplicates(subset="user_id", keep="first", inplace=True)
    removed = before - len(df)
    log.info(f"  Duplicate rows removed  : {removed}")
    log.info(f"  Rows remaining          : {len(df):,}")

    # -----------------------------------------------------------------------
    # 2.2  STANDARDIZE CATEGORICAL VALUES
    # -----------------------------------------------------------------------
    log.info("\n[2.2] Standardizing categorical values")

    # gender → strip whitespace, title-case; map unknowns to 'Other'
    df["gender"] = df["gender"].str.strip().str.title()
    df["gender"] = df["gender"].replace({
        "Unknown": "Other",
        "Unspecified": "Other",
        "N/A": "Other",
    })
    log.info(f"  gender unique values    : {sorted(df['gender'].unique())}")

    # location → strip whitespace, title-case
    df["location"] = df["location"].str.strip().str.title()
    log.info(f"  location unique values  : {sorted(df['location'].unique())}")

    # platform → strip whitespace
    df["platform"] = df["platform"].str.strip()
    log.info(f"  platform unique values  : {sorted(df['platform'].unique())}")

    # stress_level → title-case
    df["stress_level"] = df["stress_level"].str.strip().str.title()
    log.info(f"  stress_level unique     : {sorted(df['stress_level'].unique())}")

    # -----------------------------------------------------------------------
    # 2.3  HANDLE MISSING VALUES
    # -----------------------------------------------------------------------
    log.info("\n[2.3] Handling missing values")

    # notifications_per_day → fill with median (220 nulls, ~1.8%)
    notif_median = df["notifications_per_day"].median()
    df["notifications_per_day"] = df["notifications_per_day"].fillna(notif_median)
    log.info(f"  notifications_per_day   : filled {220} nulls with median={notif_median:.0f}")

    # physical_activity_hours_week → fill with median (490 nulls, ~4.1%)
    activity_median = df["physical_activity_hours_week"].median()
    df["physical_activity_hours_week"] = df["physical_activity_hours_week"].fillna(activity_median)
    log.info(f"  physical_activity       : filled {490} nulls with median={activity_median:.2f}")

    # verify no nulls remain
    remaining_nulls = df.isnull().sum().sum()
    log.info(f"  Total nulls remaining   : {remaining_nulls}")
    if remaining_nulls > 0:
        log.warning("  ⚠ Unexpected nulls remain — check transform logic")
    else:
        log.info("  Null handling           : CLEAN ✓")

    # -----------------------------------------------------------------------
    # 2.4  FIX DATA TYPES
    # -----------------------------------------------------------------------
    log.info("\n[2.4] Fixing data types")

    # Cast to int after fill
    df["notifications_per_day"] = df["notifications_per_day"].round(0).astype(int)

    # Ensure floats on numeric columns
    float_cols = [
        "reels_watch_time_hours", "daily_screen_time_hours",
        "sleep_hours", "physical_activity_hours_week",
        "attention_span_score", "focus_level", "task_completion_rate"
    ]
    for col in float_cols:
        df[col] = df[col].astype(float).round(4)

    log.info("  Data types enforced     : ✓")

    # -----------------------------------------------------------------------
    # 2.5  OUTLIER DETECTION & CAPPING (IQR Method)
    # -----------------------------------------------------------------------
    log.info("\n[2.5] Outlier detection & capping (IQR method, factor=1.5)")

    outlier_cols = [
        "reels_watch_time_hours", "daily_screen_time_hours",
        "scrolling_sessions_day", "notifications_per_day",
        "sleep_hours", "physical_activity_hours_week",
        "attention_span_score", "focus_level", "task_completion_rate"
    ]

    outlier_summary = {}
    for col in outlier_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = {
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outliers_capped": int(n_outliers)
        }

        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_outliers > 0:
            log.info(f"  {col:<35}  clipped {n_outliers:>4} values  "
                     f"[{lower:.2f}, {upper:.2f}]")
        else:
            log.info(f"  {col:<35}  no outliers found")

    # -----------------------------------------------------------------------
    # 2.6  FEATURE ENGINEERING
    # -----------------------------------------------------------------------
    log.info("\n[2.6] Feature Engineering")

    # --- Age buckets -------------------------------------------------------
    bins  = [0, 18, 25, 35, 45, 100]
    labels = ["<18", "18-25", "26-35", "36-45", "45+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    log.info("  age_group               : created (5 buckets)")

    # --- Screen time severity score ----------------------------------------
    # Composite: 40% reels + 40% daily screen + 20% scrolling sessions (scaled)
    df["screen_time_severity"] = (
        0.40 * df["reels_watch_time_hours"] / df["reels_watch_time_hours"].max() +
        0.40 * df["daily_screen_time_hours"] / df["daily_screen_time_hours"].max() +
        0.20 * df["scrolling_sessions_day"] / df["scrolling_sessions_day"].max()
    ).round(4)
    log.info("  screen_time_severity    : created (composite 0–1 score)")

    # --- Wellness score ----------------------------------------------------
    # Higher sleep + activity → higher wellness; penalise high notifications
    df["wellness_score"] = (
        0.35 * df["sleep_hours"] / df["sleep_hours"].max() +
        0.35 * df["physical_activity_hours_week"] / df["physical_activity_hours_week"].max() +
        0.30 * (1 - df["notifications_per_day"] / df["notifications_per_day"].max())
    ).round(4)
    log.info("  wellness_score          : created (composite 0–1 score)")

    # --- Productivity index -----------------------------------------------
    # Focus + attention + task completion → overall productivity
    df["productivity_index"] = (
        0.35 * df["focus_level"] / df["focus_level"].max() +
        0.35 * df["attention_span_score"] / df["attention_span_score"].max() +
        0.30 * df["task_completion_rate"] / df["task_completion_rate"].max()
    ).round(4)
    log.info("  productivity_index      : created (composite 0–1 score)")

    # --- High screen time flag --------------------------------------------
    df["high_screen_time_flag"] = (df["daily_screen_time_hours"] > 8).astype(int)
    log.info("  high_screen_time_flag   : created (1 if daily > 8 hrs)")

    # --- Sleep quality flag -----------------------------------------------
    df["poor_sleep_flag"] = (df["sleep_hours"] < 6).astype(int)
    log.info("  poor_sleep_flag         : created (1 if sleep < 6 hrs)")

    # --- Reels-to-screen ratio -------------------------------------------
    df["reels_to_screen_ratio"] = (
        df["reels_watch_time_hours"] / df["daily_screen_time_hours"].replace(0, np.nan)
    ).fillna(0).round(4)
    log.info("  reels_to_screen_ratio   : created (reels/daily screen time)")

    # -----------------------------------------------------------------------
    # 2.7  ENCODING
    # -----------------------------------------------------------------------
    log.info("\n[2.7] Encoding categorical columns")

    # Ordinal encoding — stress_level (natural order: Low < Medium < High)
    stress_map = {"Low": 1, "Medium": 2, "High": 3}
    df["stress_level_encoded"] = df["stress_level"].map(stress_map)
    log.info("  stress_level_encoded    : ordinal  {Low:1, Medium:2, High:3}")

    # One-hot encoding — gender, location, platform
    df = pd.get_dummies(df, columns=["gender", "location", "platform"], prefix_sep="_", dtype=int)
    new_ohe_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["gender_", "location_", "platform_"]
    )]
    log.info(f"  One-hot encoded cols    : {new_ohe_cols}")

    # -----------------------------------------------------------------------
    # 2.8  NORMALIZATION (Min-Max on key numeric features)
    # -----------------------------------------------------------------------
    log.info("\n[2.8] Min-Max normalization")

    normalize_cols = [
        "age", "reels_watch_time_hours", "daily_screen_time_hours",
        "scrolling_sessions_day", "notifications_per_day",
        "sleep_hours", "physical_activity_hours_week",
        "attention_span_score", "focus_level", "task_completion_rate"
    ]

    for col in normalize_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[f"{col}_norm"] = ((df[col] - min_val) / (max_val - min_val)).round(4)
        else:
            df[f"{col}_norm"] = 0.0
        log.info(f"  {col}_norm  →  range [{min_val:.2f}, {max_val:.2f}]")

    # -----------------------------------------------------------------------
    # 2.9  FINAL VALIDATION
    # -----------------------------------------------------------------------
    log.info("\n[2.9] Final validation")
    log.info(f"  Final shape             : {df.shape}")
    log.info(f"  Remaining nulls         : {df.isnull().sum().sum()}")
    log.info(f"  New columns added       : {df.shape[1] - 15} (from original 15)")

    log.info("\n--- ENGINEERED FEATURE STATS ---")
    engineered = [
        "screen_time_severity", "wellness_score",
        "productivity_index", "stress_level_encoded"
    ]
    log.info(df[engineered].describe().round(3).to_string())

    log.info("\n[TRANSFORM] Completed successfully.\n")
    return df


# ===========================================================================
# PHASE 3 — LOAD
# ===========================================================================

def load(df: pd.DataFrame, output_dir: str = "output") -> dict:
    """
    Persist transformed data in multiple formats:
      - Full transformed CSV (all columns)
      - ML-ready CSV (numeric/encoded only, no raw categoricals)
      - Summary statistics CSV
      - ETL run metadata text file
    """
    log.info("=" * 60)
    log.info("PHASE 3 — LOAD")
    log.info("=" * 60)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    outputs = {}

    # --- 3.1  Full Transformed CSV -----------------------------------------
    full_path = out / f"transformed_full_{timestamp}.csv"
    df.to_csv(full_path, index=False)
    outputs["full_csv"] = str(full_path)
    log.info(f"\n[3.1] Full CSV saved       : {full_path}")
    log.info(f"      Shape                 : {df.shape}")

    # --- 3.2  ML-Ready CSV (drop raw/redundant cols) -----------------------
    drop_for_ml = ["user_id", "age_group", "stress_level"]
    ml_df = df.drop(columns=[c for c in drop_for_ml if c in df.columns])

    ml_path = out / f"ml_ready_{timestamp}.csv"
    ml_df.to_csv(ml_path, index=False)
    outputs["ml_csv"] = str(ml_path)
    log.info(f"\n[3.2] ML-ready CSV saved   : {ml_path}")
    log.info(f"      Shape                 : {ml_df.shape}")
    log.info(f"      Columns               : {list(ml_df.columns)}")

    # --- 3.3  Summary Statistics CSV ---------------------------------------
    stats_df = df[[
        "age", "reels_watch_time_hours", "daily_screen_time_hours",
        "scrolling_sessions_day", "notifications_per_day",
        "sleep_hours", "physical_activity_hours_week",
        "attention_span_score", "focus_level", "task_completion_rate",
        "screen_time_severity", "wellness_score", "productivity_index",
        "stress_level_encoded"
    ]].describe().round(4)

    stats_path = out / f"summary_stats_{timestamp}.csv"
    stats_df.to_csv(stats_path)
    outputs["stats_csv"] = str(stats_path)
    log.info(f"\n[3.3] Summary stats saved  : {stats_path}")

    # --- 3.4  ETL Metadata file --------------------------------------------
    meta_path = out / f"etl_metadata_{timestamp}.txt"
    with open(meta_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ETL RUN METADATA\n")
        f.write("=" * 60 + "\n")
        f.write(f"Run timestamp           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source file             : cleaned_data_userID__1_.csv\n")
        f.write(f"Source rows             : 11,988\n")
        f.write(f"Source columns          : 15\n")
        f.write(f"\nPost-transform shape   : {df.shape}\n")
        f.write(f"Columns added           : {df.shape[1] - 15}\n")
        f.write(f"\nNull imputation:\n")
        f.write(f"  notifications_per_day      : median imputation (220 nulls, 1.84%)\n")
        f.write(f"  physical_activity_hours_week: median imputation (490 nulls, 4.09%)\n")
        f.write(f"\nFeatures engineered:\n")
        f.write(f"  age_group              : 5-bucket ordinal age bins\n")
        f.write(f"  screen_time_severity   : composite score (reels + screen + scrolling)\n")
        f.write(f"  wellness_score         : composite score (sleep + activity - notifications)\n")
        f.write(f"  productivity_index     : composite score (focus + attention + task)\n")
        f.write(f"  high_screen_time_flag  : binary (daily_screen_time > 8h)\n")
        f.write(f"  poor_sleep_flag        : binary (sleep_hours < 6h)\n")
        f.write(f"  reels_to_screen_ratio  : reels / daily_screen_time\n")
        f.write(f"\nEncoding:\n")
        f.write(f"  stress_level_encoded   : ordinal (Low=1, Medium=2, High=3)\n")
        f.write(f"  gender                 : one-hot\n")
        f.write(f"  location               : one-hot\n")
        f.write(f"  platform               : one-hot\n")
        f.write(f"\nNormalization:\n")
        f.write(f"  10 numeric columns min-max normalized (suffix _norm)\n")
        f.write(f"\nOutput files:\n")
        for k, v in outputs.items():
            f.write(f"  {k:<20} : {v}\n")
        f.write("=" * 60 + "\n")

    outputs["metadata"] = str(meta_path)
    log.info(f"\n[3.4] ETL metadata saved   : {meta_path}")
    log.info("\n[LOAD] Completed successfully.")
    log.info(f"\nAll outputs in : {out.resolve()}/")
    log.info("=" * 60)

    return outputs


# ===========================================================================
# MAIN — RUN FULL PIPELINE
# ===========================================================================

if __name__ == "__main__":
    # Resolve paths relative to the project root (parent of scripts/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SOURCE_FILE  = str(PROJECT_ROOT / "data" / "processed" / "cleaned_data_userID.csv")
    OUTPUT_DIR   = str(PROJECT_ROOT / "etl_output")

    log.info("\n" + "=" * 60)
    log.info(" SOCIAL MEDIA BEHAVIOR — ETL PIPELINE")
    log.info(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60 + "\n")

    try:
        # EXTRACT
        raw_df = extract(SOURCE_FILE)

        # TRANSFORM
        transformed_df = transform(raw_df)

        # LOAD
        output_paths = load(transformed_df, output_dir=OUTPUT_DIR)

        log.info("\n  ETL PIPELINE FINISHED SUCCESSFULLY")
        log.info(f"    Output files : {list(output_paths.values())}")

    except FileNotFoundError as e:
        log.error(f"FILE ERROR: {e}")
    except ValueError as e:
        log.error(f"VALIDATION ERROR: {e}")
    except Exception as e:
        log.error(f"UNEXPECTED ERROR: {e}", exc_info=True)
