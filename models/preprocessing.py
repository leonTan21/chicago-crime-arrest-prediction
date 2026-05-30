"""
preprocessing.py — shared pipeline for all Chicago Crime model scripts.
Run from project root: imported by individual model scripts.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pointbiserialr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_classif, chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

RANDOM_STATE = 42
DROP_COLS = [
    "ID", "Case Number", "Block", "IUCR", "Description",
    "Updated On", "X Coordinate", "Y Coordinate", "Location", "FBI Code",
]
ENCODE_COLS = ["Primary Type", "Location Description"]


# ── 1. Cleaning & feature engineering ──────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Latitude", "Longitude", "Community Area", "District"])

    dt = pd.to_datetime(df["Date"], errors="coerce")
    df["Hour"]      = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    df["Month"]     = dt.dt.month
    df["Year"]      = dt.dt.year
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Season"]    = df["Month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    return df


def encode_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    y = df["Arrest"].astype(int)

    drop = [c for c in DROP_COLS + ["Date", "Arrest"] if c in df.columns]
    df = df.drop(columns=drop)

    le = LabelEncoder()
    for col in ENCODE_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    if "Domestic" in df.columns:
        df["Domestic"] = df["Domestic"].astype(int)

    obj_cols = df.select_dtypes(include="object").columns.tolist()
    df = df.drop(columns=obj_cols)
    df = df.fillna(df.median(numeric_only=True))

    return df, y


# ── 2. Feature selection ────────────────────────────────────────────────────

FEATURE_SELECTION_SAMPLE = 100_000   # Kendall & Chi² are O(n²); subsample
P_THRESHOLD = 0.05
MIN_SIGNIFICANT_TESTS = 3           # feature kept if ≥3 of 5 tests pass


def select_features(
    X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True
) -> tuple[list[str], pd.DataFrame]:
    """
    Run 5 statistical tests and keep features significant in ≥3 of them.
    Uses a 100k subsample for speed (Kendall/Chi² are O(n²)).
    """
    n = min(FEATURE_SELECTION_SAMPLE, len(X_train))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X_train), size=n, replace=False)
    X_s = X_train.iloc[idx]
    y_s = y_train.iloc[idx]

    results = {}
    for feat in X_train.columns:
        v = X_s[feat].values
        t = y_s.values

        pr, pp   = stats.pearsonr(v, t)
        pb_r, pb_p = pointbiserialr(v, t)

        # Kendall: even more expensive — use 25k sub-subsample
        k = min(25_000, n)
        ki = np.random.default_rng(RANDOM_STATE).choice(n, size=k, replace=False)
        kt, kp = kendalltau(v[ki], t[ki])

        results[feat] = {
            "Pearson_r":  pr,  "Pearson_p":  pp,
            "PB_r":      pb_r, "PB_p":      pb_p,
            "Kendall_τ":  kt,  "Kendall_p":  kp,
        }

    # ANOVA F-test
    f_vals, f_pvals = f_classif(X_s, y_s)
    for i, feat in enumerate(X_train.columns):
        results[feat]["ANOVA_F"] = f_vals[i]
        results[feat]["ANOVA_p"] = f_pvals[i]

    # Chi-squared (requires non-negative values)
    X_nn = X_s - X_s.min()
    chi_vals, chi_pvals = chi2(X_nn, y_s)
    for i, feat in enumerate(X_train.columns):
        results[feat]["Chi2_stat"] = chi_vals[i]
        results[feat]["Chi2_p"]    = chi_pvals[i]

    fs_df = pd.DataFrame(results).T
    p_cols = ["Pearson_p", "PB_p", "Kendall_p", "ANOVA_p", "Chi2_p"]
    fs_df["Tests_Significant"] = (fs_df[p_cols] < P_THRESHOLD).sum(axis=1)
    fs_df["Selected"] = fs_df["Tests_Significant"] >= MIN_SIGNIFICANT_TESTS

    selected = fs_df[fs_df["Selected"]].index.tolist()
    if verbose:
        print(f"  Feature selection: {len(selected)}/{len(X_train.columns)} features kept "
              f"(≥{MIN_SIGNIFICANT_TESTS}/5 tests, p<{P_THRESHOLD})")

    return selected, fs_df


# ── 3. Full pipeline ────────────────────────────────────────────────────────

def get_preprocessed_data(
    sample_size: int | None = None,
    run_feature_selection: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    End-to-end pipeline: load → engineer → encode → split → select → scale.

    Parameters
    ----------
    sample_size : int, optional
        Row cap before splitting (useful for quick testing).
    run_feature_selection : bool
        Set False to skip statistical tests and use all features.

    Returns
    -------
    X_train_sc, X_test_sc, y_train, y_test, feature_names, scaler
    """
    if verbose:
        print("Loading data...")
    df = load_data()

    if sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    if verbose:
        print(f"  Rows after load      : {len(df):,}")

    df = engineer_features(df)
    X, y = encode_and_clean(df)

    if verbose:
        print(f"  Rows after dropna    : {len(X):,}")
        print(f"  Arrest rate          : {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    if run_feature_selection:
        if verbose:
            print("Running feature selection...")
        selected, _ = select_features(X_train, y_train, verbose=verbose)
        X_train = X_train[selected]
        X_test  = X_test[selected]
    else:
        selected = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if verbose:
        print(f"  Train: {X_train_sc.shape}, Test: {X_test_sc.shape}")

    return X_train_sc, X_test_sc, y_train, y_test, selected, scaler
