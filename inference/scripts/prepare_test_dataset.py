"""
Test Set Preprocessing 

Replicates exactly the pipeline we used on train_ready_dataset.ipynb

How to run it:
  python prepare_test_dataset.py \
      --train_parquet data/processed/train_model_ready.parquet \
      --test_json     EXIST2026_test_clean.json \
      --output        data/processed/test_model_ready.parquet
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── EEG Constants (exact match with training notebook) ──────────────────────
BANDS = ["Alpha", "Beta", "Theta", "Gamma", "Delta"]
N_CH  = 16
AUX   = 1e-6

REGION_MAP = {
    "frontal_polar": [0, 1],
    "frontal":       [8, 9, 10, 11],
    "central":       [2, 3],
    "temporal":      [12, 13],
    "parietal":      [4, 5, 14, 15],
    "occipital":     [6, 7],
}

LEFT_CHANNELS  = [0, 2, 4, 6, 8, 10, 12, 14]
RIGHT_CHANNELS = [1, 3, 5, 7, 9, 11, 13, 15]

LOG_RATIO_PAIRS = [
    ("Beta",  "Alpha"),
    ("Theta", "Beta"),
    ("Theta", "Alpha"),
    ("Gamma", "Beta"),
    ("Gamma", "Alpha"),
]

GLOBAL_DIFF_PAIRS = [
    ("Alpha", "Delta"),
    ("Alpha", "Theta"),
    ("Alpha", "Beta"),
    ("Beta",  "Gamma"),
    ("Beta",  "Delta"),
    ("Beta",  "Theta"),
    ("Delta", "Gamma"),
    ("Delta", "Theta"),
    ("Gamma", "Theta"),
]

REGIONAL_DIFF_PAIRS = [
    ("Alpha", "Theta",  "frontal_polar"),
    ("Alpha", "Beta",   "frontal_polar"),
    ("Alpha", "Gamma",  "frontal_polar"),
    ("Beta",  "Delta",  "frontal_polar"),
    ("Beta",  "Theta",  "frontal_polar"),
    ("Delta", "Gamma",  "frontal_polar"),
    ("Gamma", "Theta",  "frontal_polar"),
    ("Alpha", "Gamma",  "central"),
    ("Alpha", "Theta",  "central"),
    ("Beta",  "Gamma",  "central"),
    ("Beta",  "Delta",  "central"),
    ("Delta", "Gamma",  "central"),
    ("Delta", "Theta",  "central"),
    ("Beta",  "Delta",  "temporal"),
    ("Delta", "Theta",  "temporal"),
    ("Alpha", "Beta",   "parietal"),
    ("Beta",  "Theta",  "occipital"),
    ("Gamma", "Theta",  "occipital"),
    ("Alpha", "Theta",  "frontal"),
]

# ── ET Feature Mapping (exact match with training notebook) ─────────────────
ET_FEATURE_MAP = {
    "reaction_time":              "et_reaction_time",
    "fixations_count":            "et_fixations",
    "fixations_duration_mean_ns": "et_fixation_duration",
    "saccades_count":             "et_saccades",
}


# ── EEG Functions ────────────────────────────────────────────────────────────

def zscore_user_eeg(user_feats):
    keys = [f"EXG_Channel_{ch}_{band}_power"
            for ch in range(N_CH) for band in BANDS]
    vals = np.array([user_feats.get(k, np.nan) for k in keys], dtype=float)
    mask = ~np.isnan(vals)
    if mask.sum() > 1:
        mu, sigma = vals[mask].mean(), vals[mask].std()
        if sigma > 0:
            vals[mask] = (vals[mask] - mu) / sigma
    return dict(zip(keys, vals))


def extract_eeg_user_features(d_zscored, d_raw):
    feats = {}
    d = d_zscored

    # 1. Raw (80): EEG_EXG_Channel_{ch}_{band}_power
    for ch in range(N_CH):
        for band in BANDS:
            key = f"EXG_Channel_{ch}_{band}_power"
            feats[f"EEG_{key}"] = d.get(key, np.nan)

    # 2. Global (5): EEG_{band}_global
    for band in BANDS:
        vals = [d.get(f"EXG_Channel_{ch}_{band}_power", np.nan) for ch in range(N_CH)]
        vals = [v for v in vals if not np.isnan(v)]
        feats[f"EEG_{band}_global"] = np.mean(vals) if vals else np.nan

    # 3. Regional (30): EEG_{band}_{region}
    for region, ch_list in REGION_MAP.items():
        for band in BANDS:
            vals = [d.get(f"EXG_Channel_{ch}_{band}_power", np.nan) for ch in ch_list]
            vals = [v for v in vals if not np.isnan(v)]
            feats[f"EEG_{band}_{region}"] = np.mean(vals) if vals else np.nan

    # 4. Lateralization (10): EEG_{band}_left / EEG_{band}_right
    for band in BANDS:
        l_vals = [d.get(f"EXG_Channel_{ch}_{band}_power", np.nan) for ch in LEFT_CHANNELS]
        r_vals = [d.get(f"EXG_Channel_{ch}_{band}_power", np.nan) for ch in RIGHT_CHANNELS]
        l_vals = [v for v in l_vals if not np.isnan(v)]
        r_vals = [v for v in r_vals if not np.isnan(v)]
        feats[f"EEG_{band}_left"]  = np.mean(l_vals) if l_vals else np.nan
        feats[f"EEG_{band}_right"] = np.mean(r_vals) if r_vals else np.nan

    # 5. Global diffs (9): EEG_{b1}_minus_{b2}_global
    for b1, b2 in GLOBAL_DIFF_PAIRS:
        v1 = feats.get(f"EEG_{b1}_global", np.nan)
        v2 = feats.get(f"EEG_{b2}_global", np.nan)
        feats[f"EEG_{b1}_minus_{b2}_global"] = (
            v1 - v2 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)

    # 6. Regional diffs (19): EEG_{b1}_minus_{b2}_{region}
    for b1, b2, region in REGIONAL_DIFF_PAIRS:
        v1 = feats.get(f"EEG_{b1}_{region}", np.nan)
        v2 = feats.get(f"EEG_{b2}_{region}", np.nan)
        feats[f"EEG_{b1}_minus_{b2}_{region}"] = (
            v1 - v2 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)

    # 7. Log-ratios (30): EEG_log_{num}_{den}_ratio_{region}
    #    Uses raw (non z-scored) values with abs() — exact match with training
    for region, ch_list in REGION_MAP.items():
        for num, den in LOG_RATIO_PAIRS:
            raw_num = [d_raw.get(f"EXG_Channel_{ch}_{num}_power", np.nan) for ch in ch_list]
            raw_den = [d_raw.get(f"EXG_Channel_{ch}_{den}_power", np.nan) for ch in ch_list]
            raw_num = [v for v in raw_num if not np.isnan(v)]
            raw_den = [v for v in raw_den if not np.isnan(v)]
            if raw_num and raw_den:
                v_num = abs(np.mean(raw_num))
                v_den = abs(np.mean(raw_den))
                feats[f"EEG_log_{num}_{den}_ratio_{region}"] = np.log((v_num + AUX) / (v_den + AUX))
            else:
                feats[f"EEG_log_{num}_{den}_ratio_{region}"] = np.nan

    return feats


def get_eeg_feature_names():
    dummy = {f"EXG_Channel_{ch}_{band}_power": 0.0
             for ch in range(N_CH) for band in BANDS}
    return list(extract_eeg_user_features(dummy, d_raw=dummy).keys())


EEG_FEATURE_NAMES = get_eeg_feature_names()


def aggregate_eeg_features(eeg_by_user):
    result = {"EEG_n_users": 0}
    for feat in EEG_FEATURE_NAMES:
        result[feat] = np.nan

    if not isinstance(eeg_by_user, dict):
        return result

    valid_users = {u: f for u, f in eeg_by_user.items() if isinstance(f, dict)}
    result["EEG_n_users"] = len(valid_users)
    if not valid_users:
        return result

    user_rows = []
    for _, feats in valid_users.items():
        d_zscored = zscore_user_eeg(feats)
        user_rows.append(extract_eeg_user_features(d_zscored=d_zscored, d_raw=feats))

    user_df = pd.DataFrame(user_rows)
    for feat in EEG_FEATURE_NAMES:
        if feat in user_df.columns:
            vals = pd.to_numeric(user_df[feat], errors="coerce").dropna()
            if len(vals) > 0:
                result[feat] = float(vals.mean())

    return result


# ── ET Functions ─────────────────────────────────────────────────────────────

def aggregate_et_features(et_by_user):
    result = {"et_n_users": 0}
    for feat_name in ET_FEATURE_MAP.values():
        result[f"{feat_name}_mean"] = np.nan
        result[f"{feat_name}_std"]  = np.nan

    if not isinstance(et_by_user, dict):
        return result

    valid_users = {u: f for u, f in et_by_user.items() if isinstance(f, dict)}
    result["et_n_users"] = len(valid_users)
    if not valid_users:
        return result

    collected = {k: [] for k in ET_FEATURE_MAP}
    for _, feats in valid_users.items():
        for raw_key in ET_FEATURE_MAP:
            v = feats.get(raw_key, None)
            if v is not None:
                collected[raw_key].append(v)

    # Unit conversions (exact match with training notebook)
    collected["reaction_time"]              = [x / 1_000     for x in collected["reaction_time"]]
    collected["fixations_duration_mean_ns"] = [x / 1_000_000 for x in collected["fixations_duration_mean_ns"]]

    for raw_key, feat_name in ET_FEATURE_MAP.items():
        vals = pd.to_numeric(pd.Series(collected[raw_key]), errors="coerce").dropna()
        if len(vals) > 0:
            result[f"{feat_name}_mean"] = float(vals.mean())
            result[f"{feat_name}_std"]  = float(vals.std()) if len(vals) > 1 else 0.0

    return result


# ── Main pipeline ────────────────────────────────────────────────────────────

def build_test_dataframe(test_json_path):
    with open(test_json_path) as f:
        data = json.load(f)

    rows = []
    for _, record in data.items():
        row = {
            "id":         record["id_EXIST"],
            "lang":       record["lang"],
            "text":       record.get("text", ""),
            "image_file": record.get("meme", ""),
            "split":      record.get("split", ""),
        }
        sensorial  = record.get("sensorial", {})
        modalities = sensorial.get("modalities", {})
        row.update(aggregate_eeg_features(modalities.get("EEG", {}).get("by_user", {})))
        row.update(aggregate_et_features(modalities.get("ET",  {}).get("by_user", {})))
        rows.append(row)

    return pd.DataFrame(rows)


def align_and_impute(df_test, df_train):
    eeg_train = sorted([c for c in df_train.columns if c.startswith("EEG_")])
    et_train  = sorted([c for c in df_train.columns if c.startswith("et_")])
    meta_cols = ["id", "lang", "text", "image_file", "split"]
    feature_cols = eeg_train + et_train

    train_means = df_train[feature_cols].mean()

    df_out = df_test[meta_cols].copy()
    missing = []
    for col in feature_cols:
        if col in df_test.columns:
            df_out[col] = df_test[col].values
        else:
            df_out[col] = np.nan
            missing.append(col)

    if missing:
        print(f"  [WARN] {len(missing)} cols missing → imputed. Sample: {missing[:3]}")

    n_before = df_out[feature_cols].isna().sum().sum()
    df_out[feature_cols] = df_out[feature_cols].fillna(train_means)
    n_after  = df_out[feature_cols].isna().sum().sum()
    print(f"  Imputed {n_before} NaNs ({n_after} remaining)")

    # Drop EEG_n_users — same as training notebook
    df_out.drop(columns=["EEG_n_users"], inplace=True, errors="ignore")

    return df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", default="data/processed/train_model_ready.parquet")
    parser.add_argument("--test_json",     default="EXIST2026_test_clean.json")
    parser.add_argument("--output",        default="data/processed/test_model_ready.parquet")
    args = parser.parse_args()

    print("=" * 60)
    print("EXIST 2026 — Test Set Preprocessing (v2)")
    print("=" * 60)
    print(f"  EEG features expected: {len(EEG_FEATURE_NAMES)}")

    print(f"\n[1/3] Parsing: {args.test_json}")
    df_test_raw = build_test_dataframe(args.test_json)
    print(f"  Memes: {len(df_test_raw)} | {df_test_raw['lang'].value_counts().to_dict()}")

    print(f"\n[2/3] Loading train parquet: {args.train_parquet}")
    df_train = pd.read_parquet(args.train_parquet)
    print(f"  Train: {df_train.shape}")

    print(f"\n[3/3] Aligning + imputing")
    df_test = align_and_impute(df_test_raw, df_train)

    # Validate
    eeg_test  = sorted([c for c in df_test.columns if c.startswith("EEG_")])
    eeg_train_v = sorted([c for c in df_train.columns if c.startswith("EEG_")])
    et_test   = sorted([c for c in df_test.columns if c.startswith("et_")])
    et_train_v  = sorted([c for c in df_train.columns if c.startswith("et_")])
    print(f"\n── Validation ──────────────────────────────────")
    print(f"  EEG: train={len(eeg_train_v)} test={len(eeg_test)} "
          f"{'✓ OK' if eeg_train_v == eeg_test else '✗ MISMATCH'}")
    print(f"  ET:  train={len(et_train_v)} test={len(et_test)} "
          f"{'✓ OK' if et_train_v == et_test else '✗ MISMATCH'}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_test.to_parquet(args.output, index=False)
    print(f"\n✓ Saved: {args.output}")
    print(f"  Shape: {df_test.shape}")
    print(f"  NaN: {df_test.isna().sum().sum()}")
    print(df_test[["id", "lang", "text"]].head(3).to_string())


if __name__ == "__main__":
    main()
