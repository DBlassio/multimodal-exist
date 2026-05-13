"""
EXIST 2026 — Format Submissions
================================
Generates JSON files for EXIST 2026 challenge.

Output structure:
  exist2026_Cloud-17/
    task2_1_hard_Cloud-17_1   ← Run 1
    task2_1_soft_Cloud-17_1
    task2_2_hard_Cloud-17_1
    task2_2_soft_Cloud-17_1
    task2_3_hard_Cloud-17_1
    task2_3_soft_Cloud-17_1
    task2_1_hard_Cloud-17_2   ← Run 2
    ...                         (18 files total)

Usage:
  python format_submissions.py \
      --pred_dir   inference/predictions \
      --output_dir inference/submissions \
      --runs       gated ensemble_no_baseline cross_attention

  --runs accepts any combination of:
      baseline | gated | cross_attention | ensemble | ensemble_no_baseline
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
TEAM_NAME  = "Cloud-17"
TEST_CASE  = "EXIST2026"    # per official guidelines

TASK23_SOFT_COLS = [
    "p23_ideological",
    "p23_misogyny",
    "p23_objectification",
    "p23_sexual_violence",
    "p23_stereotyping",
]
TASK23_HARD_LABELS = [
    "IDEOLOGICAL-INEQUALITY",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "STEREOTYPING-DOMINANCE",
]

# Task 2.1 threshold
T21 = 0.50

# Task 2.3 per-category thresholds (lower for rare categories)
T23 = {
    "p23_ideological":     0.45,
    "p23_stereotyping":    0.45,
    "p23_objectification": 0.40,
    "p23_sexual_violence": 0.30,
    "p23_misogyny":        0.20,
}

# Available model predictions. Parquet filename mapping.
MODEL_FILES = {
    "baseline":             "baseline_raw.parquet",
    "gated":                "gated_raw.parquet",
    "cross_attention":      "cross_attention_raw.parquet",
    "ensemble":             "ensemble_raw.parquet",
    "ensemble_no_baseline": None,   # computed on the fly
}


# ── Prediction helpers ───────────────────────────────────────────────────────

def compute_ensemble_no_baseline(pred_dir):
    """Average Gated + CrossAttn predictions (no baseline)."""
    gated = pd.read_parquet(os.path.join(pred_dir, "gated_raw.parquet"))
    cross = pd.read_parquet(os.path.join(pred_dir, "cross_attention_raw.parquet"))

    prob_cols = ["p21", "p22_direct", "p22_judgemental",
                 "p23_ideological", "p23_misogyny", "p23_objectification",
                 "p23_sexual_violence", "p23_stereotyping"]

    df = gated[["id"]].copy()
    for col in prob_cols:
        df[col] = (gated[col].values + cross[col].values) / 2.0
    return df


def hard_21(p21): 
    return int(p21 >= T21)

def soft_21(p21):
    p = float(p21)
    return {"YES": round(p, 6), "NO": round(1.0 - p, 6)}

def hard_22(p21, p22_direct, p22_judgemental):
    if p21 < T21:
        return "NO"
    return "DIRECT" if p22_direct >= p22_judgemental else "JUDGEMENTAL"

def soft_22(p21, p22_direct, p22_judgemental):
    p = float(p21)
    # Conditional probs must sum to 1 within sexist group
    total_cond = float(p22_direct) + float(p22_judgemental)
    if total_cond > 0:
        p_d = float(p22_direct)  / total_cond
        p_j = float(p22_judgemental) / total_cond
    else:
        p_d, p_j = 0.5, 0.5
    return {
        "NO":           round(1.0 - p, 6),
        "DIRECT":       round(p * p_d, 6),
        "JUDGEMENTAL":  round(p * p_j, 6),
    }

def hard_23(p21, row):
    """Multi-label: NO if non-sexist, else categories above threshold."""
    if p21 < T21:
        return ["NO"]
    active = [
        label for col, label in zip(TASK23_SOFT_COLS, TASK23_HARD_LABELS)
        if float(row[col]) >= T23[col]
    ]
    # Fallback: always predict at least the most likely category
    if not active:
        best_col = max(TASK23_SOFT_COLS, key=lambda c: float(row[c]))
        best_idx = TASK23_SOFT_COLS.index(best_col)
        active = [TASK23_HARD_LABELS[best_idx]]
    return active

def soft_23(p21, row):
    """Multi-label soft:
      - NO = 1-p21, 
      - each cat = p21 × p(cat|sexist).
      """
    p = float(p21)
    out = {"NO": round(1.0 - p, 6)}
    for col, label in zip(TASK23_SOFT_COLS, TASK23_HARD_LABELS):
        out[label] = round(p * float(row[col]), 6)
    return out


# ── JSON builders ────────────────────────────────────────────────────────────

def build_task21(df, mode):
    records = []
    for _, row in df.iterrows():
        p = float(row["p21"])
        value = hard_21(p) if mode == "hard" else soft_21(p)
        # Hard value: integer→YES/NO string per guidelines
        if mode == "hard":
            value = "YES" if value == 1 else "NO"
        records.append({
            "test_case": TEST_CASE,
            "id":        str(row["id"]),
            "value":     value,
        })
    return records


def build_task22(df, mode):
    records = []
    for _, row in df.iterrows():
        p21 = float(row["p21"])
        p_d = float(row["p22_direct"])
        p_j = float(row["p22_judgemental"])
        value = (hard_22(p21, p_d, p_j) if mode == "hard"
                 else soft_22(p21, p_d, p_j))
        records.append({
            "test_case": TEST_CASE,
            "id":        str(row["id"]),
            "value":     value,
        })
    return records


def build_task23(df, mode):
    records = []
    for _, row in df.iterrows():
        p21 = float(row["p21"])
        value = (hard_23(p21, row) if mode == "hard"
                 else soft_23(p21, row))
        records.append({
            "test_case": TEST_CASE,
            "id":        str(row["id"]),
            "value":     value,
        })
    return records


# ── File writer ──────────────────────────────────────────────────────────────

def write_json(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"    ✓ {os.path.basename(path)}")


# ── Validation helper ─────────────────────────────────────────────────────────

def validate_with_pyevall(submission_dir):
    """Quick sanity check using PyEvALL if available."""
    try:
        from pyevall.evaluation import PyEvALLEvaluation
        from pyevall.utils.utils import PyEvALLUtils
        print("\n  PyEvALL validation (format only, no gold):")
        print("  Install gold labels to get actual ICM scores.")
    except ImportError:
        print("\n  PyEvALL not found — skipping validation.")
        print("  Run: pip install pyevall")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",   default="inference/predictions")
    parser.add_argument("--output_dir", default="inference/submissions")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["gated", "ensemble_no_baseline", "cross_attention"],
        choices=list(MODEL_FILES.keys()),
        help="Models to submit as runs (max 3). Order = run_id 1,2,3."
    )
    args = parser.parse_args()

    if len(args.runs) > 3:
        raise ValueError("Maximum 3 runs allowed per EXIST 2026 guidelines.")

    submission_folder = os.path.join(args.output_dir, f"exist2026_{TEAM_NAME}")
    os.makedirs(submission_folder, exist_ok=True)

    print("=" * 60)
    print(f"EXIST 2026 — Format Submissions — Team: {TEAM_NAME}")
    print("=" * 60)
    print(f"Runs: {args.runs}")
    print(f"T21={T21}  T23={T23}\n")

    # ── Load predictions ───────────────────────────────────────────────────
    dfs = {}
    for model in set(args.runs):
        if model == "ensemble_no_baseline":
            print(f"  Computing ensemble (Gated + CrossAttn)...")
            dfs[model] = compute_ensemble_no_baseline(args.pred_dir)
        else:
            fpath = os.path.join(args.pred_dir, MODEL_FILES[model])
            dfs[model] = pd.read_parquet(fpath)
        print(f"  Loaded: {model} ({len(dfs[model])} memes)")

    # ── Generate files per run ────────────────────────────────────────────
    for run_id, model in enumerate(args.runs, start=1):
        df = dfs[model]
        print(f"\nRun {run_id}: {model}")

        for task_num, (builder_hard, builder_soft) in enumerate(
            [(build_task21, build_task21),
             (build_task22, build_task22),
             (build_task23, build_task23)], start=1):

            task_str = f"task2_{task_num}"

            # Hard
            records = builder_hard(df, "hard")
            fname   = f"{task_str}_hard_{TEAM_NAME}_{run_id}"
            write_json(records, os.path.join(submission_folder, fname))

            # Soft
            records = builder_soft(df, "soft")
            fname   = f"{task_str}_soft_{TEAM_NAME}_{run_id}"
            write_json(records, os.path.join(submission_folder, fname))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generated {len(args.runs) * 6} files in:")
    print(f"  {submission_folder}/")
    print(f"\nFile list:")
    for f in sorted(os.listdir(submission_folder)):
        print(f"  {f}")

    # ── Prediction stats ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Prediction stats per run:")
    print(f"{'Run':<5} {'Model':<25} {'%Sexist':>8} {'%Direct':>8} {'%Judg.':>8}")
    print("-" * 60)
    for run_id, model in enumerate(args.runs, start=1):
        df = dfs[model]
        pct_s = (df["p21"] >= T21).mean() * 100
        sexist = df[df["p21"] >= T21]
        if len(sexist) > 0:
            pct_d = (sexist["p22_direct"] >= sexist["p22_judgemental"]).mean() * 100
            pct_j = 100 - pct_d
        else:
            pct_d = pct_j = 0
        print(f"  {run_id}    {model:<25} {pct_s:>7.1f}% {pct_d:>7.1f}% {pct_j:>7.1f}%")

    # ── Package for submission ────────────────────────────────────────────
    import shutil
    zip_path = os.path.join(args.output_dir, f"exist2026_{TEAM_NAME}")
    shutil.make_archive(zip_path, "zip", args.output_dir,
                        f"exist2026_{TEAM_NAME}")
    print(f"\n  Zipped: {zip_path}.zip")


if __name__ == "__main__":
    main()
