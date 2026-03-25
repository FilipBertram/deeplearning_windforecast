from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix} (use .parquet or .csv)")
    return df


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Dataset missing required column '{col}'. Columns: {list(df.columns)[:30]} ...")
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    return df


def build_splits_feature_time(
    df: pd.DataFrame,
    time_col: str,
    holdout_days: int,
    min_train_days: int,
    val_days: int,
    step_days: int,
) -> Dict:
   
    df = _ensure_datetime(df, time_col).sort_values(time_col).reset_index(drop=True)

    # unique feature_time grid
    times = pd.Index(df[time_col].drop_duplicates().sort_values())

    t_min = times.min()
    t_max = times.max()

    holdout_start = t_max - pd.Timedelta(days=holdout_days)

    trainval_times = times[times < holdout_start]
    holdout_times = times[times >= holdout_start]

    if len(trainval_times) == 0:
        raise ValueError("No train/val time left after holdout split. Reduce holdout_days.")


    def count_rows(t_start, t_end) -> int:
        m = (df[time_col] >= t_start) & (df[time_col] <= t_end)
        return int(m.sum())


    folds: List[Dict] = []


    train_end = t_min + pd.Timedelta(days=min_train_days)


    def last_time_leq(ts: pd.Timestamp) -> pd.Timestamp:
        idx = trainval_times.searchsorted(ts, side="right") - 1
        if idx < 0:
            return trainval_times[0]
        return trainval_times[idx]

    def first_time_gt(ts: pd.Timestamp) -> pd.Timestamp:
        idx = trainval_times.searchsorted(ts, side="right")
        if idx >= len(trainval_times):
            return trainval_times[-1]
        return trainval_times[idx]

    fold_id = 0
    while True:
        te = last_time_leq(train_end)
        vs = first_time_gt(te) 
        ve_target = vs + pd.Timedelta(days=val_days)
        ve = last_time_leq(ve_target)

       
        if vs >= holdout_start:
            break
        if ve < vs:
            break

        fold = {
            "fold": fold_id,
            "train_start_time": t_min.isoformat(),
            "train_end_time": te.isoformat(),
            "val_start_time": vs.isoformat(),
            "val_end_time": ve.isoformat(),
            "train_rows": count_rows(t_min, te),
            "val_rows": count_rows(vs, ve),
        }
        folds.append(fold)

        fold_id += 1
        train_end = te + pd.Timedelta(days=step_days)

        if te >= trainval_times[-1]:
            break

    out = {
        "dataset_path": str(Path.cwd() / "UNKNOWN").replace("/UNKNOWN", ""), 
        "time_col": time_col,
        "holdout": {
            "holdout_start_time": holdout_start.isoformat(),
            "holdout_days": holdout_days,
            "n_feature_times_holdout": int(len(holdout_times)),
            "n_rows_holdout": int(df[df[time_col] >= holdout_start].shape[0]),
        },
        "walk_forward": {
            "min_train_days": min_train_days,
            "val_days": val_days,
            "step_days": step_days,
            "n_folds": len(folds),
            "folds": folds,
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help="Path to parquet/csv dataset")
    ap.add_argument("--out", required=True, type=str, help="Output splits.json path")
    ap.add_argument("--time_col", default="feature_time", type=str, help="Column to split on (default: feature_time)")
    ap.add_argument("--holdout_days", default=14, type=int)
    ap.add_argument("--min_train_days", default=60, type=int)
    ap.add_argument("--val_days", default=7, type=int)
    ap.add_argument("--step_days", default=7, type=int)
    args = ap.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_dataset(dataset_path)
    splits = build_splits_feature_time(
        df=df,
        time_col=args.time_col,
        holdout_days=args.holdout_days,
        min_train_days=args.min_train_days,
        val_days=args.val_days,
        step_days=args.step_days,
    )
    splits["dataset_path"] = str(dataset_path)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print("Wrote:", out_path)
    print("Holdout start:", splits["holdout"]["holdout_start_time"])
    print("Folds:", splits["walk_forward"]["n_folds"])


if __name__ == "__main__":
    main()