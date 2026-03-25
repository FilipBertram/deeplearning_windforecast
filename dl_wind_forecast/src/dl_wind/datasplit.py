from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to parquet/csv with feature_time, lead_hours")
    ap.add_argument("--out", required=True, help="Output splits json")
    ap.add_argument("--lead_hours", type=int, default=12)

    ap.add_argument("--holdout_days", type=int, default=60)
    ap.add_argument("--train_days", type=int, default=180)
    ap.add_argument("--val_days", type=int, default=14)
    ap.add_argument("--step_days", type=int, default=1)

    args = ap.parse_args()

    ds = Path(args.dataset)
    if ds.suffix.lower() == ".parquet":
        df = pd.read_parquet(ds)
    else:
        df = pd.read_csv(ds)

    df["feature_time"] = pd.to_datetime(df["feature_time"], errors="coerce")
    df = df.dropna(subset=["feature_time", "lead_hours"])
    df["lead_hours"] = pd.to_numeric(df["lead_hours"], errors="coerce").astype(int)

    df = df[df["lead_hours"] == args.lead_hours].copy()
    df = df.sort_values("feature_time")

    ft = df["feature_time"].dropna().sort_values()
    if ft.empty:
        raise RuntimeError("No feature_time values after filtering. Check dataset/lead_hours.")

    
    ft_min = ft.min()
    ft_max = ft.max()

    holdout_start = ft_max - pd.Timedelta(days=args.holdout_days)

    # Train pool ends right before holdout_start
    train_pool = ft[ft < holdout_start]
    if train_pool.empty:
        raise RuntimeError("Train pool empty. Reduce holdout_days or check dataset range.")

    # Rolling folds
    folds = []
    step = pd.Timedelta(days=args.step_days)
    train_win = pd.Timedelta(days=args.train_days)
    val_win = pd.Timedelta(days=args.val_days)

    first_val_start = train_pool.min() + train_win
    last_val_end = train_pool.max()

    k = 0
    cur_val_start = first_val_start

    while True:
        cur_val_end = cur_val_start + val_win
        if cur_val_end > last_val_end:
            break

        train_end = cur_val_start - pd.Timedelta(seconds=1)

        folds.append(
            {
                "fold": k,
                "train_end_time": train_end.isoformat(),
                "val_start_time": cur_val_start.isoformat(),
                "val_end_time": cur_val_end.isoformat(),
            }
        )
        k += 1
        cur_val_start = cur_val_start + step

    out = {
        "dataset": str(ds),
        "lead_hours": args.lead_hours,
        "time_col": "feature_time",
        "holdout": {"holdout_start_time": holdout_start.isoformat()},
        "walk_forward": {
            "train_days": args.train_days,
            "val_days": args.val_days,
            "step_days": args.step_days,
            "folds": folds,
        },
    }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Wrote:", args.out)
    print("feature_time range:", ft_min, "→", ft_max)
    print("holdout_start:", holdout_start)
    print("folds:", len(folds))


if __name__ == "__main__":
    main()