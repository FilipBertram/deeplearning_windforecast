#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor



# Defaults


DEFAULTS = {
    "learning_rate": 0.02,
    "n_grid": [200, 400, 600, 800, 1200, 1500, 3000],

    "max_folds": 8,
    "min_train": 500,
    "min_val": 80,
    "max_depth": 6,
    "min_child_weight": 6.0,
    "gamma": 0.5,
    "subsample": 0.8,
    "colsample": 0.8,
    "reg_lambda": 4.0,
    "reg_alpha": 0.0,


    "seed": 42,

   
    "calibration_days": 120,
    "alpha": 0.1,
    "regime_conformal": True,

   
    "n_jobs": max(1, os.cpu_count() or 1),
}


# Utilities


def parse_n_grid(s: str | None) -> List[int]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []

    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(int(p))
    return out


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bias(y_true, y_pred) -> float:
    return float(np.mean(y_pred - y_true))


def angle_to_sin_cos(deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)


def sin_cos_to_angle_deg(sinv: np.ndarray, cosv: np.ndarray) -> np.ndarray:
    ang = np.rad2deg(np.arctan2(sinv, cosv))
    ang = (ang + 360.0) % 360.0
    return ang


def angular_error_deg(y_true_deg: np.ndarray, y_pred_deg: np.ndarray) -> np.ndarray:
    # minimal circular difference in degrees
    d = (y_pred_deg - y_true_deg + 180.0) % 360.0 - 180.0
    return np.abs(d)


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def plot_scatter(y_true, y_pred, out_png: Path, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=6)
    mn = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    mx = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_timeseries(df: pd.DataFrame, time_col: str, cols: List[str], out_png: Path, title: str) -> None:
    plt.figure(figsize=(10, 4))
    for c in cols:
        if c in df.columns:
            plt.plot(df[time_col], df[c], label=c)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_hist(values: np.ndarray, out_png: Path, title: str, xlabel: str) -> None:
    plt.figure()
    plt.hist(values[~np.isnan(values)], bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def pick_folds_evenly(folds: List[dict], max_folds: int) -> List[dict]:
    if max_folds <= 0 or len(folds) <= max_folds:
        return folds
    idx = np.linspace(0, len(folds) - 1, num=max_folds, dtype=int)
    return [folds[i] for i in idx]


def detect_targets(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {
        "speed": "obs_speed_mean",
        "gust": "obs_gust_max",
        "dir_deg": "obs_dir_deg",
    }
    missing = [v for v in mapping.values() if v not in df.columns]
    if missing:
        raise KeyError(f"Missing required target columns: {missing}. Available cols sample: {list(df.columns)[:30]}")
    return mapping


def build_feature_columns(df: pd.DataFrame) -> List[str]:

    drop = {"feature_time", "valid_time", "station", "lead_hours"}
    targets = {"obs_speed_mean", "obs_gust_max", "obs_dir_deg"}
    cols = [c for c in df.columns if c not in drop and c not in targets]
  
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def make_model(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    gamma: float,
    subsample: float,
    colsample: float,
    reg_lambda: float,
    reg_alpha: float,
    seed: int,
    n_jobs: int,
) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_child_weight=float(min_child_weight),
        gamma=float(gamma),
        subsample=float(subsample),
        colsample_bytree=float(colsample),
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        objective="reg:squarederror",
        random_state=int(seed),
        n_jobs=int(n_jobs),
        tree_method="hist",
    )



# Splits format


@dataclass
class Fold:
    fold: int
    train_end_time: pd.Timestamp
    val_start_time: pd.Timestamp
    val_end_time: pd.Timestamp


@dataclass
class Splits:
    time_col: str
    holdout_start_time: pd.Timestamp
    folds: List[Fold]


def load_splits(path: Path) -> Splits:
    obj = json.loads(path.read_text(encoding="utf-8"))
    time_col = obj.get("time_col", "feature_time")
    holdout_start = pd.to_datetime(obj["holdout"]["holdout_start_time"])
    folds = []
    for f in obj["walk_forward"]["folds"]:
        folds.append(
            Fold(
                fold=int(f["fold"]),
                train_end_time=pd.to_datetime(f["train_end_time"]),
                val_start_time=pd.to_datetime(f["val_start_time"]),
                val_end_time=pd.to_datetime(f["val_end_time"]),
            )
        )
    return Splits(time_col=time_col, holdout_start_time=holdout_start, folds=folds)



# Conformal


def make_regimes_speed_knots(speed: np.ndarray) -> np.ndarray:
    r = np.full(speed.shape, "medium", dtype=object)
    r[speed < 10.0] = "low"
    r[speed >= 20.0] = "high"
    return r


def conformal_q(abs_resid: np.ndarray, alpha: float) -> float:
    q = np.quantile(abs_resid[~np.isnan(abs_resid)], 1.0 - alpha)
    return float(q)

# Training logic per lead


def train_grid_for_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    splits: Splits,
    n_grid: List[int],
    args,
    folds_to_use: List[Fold],
    lead_dir: Path,
    target_name: str,
) -> Tuple[int, pd.DataFrame]:
  
    rows = []
    for n_est in n_grid:
        fold_metrics = []
        for f in folds_to_use:
            train_mask = df[splits.time_col] <= f.train_end_time
            val_mask = (df[splits.time_col] >= f.val_start_time) & (df[splits.time_col] <= f.val_end_time)

            train_df = df.loc[train_mask]
            val_df = df.loc[val_mask]

            if len(train_df) < args.min_train or len(val_df) < args.min_val:
                continue

            X_tr = train_df[feature_cols]
            y_tr = train_df[y_col].values
            X_va = val_df[feature_cols]
            y_va = val_df[y_col].values

            model = make_model(
                n_estimators=n_est,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                min_child_weight=args.min_child_weight,
                gamma=args.gamma,
                subsample=args.subsample,
                colsample=args.colsample,
                reg_lambda=args.reg_lambda,
                reg_alpha=args.reg_alpha,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            model.fit(X_tr, y_tr)
            p_va = model.predict(X_va)

            m = {
                "target": target_name,
                "n_estimators": int(n_est),
                "fold": int(f.fold),
                "train_end_time": str(f.train_end_time),
                "val_start_time": str(f.val_start_time),
                "val_end_time": str(f.val_end_time),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "val_rmse": rmse(y_va, p_va),
                "val_mae": float(mean_absolute_error(y_va, p_va)),
                "val_bias": bias(y_va, p_va),
                "val_r2": float(r2_score(y_va, p_va)),
            }
            fold_metrics.append(m)
            rows.append(m)

       
        if fold_metrics:
            mean_rmse = float(np.mean([x["val_rmse"] for x in fold_metrics]))
            print(f"    [GRID] {target_name} n={n_est} folds={len(fold_metrics)} mean_rmse={mean_rmse:.4f}", flush=True)
        else:
            print(f"    [GRID] {target_name} n={n_est} folds=0 (skipped)", flush=True)

    grid_df = pd.DataFrame(rows)
    if len(grid_df) == 0:
        raise RuntimeError(f"No folds usable for target={target_name}. Lower --min_train/--min_val or check splits.")

    agg = grid_df.groupby("n_estimators")["val_rmse"].mean().sort_values()
    best_n = int(agg.index[0])

    grid_df.to_csv(lead_dir / f"grid_metrics_{target_name}.csv", index=False)
    (lead_dir / f"grid_summary_{target_name}.csv").write_text(
        agg.reset_index().to_csv(index=False), encoding="utf-8"
    )

    print(f"    [GRID BEST] {target_name}: best_n={best_n} mean_val_rmse={float(agg.iloc[0]):.4f}", flush=True)
    return best_n, grid_df


def fit_final_and_eval_holdout(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    target_name: str,
    best_n: int,
    splits: Splits,
    args,
    lead_dir: Path,
) -> Dict[str, float]:
   
    time_col = splits.time_col

    pre_hold = df[df[time_col] < splits.holdout_start_time].copy()
    hold = df[df[time_col] >= splits.holdout_start_time].copy()

    if len(hold) == 0:
        raise RuntimeError("Holdout is empty. Increase holdout window or check holdout_start_time.")

    model = make_model(
        n_estimators=best_n,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample=args.colsample,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(pre_hold[feature_cols], pre_hold[y_col].values)

    p_ho = model.predict(hold[feature_cols])
    y_ho = hold[y_col].values

    out = {
        "holdout_rmse": rmse(y_ho, p_ho),
        "holdout_mae": float(mean_absolute_error(y_ho, p_ho)),
        "holdout_bias": bias(y_ho, p_ho),
        "holdout_r2": float(r2_score(y_ho, p_ho)),
        "n_estimators": int(best_n),
        "holdout_rows": int(len(hold)),
        "train_rows": int(len(pre_hold)),
    }

    pred_df = hold[[time_col, "lead_hours"]].copy()
    pred_df["y_true"] = y_ho
    pred_df["y_pred"] = p_ho
    pred_df.to_csv(lead_dir / f"pred_holdout_{target_name}.csv", index=False)

    plot_scatter(y_ho, p_ho, lead_dir / f"scatter_{target_name}.png", f"Holdout scatter {target_name}")
    plot_hist((p_ho - y_ho), lead_dir / f"resid_hist_{target_name}.png", f"Residuals {target_name}", "pred-true")
    # short timeseries sample
    ts = pred_df.sort_values(time_col).copy()
    ts = ts.iloc[: min(len(ts), 400)].copy()
    plot_timeseries(
        ts.rename(columns={"y_true": f"{target_name}_true", "y_pred": f"{target_name}_pred"}),
        time_col=time_col,
        cols=[f"{target_name}_true", f"{target_name}_pred"],
        out_png=lead_dir / f"timeseries_{target_name}.png",
        title=f"Holdout timeseries sample {target_name}",
    )

    # feature importance
    if args.save_feature_importance:
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        imp = pd.DataFrame({
            "feature": feature_cols,
            "gain": [gain.get(f, 0.0) for f in feature_cols],
            "weight": [weight.get(f, 0.0) for f in feature_cols],
        }).sort_values("gain", ascending=False)
        imp.to_csv(lead_dir / f"feature_importance_{target_name}.csv", index=False)
        top = imp.head(30).iloc[::-1]
        plt.figure(figsize=(8, 10))
        plt.barh(top["feature"], top["gain"])
        plt.title(f"Top feature gain ({target_name})")
        plt.tight_layout()
        plt.savefig(lead_dir / f"feature_importance_{target_name}.png", dpi=160)
        plt.close()

    # save model
    joblib.dump(model, lead_dir / f"model_{target_name}.joblib")

    return out


def conformal_intervals_for_holdout(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    target_name: str,
    model: XGBRegressor,
    splits: Splits,
    args,
    lead_dir: Path,
) -> Dict[str, float]:
  
    time_col = splits.time_col
    hold = df[df[time_col] >= splits.holdout_start_time].copy()
    pre_hold = df[df[time_col] < splits.holdout_start_time].copy()


    calib_start = splits.holdout_start_time - pd.Timedelta(days=int(args.calibration_days))
    calib = pre_hold[pre_hold[time_col] >= calib_start].copy()
    if len(calib) < 50:
        n = max(50, int(0.15 * len(pre_hold)))
        calib = pre_hold.iloc[-n:].copy()

    p_cal = model.predict(calib[feature_cols])
    abs_resid_cal = np.abs(p_cal - calib[y_col].values)

    p_ho = model.predict(hold[feature_cols])
    y_ho = hold[y_col].values
    abs_resid_ho = np.abs(p_ho - y_ho)

    if args.regime_conformal and ("obs_speed_mean" in df.columns):
        reg_cal = make_regimes_speed_knots(calib["obs_speed_mean"].values)
        reg_ho = make_regimes_speed_knots(hold["obs_speed_mean"].values)
        qs = {}
        for r in ["low", "medium", "high"]:
            mask = (reg_cal == r)
            if np.sum(mask) < 30:
                qs[r] = conformal_q(abs_resid_cal, args.alpha)
            else:
                qs[r] = conformal_q(abs_resid_cal[mask], args.alpha)

        q_ho = np.array([qs[r] for r in reg_ho], dtype=float)
        lo = p_ho - q_ho
        hi = p_ho + q_ho
        covered = (y_ho >= lo) & (y_ho <= hi)
        coverage = float(np.mean(covered))
        avg_width = float(np.mean(hi - lo))

        out = {
            "conformal_mode": "regime",
            "q_low": float(qs["low"]),
            "q_medium": float(qs["medium"]),
            "q_high": float(qs["high"]),
            "coverage": coverage,
            "avg_interval_width": avg_width,
            "calib_rows": int(len(calib)),
        }

        # save holdout with intervals
        out_df = hold[[time_col, "lead_hours"]].copy()
        out_df["y_true"] = y_ho
        out_df["y_pred"] = p_ho
        out_df["pi_lo"] = lo
        out_df["pi_hi"] = hi
        out_df["regime"] = reg_ho
        out_df.to_csv(lead_dir / f"pred_holdout_{target_name}_conformal.csv", index=False)

    else:
        q = conformal_q(abs_resid_cal, args.alpha)
        lo = p_ho - q
        hi = p_ho + q
        covered = (y_ho >= lo) & (y_ho <= hi)
        out = {
            "conformal_mode": "global",
            "q": float(q),
            "coverage": float(np.mean(covered)),
            "avg_interval_width": float(np.mean(hi - lo)),
            "calib_rows": int(len(calib)),
        }
        out_df = hold[[time_col, "lead_hours"]].copy()
        out_df["y_true"] = y_ho
        out_df["y_pred"] = p_ho
        out_df["pi_lo"] = lo
        out_df["pi_hi"] = hi
        out_df.to_csv(lead_dir / f"pred_holdout_{target_name}_conformal.csv", index=False)

    return out



# Main


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to parquet dataset")
    p.add_argument("--splits", required=True, help="Path to splits_12h.json")
    p.add_argument("--spot", required=True, help="Spot name (e.g., Holnis)")
    p.add_argument("--project_root", required=True, help="Project root for logs/models/reports")

    p.add_argument("--lead_min", type=int, default=1)
    p.add_argument("--lead_max", type=int, default=12)

    # overrides
    p.add_argument("--learning_rate", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--n_grid", type=str, default=",".join(map(str, DEFAULTS["n_grid"])))
    p.add_argument("--max_folds", type=int, default=DEFAULTS["max_folds"])
    p.add_argument("--min_train", type=int, default=DEFAULTS["min_train"])
    p.add_argument("--min_val", type=int, default=DEFAULTS["min_val"])

    p.add_argument("--max_depth", type=int, default=DEFAULTS["max_depth"])
    p.add_argument("--min_child_weight", type=float, default=DEFAULTS["min_child_weight"])
    p.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    p.add_argument("--subsample", type=float, default=DEFAULTS["subsample"])
    p.add_argument("--colsample", type=float, default=DEFAULTS["colsample"])
    p.add_argument("--reg_lambda", type=float, default=DEFAULTS["reg_lambda"])
    p.add_argument("--reg_alpha", type=float, default=DEFAULTS["reg_alpha"])

    p.add_argument("--calibration_days", type=int, default=DEFAULTS["calibration_days"])
    p.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    p.add_argument("--regime_conformal", action="store_true", default=DEFAULTS["regime_conformal"])

    p.add_argument("--save_feature_importance", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--n_jobs", type=int, default=DEFAULTS["n_jobs"])
    return p


def main():
    args = build_argparser().parse_args()
    args.n_grid = parse_n_grid(args.n_grid)

    project_root = Path(args.project_root).resolve()
    dataset_path = Path(args.dataset).resolve()
    splits_path = Path(args.splits).resolve()

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{args.spot}_multihorizon_" + uuid.uuid4().hex[:6]

    root_run_dir = mkdir(project_root / "logs" / "runs" / run_id)
    root_model_dir = mkdir(project_root / "models" / "runs" / run_id)
    root_fig_dir = mkdir(project_root / "reports" / "figures" / run_id)
    root_pred_dir = mkdir(project_root / "reports" / "predictions" / run_id)

    print(f"MULTIHORIZON_RUN: {run_id}", flush=True)
    print(f"ROOT_RUN_DIR: {root_run_dir}", flush=True)
    print(f"DATASET: {dataset_path}", flush=True)
    print(f"SPLITS: {splits_path}", flush=True)
    print(f"LEADS: {args.lead_min} → {args.lead_max}", flush=True)
    print(f"GRID: {args.n_grid}", flush=True)
    print(f"PARAMS: lr={args.learning_rate} max_depth={args.max_depth} subsample={args.subsample} colsample={args.colsample} lambda={args.reg_lambda}", flush=True)

    splits = load_splits(splits_path)
    print(f"Holdout start: {splits.holdout_start_time}", flush=True)
    folds_to_use = pick_folds_evenly(splits.folds, args.max_folds)
    print(f"Total folds in splits: {len(splits.folds)} | folds used: {len(folds_to_use)}", flush=True)

    # Load data
    df_all = pd.read_parquet(dataset_path)
    df_all = ensure_datetime(df_all, splits.time_col)

    # Filter spot
    if "station" in df_all.columns:
        
        df_all = df_all[df_all["station"].astype(str).str.contains(args.spot, case=False, na=False)].copy()

    # Filter leads range
    if "lead_hours" not in df_all.columns:
        raise KeyError("Dataset missing 'lead_hours' column.")
    df_all = df_all[(df_all["lead_hours"] >= args.lead_min) & (df_all["lead_hours"] <= args.lead_max)].copy()

    # detect targets & make direction sin/cos columns for modeling
    targets = detect_targets(df_all)
    df_all["dir_sin"], df_all["dir_cos"] = angle_to_sin_cos(df_all[targets["dir_deg"]].astype(float).values)

    df_all = df_all.dropna(subset=[targets["speed"], targets["gust"], targets["dir_deg"], splits.time_col])

    feature_cols = build_feature_columns(df_all)
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns found (numeric). Check your dataset columns.")

    print(f"Rows after spot+lead filter: {len(df_all)} | Features: {len(feature_cols)}", flush=True)

    meta = {
        "multihorizon_run_id": run_id,
        "spot": args.spot,
        "dataset": str(dataset_path),
        "splits": str(splits_path),
        "holdout_start": str(splits.holdout_start_time),
        "lead_min": args.lead_min,
        "lead_max": args.lead_max,
        "learning_rate": args.learning_rate,
        "n_grid": args.n_grid,
        "max_folds": args.max_folds,
        "min_train": args.min_train,
        "min_val": args.min_val,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "subsample": args.subsample,
        "colsample": args.colsample,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "calibration_days": args.calibration_days,
        "alpha": args.alpha,
        "regime_conformal": bool(args.regime_conformal),
        "python": sys.version,
        "platform": platform.platform(),
        "n_jobs": args.n_jobs,
    }
    save_json(meta, root_run_dir / "run_meta.json")

    summary_rows = []

    t0_all = time.time()

    for lead in range(args.lead_min, args.lead_max + 1):
        print("\n" + "=" * 28, flush=True)
        print(f"==== TRAIN LEAD {lead} ====", flush=True)
        lead_t0 = time.time()

        lead_df = df_all[df_all["lead_hours"] == lead].copy()
        lead_dir = mkdir(root_run_dir / f"lead_{lead:02d}")
        lead_model_dir = mkdir(root_model_dir / f"lead_{lead:02d}")
        lead_fig_dir = mkdir(root_fig_dir / f"lead_{lead:02d}")
        lead_pred_dir = mkdir(root_pred_dir / f"lead_{lead:02d}")

        # Ensure targets exist for this lead
        lead_df = lead_df.dropna(subset=[targets["speed"], targets["gust"], "dir_sin", "dir_cos"])

        # Quick fold feasibility check
        print(f"Lead rows: {len(lead_df)}", flush=True)

        # grid select for each target
        print("  [GRID SELECT] speed", flush=True)
        best_n_speed, _ = train_grid_for_target(
            lead_df, feature_cols, targets["speed"], splits, args.n_grid, args, folds_to_use, lead_dir, "speed"
        )
        print("  [GRID SELECT] gust", flush=True)
        best_n_gust, _ = train_grid_for_target(
            lead_df, feature_cols, targets["gust"], splits, args.n_grid, args, folds_to_use, lead_dir, "gust"
        )
        print("  [GRID SELECT] dir_sin", flush=True)
        best_n_sin, _ = train_grid_for_target(
            lead_df, feature_cols, "dir_sin", splits, args.n_grid, args, folds_to_use, lead_dir, "dir_sin"
        )
        print("  [GRID SELECT] dir_cos", flush=True)
        best_n_cos, _ = train_grid_for_target(
            lead_df, feature_cols, "dir_cos", splits, args.n_grid, args, folds_to_use, lead_dir, "dir_cos"
        )

        best_map = {"speed": best_n_speed, "gust": best_n_gust, "dir_sin": best_n_sin, "dir_cos": best_n_cos}
        save_json(best_map, lead_dir / "final_n_estimators.json")
        print(f"  Final n_estimators: {best_map}", flush=True)

 
        # speed
        s_metrics = fit_final_and_eval_holdout(
            lead_df, feature_cols, targets["speed"], "speed", best_n_speed, splits, args, lead_dir
        )
        # gust
        g_metrics = fit_final_and_eval_holdout(
            lead_df, feature_cols, targets["gust"], "gust", best_n_gust, splits, args, lead_dir
        )

        sin_metrics = fit_final_and_eval_holdout(
            lead_df, feature_cols, "dir_sin", "dir_sin", best_n_sin, splits, args, lead_dir
        )
        cos_metrics = fit_final_and_eval_holdout(
            lead_df, feature_cols, "dir_cos", "dir_cos", best_n_cos, splits, args, lead_dir
        )

        # compute angular MAE on holdout for direction
        hold = lead_df[lead_df[splits.time_col] >= splits.holdout_start_time].copy()
        m_sin = joblib.load(lead_dir / "model_dir_sin.joblib")
        m_cos = joblib.load(lead_dir / "model_dir_cos.joblib")
        p_sin = m_sin.predict(hold[feature_cols])
        p_cos = m_cos.predict(hold[feature_cols])
        pred_deg = sin_cos_to_angle_deg(p_sin, p_cos)
        true_deg = hold[targets["dir_deg"]].values.astype(float)
        ang_mae = float(np.mean(angular_error_deg(true_deg, pred_deg)))

        # Save combined holdout predictions
        pred_speed = pd.read_csv(lead_dir / "pred_holdout_speed.csv")
        pred_gust = pd.read_csv(lead_dir / "pred_holdout_gust.csv")

        combined = pred_speed[[splits.time_col, "lead_hours", "y_true", "y_pred"]].rename(
            columns={"y_true": "speed_true", "y_pred": "speed_pred"}
        )
        combined["gust_true"] = pred_gust["y_true"].values
        combined["gust_pred"] = pred_gust["y_pred"].values
        combined["dir_true_deg"] = true_deg
        combined["dir_pred_deg"] = pred_deg
        combined.to_csv(lead_pred_dir / "predictions_holdout_all.csv", index=False)

        # Conformal intervals
        conf = {}
        if args.alpha is not None:
            # reload fitted models
            m_speed = joblib.load(lead_dir / "model_speed.joblib")
            m_gust = joblib.load(lead_dir / "model_gust.joblib")
            conf["speed"] = conformal_intervals_for_holdout(
                lead_df, feature_cols, targets["speed"], "speed", m_speed, splits, args, lead_dir
            )
            conf["gust"] = conformal_intervals_for_holdout(
                lead_df, feature_cols, targets["gust"], "gust", m_gust, splits, args, lead_dir
            )

        lead_report = {
            "lead_hours": lead,
            "best_n_estimators": best_map,
            "speed": s_metrics,
            "gust": g_metrics,
            "dir_sin": sin_metrics,
            "dir_cos": cos_metrics,
            "direction": {"angular_mae_deg": ang_mae},
            "conformal": conf,
            "lead_duration_minutes": (time.time() - lead_t0) / 60.0,
        }
        save_json(lead_report, lead_dir / "lead_report.json")

        # Add to summary
        summary_rows.append({
            "lead_hours": lead,
            "speed_rmse": s_metrics["holdout_rmse"],
            "speed_mae": s_metrics["holdout_mae"],
            "speed_bias": s_metrics["holdout_bias"],
            "speed_r2": s_metrics["holdout_r2"],
            "gust_rmse": g_metrics["holdout_rmse"],
            "gust_mae": g_metrics["holdout_mae"],
            "gust_bias": g_metrics["holdout_bias"],
            "gust_r2": g_metrics["holdout_r2"],
            "dir_angular_mae_deg": ang_mae,
            "best_n_speed": best_n_speed,
            "best_n_gust": best_n_gust,
            "best_n_dir_sin": best_n_sin,
            "best_n_dir_cos": best_n_cos,
            "lead_duration_min": (time.time() - lead_t0) / 60.0,
            "conformal_speed_coverage": conf.get("speed", {}).get("coverage", np.nan),
            "conformal_gust_coverage": conf.get("gust", {}).get("coverage", np.nan),
            "conformal_speed_width": conf.get("speed", {}).get("avg_interval_width", np.nan),
            "conformal_gust_width": conf.get("gust", {}).get("avg_interval_width", np.nan),
        })

        print(
            f"Lead {lead} done. "
            f"speed_rmse={s_metrics['holdout_rmse']:.3f} gust_rmse={g_metrics['holdout_rmse']:.3f} "
            f"dir_mae_deg={ang_mae:.2f} "
            f"({(time.time()-lead_t0)/60:.2f} min)",
            flush=True
        )

    # write summary
    summary_df = pd.DataFrame(summary_rows).sort_values("lead_hours")
    summary_df.to_csv(root_run_dir / "summary_all_leads.csv", index=False)
    summary_df.to_csv(root_pred_dir / "summary_all_leads.csv", index=False)

    # plots across leads
    def plot_metric(metric: str, out_png: Path, title: str):
        plt.figure()
        plt.plot(summary_df["lead_hours"], summary_df[metric], marker="o")
        plt.xlabel("lead_hours")
        plt.ylabel(metric)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()

    plot_metric("speed_rmse", root_fig_dir / "rmse_speed_vs_lead.png", "Speed RMSE vs lead")
    plot_metric("gust_rmse", root_fig_dir / "rmse_gust_vs_lead.png", "Gust RMSE vs lead")
    plot_metric("speed_mae", root_fig_dir / "mae_speed_vs_lead.png", "Speed MAE vs lead")
    plot_metric("gust_mae", root_fig_dir / "mae_gust_vs_lead.png", "Gust MAE vs lead")
    plot_metric("dir_angular_mae_deg", root_fig_dir / "dir_mae_vs_lead.png", "Direction angular MAE vs lead")

    if "conformal_speed_coverage" in summary_df.columns:
        plot_metric("conformal_speed_coverage", root_fig_dir / "coverage_speed_vs_lead.png", "Speed conformal coverage vs lead")
        plot_metric("conformal_gust_coverage", root_fig_dir / "coverage_gust_vs_lead.png", "Gust conformal coverage vs lead")
        plot_metric("conformal_speed_width", root_fig_dir / "pi_width_speed_vs_lead.png", "Speed PI width vs lead")
        plot_metric("conformal_gust_width", root_fig_dir / "pi_width_gust_vs_lead.png", "Gust PI width vs lead")

    meta["duration_minutes"] = (time.time() - t0_all) / 60.0
    save_json(meta, root_run_dir / "run_meta.json")

    print("\nDONE.", flush=True)
    print(f"Summary: {root_run_dir / 'summary_all_leads.csv'}", flush=True)
    print(f"Figures: {root_fig_dir}", flush=True)
    print(f"Predictions: {root_pred_dir}", flush=True)


if __name__ == "__main__":
    main()