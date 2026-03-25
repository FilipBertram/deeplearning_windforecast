from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


#Config
DEFAULT_MODEL_DBS = [
    "gfs.db",
    "harmeu.db",
    "icon.db",
    "iconeu.db",
    "ifs.db",
    "metno.db",
    "wrfeuh.db",
]

FC_COLS = [
    "WINDSPD",
    "GUST",
    "WINDDIR",
    "TMP",
    "RH",
    "SLP",

    "LCDC",
    "APCP1",
    "APCP",
    "TCDC",
    "HCDC",
    "MCDC",
]


TARGET_COLS = ["obs_speed_mean", "obs_gust_max", "obs_dir_deg"]


# SQLite helpers
def _read_sqlite_table(db_path: Path, table: str) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()


def _list_tables(db_path: Path) -> List[str]:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in cur.fetchall()]
    finally:
        con.close()


def _find_table(db_path: Path, preferred: List[str]) -> Optional[str]:
    tables = _list_tables(db_path)
    for t in preferred:
        if t in tables:
            return t
    if len(tables) == 1:
        return tables[0]
    return None


def _add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
  
    out = df.copy()

    if "unixtime_init" in out.columns:
        out["feature_time"] = pd.to_datetime(out["unixtime_init"], unit="s", utc=True).dt.tz_convert(None)

    if "unixtime" in out.columns:
        out["valid_time"] = pd.to_datetime(out["unixtime"], unit="s", utc=True).dt.tz_convert(None)

    return out


def _rename_obs_cols(df: pd.DataFrame) -> pd.DataFrame:
  
    mapping = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "windspd":
            mapping[c] = "obs_speed_mean"
        elif cl == "gust":
            mapping[c] = "obs_gust_max"
        elif cl == "winddir":
            mapping[c] = "obs_dir_deg"
        else:
            mapping[c] = c
    return df.rename(columns=mapping)


# Build measurements
def build_measurements_hourly(station_dir: Path) -> pd.DataFrame:

    db_path = station_dir / "data.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing measurements db: {db_path}")

    table = _find_table(db_path, ["measurement", "measurements", "obs", "observation", "observations"])
    if table is None:
        raise RuntimeError(f"Could not find measurement table in {db_path}. tables={_list_tables(db_path)}")

    df = _read_sqlite_table(db_path, table)
    if df.empty:
        raise RuntimeError(f"Empty measurement table in {db_path}:{table}")

    df = _add_time_cols(df)
    if "valid_time" not in df.columns:
        raise RuntimeError(f"Measurements missing unixtime -> valid_time in {db_path}:{table}")

    keep = ["valid_time"]
    for c in ["WINDSPD", "WINDDIR", "GUST", "TMP", "RH", "SLP"]:
        if c in df.columns:
            keep.append(c)
    df = df[keep].copy()

    df = _rename_obs_cols(df)

    # hourly bucket
    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    df = df.dropna(subset=["valid_time"])
    df["valid_time"] = df["valid_time"].dt.floor("h")

    agg = {}
    if "obs_speed_mean" in df.columns:
        agg["obs_speed_mean"] = "mean"
    if "obs_gust_max" in df.columns:
        agg["obs_gust_max"] = "max"
    if "obs_dir_deg" in df.columns:
        agg["obs_dir_deg"] = "mean"

    obs_h = df.groupby("valid_time", as_index=False).agg(agg)
    return obs_h


# Build forecast for one model
def build_forecast_model(
    station_dir: Path,
    model_db: str,
    lead_min: int,
    lead_max: int,
    keep_cols: List[str],
) -> Optional[pd.DataFrame]:
    db_path = station_dir / model_db
    if not db_path.exists():
        print(f"[WARN] missing model db: {db_path}")
        return None

    table = _find_table(db_path, ["forecast", "Forecast"])
    if table is None:
        print(f"[WARN] no forecast table found in {db_path}. tables={_list_tables(db_path)}")
        return None

    df = _read_sqlite_table(db_path, table)
    if df.empty:
        print(f"[WARN] empty forecast table in {db_path}:{table}")
        return None

    df = _add_time_cols(df)
    if "feature_time" not in df.columns or "valid_time" not in df.columns:
        print(f"missing time cols in {db_path}:{table}")
        return None

    df["lead_hours"] = ((df["valid_time"] - df["feature_time"]).dt.total_seconds() / 3600.0).round().astype(int)
    df = df[(df["lead_hours"] >= lead_min) & (df["lead_hours"] <= lead_max)].copy()
    if df.empty:
        print(f" no rows in lead range for {db_path}")
        return None

    cols = ["feature_time", "valid_time", "lead_hours"]
    for c in keep_cols:
        if c in df.columns:
            cols.append(c)
    df = df[cols].copy()

    model_name = model_db.replace(".db", "")
    rename = {}
    for c in df.columns:
        if c in ("feature_time", "valid_time", "lead_hours"):
            continue
        rename[c] = f"fc_{c.lower()}__{model_name}"
    df = df.rename(columns=rename)

    return df


# Main builder
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--lead_min", type=int, default=1)
    ap.add_argument("--lead_max", type=int, default=12)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODEL_DBS))
    ap.add_argument("--drop_cloud_rain", action="store_true")
    args = ap.parse_args()

    station_dir = Path(args.station_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"== Station: {station_dir.name} ==")
    print("models:", [m.replace(".db", "") for m in models])
    print("leads:", args.lead_min, "→", args.lead_max)
    print("drop_cloud_rain:", bool(args.drop_cloud_rain))

    # 1) Measurements hourly
    obs_h = build_measurements_hourly(station_dir)
    print("Measurements hourly rows:", len(obs_h))

    # 2) Forecasts per model
    keep_cols = FC_COLS.copy()
    if args.drop_cloud_rain:
        pass

    parts = []
    used_models = []
    for m in models:
        fm = build_forecast_model(
            station_dir=station_dir,
            model_db=m,
            lead_min=args.lead_min,
            lead_max=args.lead_max,
            keep_cols=keep_cols,
        )
        if fm is None or fm.empty:
            continue
        parts.append(fm)
        used_models.append(m.replace(".db", ""))

    if not parts:
        raise RuntimeError(f"No forecast models readable for {station_dir}")

    print("Used models:", used_models)

    # 3) Outer join forecasts on (feature_time, valid_time, lead_hours)
    fc = parts[0]
    for p in parts[1:]:
        fc = fc.merge(p, on=["feature_time", "valid_time", "lead_hours"], how="outer")

    # 4) Join obs on valid_time
    df = fc.merge(obs_h, on="valid_time", how="left")

    # 5) Optional drop cloud/rain columns
    if args.drop_cloud_rain:
        drop_prefixes = (
            "fc_lcdc__",
            "fc_apcp1__",
            "fc_apcp__",
            "fc_tcdc__",
            "fc_hcdc__",
            "fc_mcdc__",
        )
        drop_cols = [c for c in df.columns if any(c.startswith(p) for p in drop_prefixes)]
        df = df.drop(columns=drop_cols, errors="ignore")
        print("Dropped cloud/rain cols:", len(drop_cols))

    # 6) Add station
    if "station" not in df.columns:
        df["station"] = station_dir.name

    # 7) Sort
    df = df.sort_values(["feature_time", "valid_time", "lead_hours"]).reset_index(drop=True)

    print("\nApplying target-cleaning (drop rows with missing obs targets):")
    print("Rows before:", len(df))
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing target cols {missing} (cannot clean). Columns: {list(df.columns)[:30]}...")

    df = df.dropna(subset=TARGET_COLS)
    print("Rows after:", len(df))
    for c in TARGET_COLS:
        print(f"null_ratio {c}: {df[c].isna().mean():.6f}")

    # 9) Write outputs
    out_parq = out_dir / "features_1to12_core_clean.parquet"
    out_csv = out_dir / "features_1to12_core_clean.csv"
    df.to_parquet(out_parq, index=False)
    df.to_csv(out_csv, index=False)

    print("\nWrote:", out_parq)
    print("Wrote:", out_csv)
    print("Rows:", len(df), "Cols:", df.shape[1])
    print("Leads:", sorted(df["lead_hours"].unique().tolist()))


if __name__ == "__main__":
    main()