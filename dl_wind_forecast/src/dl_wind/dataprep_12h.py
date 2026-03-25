
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import re
import numpy as np
import pandas as pd


# FIXED ROOT
ROOT = Path("/Users/filipbertram/Hochschule/Master/Deep Learning/dl_wind_forecast/data/raw/windguru_data/aide")

OUT_DIRNAME = "ml_ready_12h_pipeline"

HORIZON_H = 12

OBS_RESAMPLE = "1h"

FORECAST_KEEP_VARS = ["WINDSPD", "WINDDIR", "GUST", "TMP", "RH", "SLP", "FLHGT"]
MEAS_KEEP_VARS = ["WINDSPD", "WINDDIR", "GUST", "TMP", "RH", "SLP"]


COVERAGE_THRESHOLD = 0.45


FINAL_FORECAST_VARS = ["windspd", "gust", "winddir"]  


KEEP_MODELS = None


# SQLite helpers
def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    return [r[0] for r in rows]


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM '{table}'", conn)


def pick_forecast_table(conn: sqlite3.Connection) -> str | None:
    for t in list_tables(conn):
        cols = table_columns(conn, t)
        lower = {c.lower() for c in cols}
        if "unixtime_init" in lower and "unixtime" in lower:
            return t
    return None


def pick_measurement_table(conn: sqlite3.Connection) -> str | None:
    for t in list_tables(conn):
        cols = table_columns(conn, t)
        lower = {c.lower() for c in cols}
        if "unixtime" in lower and "unixtime_init" not in lower and "windspd" in lower:
            return t
    return None


# Utils
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_model_name(db_path: Path) -> str:
    return re.sub(r"\W+", "_", db_path.stem.strip().lower())


def to_dt_unix_seconds(s: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="s", utc=True).dt.tz_convert(None)


def circular_mean_deg(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.nan
    rad = np.deg2rad(series.astype(float))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0)


def infer_high_freq(dt_index: pd.DatetimeIndex) -> bool:
    if len(dt_index) < 3:
        return False
    unix = (dt_index.view("int64") // 10**9).astype("int64")
    diffs = pd.Series(unix).sort_values().diff().dropna()
    return (not diffs.empty) and (diffs.median() < 3600)


def ci_map(columns: list[str]) -> dict[str, str]:
    return {c.lower(): c for c in columns}


# Load measurements
def load_measurements_from_data_db(data_db_path: Path) -> pd.DataFrame | None:
    try:
        conn = sqlite3.connect(data_db_path)
    except Exception as e:
        print(f"open data.db failed: {data_db_path} -> {e}")
        return None

    try:
        table = pick_measurement_table(conn)
        if table is None:
            print(f"no measurement table found in {data_db_path}; tables={list_tables(conn)}")
            return None
        df = read_table(conn, table)
    finally:
        conn.close()

    cmap = ci_map(df.columns.tolist())
    ut = cmap.get("unixtime")
    if ut is None:
        print(f"measurement missing unixtime: {data_db_path}")
        return None

    df["valid_time"] = to_dt_unix_seconds(df[ut])

    keep = ["valid_time"]
    for v in MEAS_KEEP_VARS:
        c = cmap.get(v.lower())
        if c is not None:
            keep.append(c)

    df = df[keep].dropna(subset=["valid_time"]).sort_values("valid_time")
    if df.empty:
        return None

    df = df.set_index("valid_time")
    high_freq = infer_high_freq(df.index)

    c_windspd = cmap.get("windspd")
    c_gust = cmap.get("gust")
    c_winddir = cmap.get("winddir")
    c_tmp = cmap.get("tmp")
    c_rh = cmap.get("rh")
    c_slp = cmap.get("slp")

    if high_freq:
        out = pd.DataFrame(index=df.resample(OBS_RESAMPLE).mean().index)

        if c_windspd in df.columns:
            out["obs_speed_mean"] = pd.to_numeric(df[c_windspd], errors="coerce").resample(OBS_RESAMPLE).mean()
        if c_gust in df.columns:
            out["obs_gust_max"] = pd.to_numeric(df[c_gust], errors="coerce").resample(OBS_RESAMPLE).max()
        if c_winddir in df.columns:
            out["obs_dir_deg"] = pd.to_numeric(df[c_winddir], errors="coerce").resample(OBS_RESAMPLE).apply(circular_mean_deg)

        if c_tmp in df.columns:
            out["obs_tmp_mean"] = pd.to_numeric(df[c_tmp], errors="coerce").resample(OBS_RESAMPLE).mean()
        if c_rh in df.columns:
            out["obs_rh_mean"] = pd.to_numeric(df[c_rh], errors="coerce").resample(OBS_RESAMPLE).mean()
        if c_slp in df.columns:
            out["obs_slp_mean"] = pd.to_numeric(df[c_slp], errors="coerce").resample(OBS_RESAMPLE).mean()

        return out.dropna(how="all")

    out = pd.DataFrame(index=df.index)
    if c_windspd in df.columns:
        out["obs_speed_mean"] = pd.to_numeric(df[c_windspd], errors="coerce")
    if c_gust in df.columns:
        out["obs_gust_max"] = pd.to_numeric(df[c_gust], errors="coerce")
    if c_winddir in df.columns:
        out["obs_dir_deg"] = pd.to_numeric(df[c_winddir], errors="coerce")
    if c_tmp in df.columns:
        out["obs_tmp_mean"] = pd.to_numeric(df[c_tmp], errors="coerce")
    if c_rh in df.columns:
        out["obs_rh_mean"] = pd.to_numeric(df[c_rh], errors="coerce")
    if c_slp in df.columns:
        out["obs_slp_mean"] = pd.to_numeric(df[c_slp], errors="coerce")

    return out.dropna(how="all")


# Load forecast db
def load_forecast_db(db_path: Path) -> pd.DataFrame | None:
    model = sanitize_model_name(db_path)
    if KEEP_MODELS is not None and model not in KEEP_MODELS:
        return None

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"open forecast db failed: {db_path} -> {e}")
        return None

    try:
        table = pick_forecast_table(conn)
        if table is None:
            print(f"no forecast table found in {db_path}; tables={list_tables(conn)}")
            return None
        df = read_table(conn, table)
    finally:
        conn.close()

    cmap = ci_map(df.columns.tolist())
    ut_init = cmap.get("unixtime_init")
    ut_valid = cmap.get("unixtime")
    if ut_init is None or ut_valid is None:
        print(f"forecast missing unixtime_init/unixtime in {db_path}")
        return None

    df["init_time"] = to_dt_unix_seconds(df[ut_init])
    df["valid_time"] = to_dt_unix_seconds(df[ut_valid])
    df["horizon_h"] = ((df["valid_time"] - df["init_time"]).dt.total_seconds() / 3600.0).round().astype("Int64")

    keep = ["init_time", "valid_time", "horizon_h"]
    for v in FORECAST_KEEP_VARS:
        c = cmap.get(v.lower())
        if c is not None:
            keep.append(c)

    df = df[keep].dropna(subset=["init_time", "valid_time", "horizon_h"])
    return df


# Coverage cleaning
def model_coverage(df: pd.DataFrame) -> pd.Series:
    fc_cols = [c for c in df.columns if c.startswith("fc_") and "__" in c]
    models = sorted({c.split("__", 1)[1] for c in fc_cols})
    cov = {}
    for m in models:
        m_cols = [c for c in fc_cols if c.endswith(f"__{m}")]
        cov[m] = df[m_cols].notna().any(axis=1).mean()
    return pd.Series(cov).sort_values(ascending=False)


def drop_sparse_models(df: pd.DataFrame, cov: pd.Series, threshold: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    drop_models = cov[cov < threshold].index.tolist()
    keep_models = cov[cov >= threshold].index.tolist()

    drop_cols = []
    for m in drop_models:
        drop_cols.extend([c for c in df.columns if c.startswith("fc_") and c.endswith(f"__{m}")])

    df2 = df.drop(columns=drop_cols, errors="ignore")
    return df2, keep_models, drop_models



# Build station features_12h_core
def build_station_core(station_dir: Path) -> pd.DataFrame | None:
    data_db = station_dir / "data.db"
    if not data_db.exists():
        print(f"⚠️ {station_dir.name}: data.db missing")
        return None

    obs = load_measurements_from_data_db(data_db)
    if obs is None or obs.empty:
        print(f"⚠️ {station_dir.name}: no obs")
        return None

    model_dbs = sorted([p for p in station_dir.glob("*.db") if p.name != "data.db"])
    if KEEP_MODELS is not None:
        model_dbs = [p for p in model_dbs if sanitize_model_name(p) in KEEP_MODELS]

    merged = None
    used = 0

    for db in model_dbs:
        model = sanitize_model_name(db)
        f = load_forecast_db(db)
        if f is None or f.empty:
            continue

        f = f[f["horizon_h"] == HORIZON_H].copy()
        if f.empty:
            continue

        f = f.drop(columns=["horizon_h"], errors="ignore")
        cols = [c for c in f.columns if c not in ("init_time", "valid_time")]
        f = f.rename(columns={c: f"fc_{c.lower()}__{model}" for c in cols})
        f = f.loc[:, ~f.columns.duplicated()]

        f = f.set_index(["init_time", "valid_time"]).sort_index()
        f = f.groupby(level=[0, 1]).mean(numeric_only=True)

        merged = f if merged is None else merged.join(f, how="outer")
        used += 1

    if merged is None or used == 0:
        print(f"{station_dir.name}: no forecasts")
        return None

    df = merged.reset_index().merge(
        obs.reset_index().rename(columns={"index": "valid_time"}),
        on="valid_time",
        how="inner",
    )
    df["station"] = station_dir.name
    df = df.rename(columns={"init_time": "feature_time"})
    df = df.set_index("feature_time").sort_index()
    return df



# Build final training dataset (direction sin/cos, targets)

def build_final_training_df(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()


    fc_cols = [c for c in df.columns if c.startswith("fc_") and "__" in c]
    keep_fc = []
    for c in fc_cols:
        left = c.split("__", 1)[0] 
        var = left.replace("fc_", "")
        if var in FINAL_FORECAST_VARS:
            keep_fc.append(c)

    keep_cols = ["valid_time", "station"] + keep_fc + ["obs_speed_mean", "obs_gust_max", "obs_dir_deg"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Convert forecast winddir columns to sin/cos
    dir_cols = [c for c in df.columns if c.startswith("fc_winddir__")]
    for c in dir_cols:
        rad = np.deg2rad(pd.to_numeric(df[c], errors="coerce"))
        df[c.replace("fc_winddir__", "fc_winddir_sin__")] = np.sin(rad)
        df[c.replace("fc_winddir__", "fc_winddir_cos__")] = np.cos(rad)
    df = df.drop(columns=dir_cols, errors="ignore")

    # Targets
    df["y_speed"] = pd.to_numeric(df["obs_speed_mean"], errors="coerce")
    df["y_gust"] = pd.to_numeric(df["obs_gust_max"], errors="coerce")

    rad = np.deg2rad(pd.to_numeric(df["obs_dir_deg"], errors="coerce"))
    df["y_dir_sin"] = np.sin(rad)
    df["y_dir_cos"] = np.cos(rad)

    df = df.drop(columns=["obs_speed_mean", "obs_gust_max", "obs_dir_deg"], errors="ignore")

    # Drop rows missing targets
    df = df.dropna(subset=["y_speed", "y_gust", "y_dir_sin", "y_dir_cos"])

    return df


def run_for_station(station_dir: Path) -> None:
    out_dir = station_dir / OUT_DIRNAME
    ensure_dir(out_dir)

    report = {
        "station": station_dir.name,
        "horizon_h": HORIZON_H,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "keep_models_for_loading": sorted(list(KEEP_MODELS)) if KEEP_MODELS else None,
    }

    print(f"\n==============================")
    print(f"{station_dir.name}")
    print(f"==============================")

    df_core = build_station_core(station_dir)
    if df_core is None or df_core.empty:
        print(" No core dataset produced.")
        return

    # Save core
    core_parquet = out_dir / "features_12h_core.parquet"
    core_csv = out_dir / "features_12h_core.csv"
    df_core.to_parquet(core_parquet, index=True)
    df_core.reset_index().to_csv(core_csv, index=False)

    report["core_shape"] = list(df_core.shape)

    # Drop all-NaN columns
    all_nan_cols = df_core.columns[df_core.isna().all()].tolist()
    df_tmp = df_core.drop(columns=all_nan_cols) if all_nan_cols else df_core
    report["dropped_all_nan_cols"] = all_nan_cols

    # Coverage clean
    cov = model_coverage(df_tmp)
    df_clean, kept_models, dropped_models = drop_sparse_models(df_tmp, cov, COVERAGE_THRESHOLD)

    report["model_coverage"] = cov.to_dict()
    report["kept_models_after_coverage"] = kept_models
    report["dropped_models_after_coverage"] = dropped_models
    report["clean_shape"] = list(df_clean.shape)

    clean_parquet = out_dir / "features_12h_core_clean.parquet"
    clean_csv = out_dir / "features_12h_core_clean.csv"
    df_clean.to_parquet(clean_parquet, index=True)
    df_clean.reset_index().to_csv(clean_csv, index=False)

    # Final training df
    df_final = build_final_training_df(df_clean)
    report["final_shape"] = list(df_final.shape)

    final_parquet = out_dir / "final_12h.parquet"
    final_csv = out_dir / "final_12h.csv"
    df_final.to_parquet(final_parquet, index=False)
    df_final.to_csv(final_csv, index=False)

    # Save report
    report_path = out_dir / "prep_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"core:  {core_parquet}")
    print(f"Clean: {clean_parquet}")
    print(f"final: {final_parquet}")
    print(f"report:{report_path}")


def main() -> None:
    station_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.lower().startswith("station")])
    if not station_dirs:
        raise RuntimeError(f"No station folders found under {ROOT}")

    for st in station_dirs:
        run_for_station(st)


if __name__ == "__main__":
    main()