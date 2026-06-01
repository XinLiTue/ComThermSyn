"""
Representative-day validation: select characteristic days from measured data
and compare simulated vs. measured heat-power profiles.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import _safe_numeric, _slug, infer_dt_hours as _infer_dt_hours

try:
    from scipy.stats import linregress as _linregress
except ImportError:
    _linregress = None

MethodBuilder = Callable[[pd.Series, pd.DataFrame, np.random.Generator], Dict[str, object]]


def _theoretical_points_per_day(times: pd.Series) -> int:
    dt_hours = _infer_dt_hours(times)
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        return 24
    return max(int(round(24.0 / dt_hours)), 1)


def _pick_nearest_day(day_df: pd.DataFrame, value_col: str, target: float) -> Optional[pd.Series]:
    if len(day_df) == 0 or not np.isfinite(target):
        return None
    work = day_df.copy()
    work["_distance"] = (_safe_numeric(work[value_col]) - float(target)).abs()
    return work.sort_values(["_distance", value_col, "day"]).iloc[0]


def _empty_selected_days(user_id: object, case_name: Optional[str] = None) -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "user_id", "case_name", "day", "day_type",
        "daily_q_sum", "daily_q_mean", "daily_q_max",
        "daily_Tamb_mean", "daily_S_sum", "nonzero_share", "n_points", "skip_reason",
    ])


def select_representative_validation_days(
    measured_df: pd.DataFrame,
    user_id: object,
    n_days: int = 5,
    *,
    analysis_mode: str = "fast",
    case_name: Optional[str] = None,
) -> pd.DataFrame:
    required = {"user_id", "time", "Tamb", "S", "Q"}
    missing = required - set(measured_df.columns)
    if missing:
        raise ValueError(f"measured_df missing required columns: {sorted(missing)}")

    part = measured_df.loc[measured_df["user_id"].astype(str) == str(user_id)].copy()
    if len(part) == 0:
        out = _empty_selected_days(user_id=user_id, case_name=case_name)
        out.loc[0, ["user_id", "case_name", "skip_reason"]] = [user_id, case_name, "no_measured_rows_for_user"]
        return out

    part["time"] = pd.to_datetime(part["time"], errors="coerce")
    part["Q"] = _safe_numeric(part["Q"])
    part["Tamb"] = _safe_numeric(part["Tamb"])
    part["S"] = _safe_numeric(part["S"])
    part = part.dropna(subset=["time", "Q", "Tamb", "S"]).sort_values("time")
    if len(part) == 0:
        out = _empty_selected_days(user_id=user_id, case_name=case_name)
        out.loc[0, ["user_id", "case_name", "skip_reason"]] = [user_id, case_name, "no_valid_measured_rows_after_cleaning"]
        return out

    part["day"] = part["time"].dt.floor("D")
    theoretical_points = _theoretical_points_per_day(part["time"])
    agg = (
        part.groupby("day", as_index=False)
        .agg(
            daily_q_sum=("Q", "sum"),
            daily_q_mean=("Q", "mean"),
            daily_q_max=("Q", "max"),
            daily_Tamb_mean=("Tamb", "mean"),
            daily_S_sum=("S", "sum"),
            n_points=("Q", "size"),
            nonzero_share=("Q", lambda s: float((_safe_numeric(s) > 1e-9).mean())),
        )
        .sort_values("day")
        .reset_index(drop=True)
    )
    agg["user_id"] = user_id
    agg["case_name"] = case_name
    agg["skip_reason"] = ""
    agg["is_valid"] = (
        (agg["n_points"] >= max(int(math.ceil(0.8 * theoretical_points)), 1))
        & (_safe_numeric(agg["daily_q_sum"]) > 0)
        & (_safe_numeric(agg["nonzero_share"]) > 0.1)
    )

    valid = agg.loc[agg["is_valid"]].copy()
    if len(valid) == 0:
        out = _empty_selected_days(user_id=user_id, case_name=case_name)
        out.loc[0, ["user_id", "case_name", "skip_reason"]] = [user_id, case_name, "no_valid_days_after_filters"]
        return out

    daily_q_q25 = float(valid["daily_q_sum"].quantile(0.25))
    daily_q_q50 = float(valid["daily_q_sum"].quantile(0.50))
    tamb_q25 = float(valid["daily_Tamb_mean"].quantile(0.25))
    solar_q25 = float(valid["daily_S_sum"].quantile(0.25))
    solar_q75 = float(valid["daily_S_sum"].quantile(0.75))

    picks: List[Tuple[str, Optional[pd.Series]]] = [
        ("peak_day", valid.sort_values(["daily_q_sum", "day"], ascending=[False, True]).iloc[0]),
        ("median_day", _pick_nearest_day(valid, "daily_q_sum", daily_q_q50)),
        ("low_nonzero_day", _pick_nearest_day(valid, "daily_q_sum", daily_q_q25)),
    ]
    cold_pool = valid.loc[valid["daily_Tamb_mean"] <= tamb_q25].copy()
    cold_cloudy_pool = cold_pool.loc[cold_pool["daily_S_sum"] <= solar_q25].copy()
    cold_sunny_pool = cold_pool.loc[cold_pool["daily_S_sum"] >= solar_q75].copy()
    picks.append((
        "cold_cloudy_day",
        cold_cloudy_pool.sort_values(["daily_Tamb_mean", "daily_S_sum", "day"]).iloc[0] if len(cold_cloudy_pool) else None,
    ))
    picks.append((
        "cold_sunny_day",
        cold_sunny_pool.sort_values(["daily_Tamb_mean", "daily_S_sum", "day"], ascending=[True, False, True]).iloc[0] if len(cold_sunny_pool) else None,
    ))

    if str(analysis_mode).lower() == "debug":
        wanted = ["peak_day", "median_day", "cold_cloudy_day"]
    else:
        wanted = ["peak_day", "median_day", "low_nonzero_day", "cold_cloudy_day", "cold_sunny_day"]
    wanted = wanted[: max(int(n_days), 1)]

    selected_rows: List[Dict[str, object]] = []
    used_days: set = set()
    fallback_pool = valid.copy()
    for day_type in wanted:
        picked = next((row for name, row in picks if name == day_type), None)
        if picked is not None:
            day_value = pd.Timestamp(picked["day"])
            if day_value in used_days:
                picked = None
        if picked is None:
            remaining = fallback_pool.loc[~fallback_pool["day"].isin(list(used_days))].copy()
            if len(remaining) == 0:
                continue
            if day_type == "cold_sunny_day":
                picked = remaining.sort_values(["daily_Tamb_mean", "daily_S_sum", "day"], ascending=[True, False, True]).iloc[0]
            elif day_type == "cold_cloudy_day":
                picked = remaining.sort_values(["daily_Tamb_mean", "daily_S_sum", "day"]).iloc[0]
            elif day_type == "low_nonzero_day":
                picked = _pick_nearest_day(remaining, "daily_q_sum", daily_q_q25)
            elif day_type == "median_day":
                picked = _pick_nearest_day(remaining, "daily_q_sum", daily_q_q50)
            else:
                picked = remaining.sort_values(["daily_q_sum", "day"], ascending=[False, True]).iloc[0]
        if picked is None:
            continue
        day_value = pd.Timestamp(picked["day"])
        if day_value in used_days:
            continue
        used_days.add(day_value)
        row = picked.drop(labels=["is_valid"], errors="ignore").to_dict()
        row["day_type"] = day_type
        selected_rows.append(row)

    if not selected_rows:
        out = _empty_selected_days(user_id=user_id, case_name=case_name)
        out.loc[0, ["user_id", "case_name", "skip_reason"]] = [user_id, case_name, "selection_failed_no_representative_days"]
        return out

    out = pd.DataFrame(selected_rows)
    ordered_cols = [
        "user_id", "case_name", "day", "day_type",
        "daily_q_sum", "daily_q_mean", "daily_q_max",
        "daily_Tamb_mean", "daily_S_sum", "nonzero_share", "n_points", "skip_reason",
    ]
    return out.reindex(columns=ordered_cols)


def _normalize_shape(values: np.ndarray) -> np.ndarray:
    peak = float(np.nanmax(values)) if len(values) else np.nan
    if not np.isfinite(peak) or peak <= 1e-9:
        return np.full(len(values), np.nan, dtype=float)
    return values / peak


def _peak_time_error_hours(
    index: pd.Index,
    sim_q: np.ndarray,
    measured_q: np.ndarray,
    nonzero_share: float,
) -> float:
    if not np.isfinite(nonzero_share) or nonzero_share <= 0.3 or len(index) == 0:
        return np.nan
    sim_peak_idx = int(np.nanargmax(sim_q))
    measured_peak_idx = int(np.nanargmax(measured_q))
    dt_hours = _infer_dt_hours(pd.Series(index))
    return float(abs(sim_peak_idx - measured_peak_idx) * dt_hours)


def compare_day_profiles(compare_df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    work = compare_df.copy().dropna(subset=["Q", "heat_kw"])
    if len(work) == 0:
        raise ValueError("compare_df has no overlapping non-null Q/heat_kw rows")

    measured_q = _safe_numeric(work["Q"]).to_numpy(dtype=float)
    sim_q = _safe_numeric(work["heat_kw"]).to_numpy(dtype=float)
    error = sim_q - measured_q
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    mean_measured = float(np.mean(measured_q)) if len(measured_q) else np.nan
    sum_measured = float(np.sum(measured_q))
    sum_sim = float(np.sum(sim_q))
    dt_hours = _infer_dt_hours(pd.Series(work.index))
    nonzero_share = float((measured_q > 1e-9).mean()) if len(measured_q) else np.nan

    measured_diff = np.diff(measured_q) if len(measured_q) >= 2 else np.array([], dtype=float)
    sim_diff = np.diff(sim_q) if len(sim_q) >= 2 else np.array([], dtype=float)
    ramp_mae = float(np.mean(np.abs(sim_diff - measured_diff))) if len(measured_diff) == len(sim_diff) and len(measured_diff) > 0 else np.nan

    measured_norm = _normalize_shape(measured_q)
    sim_norm = _normalize_shape(sim_q)
    shape_rmse = float(np.sqrt(np.nanmean((sim_norm - measured_norm) ** 2))) if np.isfinite(measured_norm).any() and np.isfinite(sim_norm).any() else np.nan

    correlation = np.nan
    if len(work) >= 2 and np.nanstd(measured_q) > 1e-9 and np.nanstd(sim_q) > 1e-9:
        correlation = float(np.corrcoef(measured_q, sim_q)[0, 1])

    peak_measured = float(np.nanmax(measured_q)) if len(measured_q) else np.nan
    peak_sim = float(np.nanmax(sim_q)) if len(sim_q) else np.nan
    peak_time_error = _peak_time_error_hours(work.index, sim_q, measured_q, nonzero_share)

    metrics = {
        "mae_kw": mae,
        "rmse_kw": rmse,
        "cv_rmse_pct_day": float(100.0 * rmse / max(mean_measured, 1e-9)),
        "nmbe_pct_day": float(100.0 * np.mean(error) / max(mean_measured, 1e-9)),
        "daily_kwh_error_pct": float(100.0 * (sum_sim - sum_measured) / max(sum_measured, 1e-9)),
        "peak_kw_error_pct": float(100.0 * (peak_sim - peak_measured) / max(peak_measured, 1e-9)),
        "peak_time_error_hour": peak_time_error,
        "correlation": correlation,
        "ramp_mae_kw": ramp_mae,
        "shape_rmse_normalized": shape_rmse,
        "nonzero_share": nonzero_share,
        "dt_hours": dt_hours,
    }
    return metrics, work


def _prepare_day_weather(measured_day_df: pd.DataFrame) -> pd.DataFrame:
    return (
        measured_day_df[["time", "Tamb", "S"]]
        .copy()
        .rename(columns={"time": "datetime"})
        .assign(
            ta=lambda x: _safe_numeric(x["Tamb"]),
            qg=lambda x: _safe_numeric(x["S"]) * 1000.0,
        )
        .set_index("datetime")[["ta", "qg"]]
    )


def _select_day_measurements(
    measured_df: pd.DataFrame, user_id: object, day: pd.Timestamp
) -> pd.DataFrame:
    part = measured_df.loc[measured_df["user_id"].astype(str) == str(user_id)].copy()
    part["time"] = pd.to_datetime(part["time"], errors="coerce")
    part["day"] = part["time"].dt.floor("D")
    return part.loc[part["day"] == pd.Timestamp(day)].sort_values("time")


def _save_day_plot(compare_df: pd.DataFrame, plot_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    keep_cols = [
        col for col in [
            "Q", "generated_heat_kw",
            "conditional_median_baseline_heat_kw",
            "random_train_sample_baseline_heat_kw",
        ]
        if col in compare_df.columns
    ]
    plot_df = compare_df[keep_cols].copy().rename(columns={
        "Q": "measured_Q",
        "generated_heat_kw": "generated",
        "conditional_median_baseline_heat_kw": "conditional_median_baseline",
        "random_train_sample_baseline_heat_kw": "random_train_sample_baseline",
    })
    plot_df.plot(ax=ax, linewidth=1.4, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel("kW")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# On/off threshold and segment helpers
# ---------------------------------------------------------------------------

def get_on_thresh(q_series: "pd.Series", abs_kw: float = 0.5, rel_frac: float = 0.05) -> float:
    """Adaptive on/off threshold: max(abs_kw, rel_frac * peak)."""
    q = _safe_numeric(q_series).dropna()
    if len(q) == 0:
        return abs_kw
    return max(abs_kw, rel_frac * float(q.max()))


def get_segments(bool_series: "pd.Series") -> List[Tuple[bool, int, int]]:
    """
    Identify contiguous True/False runs.
    Returns list of (state, start_pos, end_pos) where end_pos is inclusive.
    """
    arr = np.asarray(bool_series, dtype=bool)
    if len(arr) == 0:
        return []
    segs: List[Tuple[bool, int, int]] = []
    cur = bool(arr[0])
    start = 0
    for i in range(1, len(arr)):
        if bool(arr[i]) != cur:
            segs.append((cur, start, i - 1))
            cur = bool(arr[i])
            start = i
    segs.append((cur, start, len(arr) - 1))
    return segs


# ---------------------------------------------------------------------------
# A + B layer: daily energy and switching-rhythm metrics
# ---------------------------------------------------------------------------

def compute_day_metrics(
    measured_q: "pd.Series",
    simulated_q: "pd.Series",
    dt_min: float = 15.0,
    on_thresh_kw: float = 0.5,
    run_b: bool = True,
) -> Dict[str, object]:
    """
    A layer: daily kWh and peak kW.
    B layer (run_b=True): on/off event counts and durations.
    """
    mq = _safe_numeric(measured_q).to_numpy(dtype=float)
    sq = _safe_numeric(simulated_q).to_numpy(dtype=float)
    dt_h = dt_min / 60.0

    meas_daily_kwh = float(np.nansum(mq) * dt_h)
    sim_daily_kwh = float(np.nansum(sq) * dt_h)
    meas_peak_kw = float(np.nanmax(mq)) if len(mq) and np.any(np.isfinite(mq)) else np.nan
    sim_peak_kw = float(np.nanmax(sq)) if len(sq) and np.any(np.isfinite(sq)) else np.nan

    result: Dict[str, object] = {
        "meas_daily_kwh": meas_daily_kwh,
        "sim_daily_kwh": sim_daily_kwh,
        "daily_kwh_err": sim_daily_kwh - meas_daily_kwh,
        "meas_peak_kw": meas_peak_kw,
        "sim_peak_kw": sim_peak_kw,
        "peak_kw_err": (
            sim_peak_kw - meas_peak_kw
            if np.isfinite(sim_peak_kw) and np.isfinite(meas_peak_kw)
            else np.nan
        ),
    }

    if not run_b:
        return result

    def _event_stats(q_arr: np.ndarray) -> Dict[str, object]:
        on_bool = pd.Series(q_arr >= on_thresh_kw)
        segs = get_segments(on_bool)
        on_segs = [(s, e) for state, s, e in segs if state]
        off_segs_all = [(s, e) for state, s, e in segs if not state]

        n_on = len(on_segs)
        on_ratio = float(on_bool.mean()) if len(on_bool) else np.nan
        mean_on_dur = float(np.mean([(e - s + 1) * dt_min for s, e in on_segs])) if on_segs else np.nan

        # off-segments strictly between first and last on-event
        if len(on_segs) >= 2:
            first_on_s = on_segs[0][0]
            last_on_e = on_segs[-1][1]
            internal_off = [(s, e) for s, e in off_segs_all if s > first_on_s and e < last_on_e]
        else:
            internal_off = []
        mean_off_dur = float(np.mean([(e - s + 1) * dt_min for s, e in internal_off])) if internal_off else np.nan

        # mean of per-event peak power
        event_peaks = [float(np.nanmax(q_arr[s: e + 1])) for s, e in on_segs if e < len(q_arr)]
        event_peak_kw = float(np.mean(event_peaks)) if event_peaks else np.nan

        return {
            "n_on_events": n_on,
            "heating_on_ratio": on_ratio,
            "mean_on_duration_min": mean_on_dur,
            "mean_off_duration_min": mean_off_dur,
            "event_peak_kw": event_peak_kw,
        }

    ms = _event_stats(mq)
    ss = _event_stats(sq)
    for k, v in ms.items():
        result[f"meas_{k}"] = v
    for k, v in ss.items():
        result[f"sim_{k}"] = v

    return result


# ---------------------------------------------------------------------------
# C layer: tau proxy from natural-cooling regression
# ---------------------------------------------------------------------------

def estimate_tau_proxy(
    all_days_df: "pd.DataFrame",
    t_in_col: str = "Tin",
    t_amb_col: str = "Tamb",
    q_col: str = "Q",
    on_thresh_kw: float = 0.5,
    dt_min: float = 15.0,
    min_off_steps: int = 20,
    min_off_segs: int = 5,
    min_r2: float = 0.4,
) -> Dict[str, object]:
    """
    Estimate thermal time constant tau (hours) from off-period temperature decay.
    Pools off-segments across all supplied days.

    Regression model: dT_in/dt = (-1/tau) * (T_in - T_amb)
    """
    _empty: Dict[str, object] = {
        "tau_proxy": np.nan, "R2": np.nan,
        "n_off_steps": 0, "n_off_segments": 0, "reliable": False,
    }

    df = all_days_df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    tin = _safe_numeric(df[t_in_col]).to_numpy(dtype=float) if t_in_col in df.columns else np.full(len(df), np.nan)
    tamb = _safe_numeric(df[t_amb_col]).to_numpy(dtype=float) if t_amb_col in df.columns else np.full(len(df), np.nan)
    q = _safe_numeric(df[q_col]).to_numpy(dtype=float) if q_col in df.columns else np.full(len(df), np.nan)

    # Mark valid consecutive steps (guard against time gaps)
    if "time" in df.columns:
        t_secs = df["time"].dt.total_seconds() if hasattr(df["time"].dt, "total_seconds") else None
        if t_secs is None:
            t_secs = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
        expected_dt_s = dt_min * 60.0
        time_diff_s = t_secs.diff().fillna(0).abs().to_numpy()
        valid_step = (np.abs(time_diff_s - expected_dt_s) < expected_dt_s * 0.6)
        valid_step[0] = False
    else:
        valid_step = np.ones(len(df), dtype=bool)
        valid_step[0] = False

    dt_h = dt_min / 60.0
    off_bool = pd.Series(q < on_thresh_kw)
    segs = get_segments(off_bool)
    off_segs_qual = [(s, e) for state, s, e in segs if state and (e - s + 1) >= 3]

    dT_dt_vals: List[float] = []
    delta_T_vals: List[float] = []

    for s_i, e_i in off_segs_qual:
        for i in range(s_i + 1, e_i + 1):
            if not valid_step[i]:
                continue
            if not (np.isfinite(tin[i]) and np.isfinite(tin[i - 1]) and np.isfinite(tamb[i - 1])):
                continue
            dT_dt_vals.append((tin[i] - tin[i - 1]) / dt_h)
            delta_T_vals.append(tin[i - 1] - tamb[i - 1])

    n_steps = len(dT_dt_vals)
    n_segs = len(off_segs_qual)

    if n_steps < min_off_steps or n_segs < min_off_segs:
        return {**_empty, "n_off_steps": n_steps, "n_off_segments": n_segs}

    x = np.array(delta_T_vals, dtype=float)
    y = np.array(dT_dt_vals, dtype=float)

    if _linregress is not None:
        slope, _intercept, r_val, _p, _se = _linregress(x, y)
        r2 = float(r_val ** 2)
    else:
        x_m, y_m = x.mean(), y.mean()
        ss_xx = float(np.sum((x - x_m) ** 2))
        ss_xy = float(np.sum((x - x_m) * (y - y_m)))
        slope = ss_xy / ss_xx if ss_xx > 1e-12 else np.nan
        if np.isfinite(slope):
            ss_res = float(np.sum((y - (slope * x + (y_m - slope * x_m))) ** 2))
            ss_tot = float(np.sum((y - y_m) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
        else:
            r2 = np.nan

    slope = float(slope) if np.isfinite(slope) else np.nan
    r2 = float(r2) if np.isfinite(r2) else np.nan

    tau_proxy = (-1.0 / slope) if (np.isfinite(slope) and slope < 0) else np.nan

    reliable = bool(
        np.isfinite(slope) and slope < 0
        and np.isfinite(r2) and r2 >= min_r2
        and n_steps >= min_off_steps
        and n_segs >= min_off_segs
    )

    return {
        "tau_proxy": tau_proxy,
        "R2": r2,
        "n_off_steps": n_steps,
        "n_off_segments": n_segs,
        "reliable": reliable,
    }


# ---------------------------------------------------------------------------
# D layer: RCAQ heat-balance residuals during ramp-up
# ---------------------------------------------------------------------------

def compute_ramp_residuals(
    measured_day_df: "pd.DataFrame",
    building_params: Dict[str, float],
    t_in_col: str = "Tin",
    t_amb_col: str = "Tamb",
    q_col: str = "Q",
    qg_col: str = "S",
    dt_min: float = 15.0,
    ramp_window: int = 3,
    on_thresh_kw: float = 0.5,
) -> Dict[str, object]:
    """
    Validate the RCAQ heat-balance equation at ramp-up onset.

    qg_col ('S') is expected in kW/m²; A is in m² → solar_kw = A * S.
    C in kWh/°C, R in °C/kW, Q in kW → dT_predicted in °C/h.
    """
    _empty: Dict[str, object] = {
        "ramp_mean_dT_measured": np.nan, "ramp_mean_dT_predicted": np.nan,
        "ramp_mean_residual": np.nan, "ramp_rmse_residual": np.nan,
        "ramp_n_events_used": 0, "ramp_n_steps_used": 0,
    }

    R = float(building_params.get("R", np.nan))
    C = float(building_params.get("C", np.nan))
    A = float(building_params.get("A", np.nan))
    Qint = float(building_params.get("Qint", 0.0))

    if not (np.isfinite(R) and np.isfinite(C) and np.isfinite(A)) or C <= 0 or R <= 0:
        return _empty

    df = measured_day_df.copy()
    if hasattr(df.index, "sort_values"):
        df = df.sort_index()
    df = df.reset_index(drop=True)

    tin = _safe_numeric(df[t_in_col]).to_numpy(dtype=float) if t_in_col in df.columns else np.full(len(df), np.nan)
    tamb = _safe_numeric(df[t_amb_col]).to_numpy(dtype=float) if t_amb_col in df.columns else np.full(len(df), np.nan)
    q_arr = _safe_numeric(df[q_col]).to_numpy(dtype=float) if q_col in df.columns else np.full(len(df), np.nan)
    s_arr = _safe_numeric(df[qg_col]).to_numpy(dtype=float) if qg_col in df.columns else np.zeros(len(df))

    dt_h = dt_min / 60.0
    on_bool = pd.Series(q_arr >= on_thresh_kw)
    segs = get_segments(on_bool)
    on_segs = [(s_i, e_i) for state, s_i, e_i in segs if state and (e_i - s_i + 1) >= ramp_window + 1]

    dT_meas_all: List[float] = []
    dT_pred_all: List[float] = []
    n_events = 0

    for s_idx, e_idx in on_segs:
        ramp_end = min(s_idx + ramp_window, e_idx)
        for t in range(s_idx + 1, ramp_end + 1):
            if t + 1 >= len(tin):
                continue
            if not all(np.isfinite([tin[t], tin[t + 1], tamb[t], q_arr[t]])):
                continue
            dT_m = (tin[t + 1] - tin[t]) / dt_h
            solar_kw = A * (s_arr[t] if np.isfinite(s_arr[t]) else 0.0)
            dT_p = (q_arr[t] - (tin[t] - tamb[t]) / R + solar_kw + Qint) / C
            dT_meas_all.append(dT_m)
            dT_pred_all.append(dT_p)
        n_events += 1

    if not dT_meas_all:
        return _empty

    dT_m_arr = np.array(dT_meas_all, dtype=float)
    dT_p_arr = np.array(dT_pred_all, dtype=float)
    resid = dT_m_arr - dT_p_arr

    return {
        "ramp_mean_dT_measured": float(np.mean(dT_m_arr)),
        "ramp_mean_dT_predicted": float(np.mean(dT_p_arr)),
        "ramp_mean_residual": float(np.mean(resid)),
        "ramp_rmse_residual": float(np.sqrt(np.mean(resid ** 2))),
        "ramp_n_events_used": n_events,
        "ramp_n_steps_used": len(dT_meas_all),
    }


def _default_comparison_row(base: Dict[str, object]) -> Dict[str, object]:
    row = dict(base)
    for method in ["generated", "conditional_median_baseline", "random_train_sample_baseline"]:
        for metric in ["mae_kw", "rmse_kw", "shape_rmse_normalized", "daily_kwh_error_pct", "peak_kw_error_pct"]:
            row[f"{method}_{metric}"] = np.nan
    row["generated_best_method_by_mae"] = False
    row["generated_best_method_by_shape"] = False
    return row


def run_representative_day_validation(
    case_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    train_df: pd.DataFrame,
    *,
    method_builders: Dict[str, MethodBuilder],
    simulate_fn: Callable[..., pd.DataFrame],
    output_dir: Path,
    analysis_mode: str = "fast",
    n_days: Optional[int] = None,
    simulation_kwargs: Optional[Dict[str, object]] = None,
    random_state: int = 42,
    # ── new parameters ────────────────────────────────────────────────────────
    holdout_meta_df: Optional[pd.DataFrame] = None,
    run_a_energy_validation: bool = True,
    run_b_switching_validation: bool = True,
    run_c_tau_validation: bool = True,
    run_d_ramp_validation: bool = True,
    on_thresh_abs_kw: float = 0.5,
    on_thresh_rel_frac: float = 0.05,
    switching_ramp_window_steps: int = 3,
    tau_min_off_steps: int = 20,
    tau_min_off_segs: int = 5,
    tau_min_r2: float = 0.4,
) -> Dict[str, pd.DataFrame]:
    simulation_kwargs = {} if simulation_kwargs is None else dict(simulation_kwargs)
    rep_dir = Path(output_dir) / "representative_days"
    rep_dir.mkdir(parents=True, exist_ok=True)

    selected_days_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    day_metrics_rows: List[Dict[str, object]] = []
    tau_rows: List[Dict[str, object]] = []
    ramp_rows: List[Dict[str, object]] = []

    if n_days is None:
        n_days = 3 if str(analysis_mode).lower() == "debug" else 5

    rng_master = np.random.default_rng(random_state)

    for case_idx, (_, case_row) in enumerate(case_df.iterrows(), start=1):
        user_id = case_row["user_id"]
        case_name = case_row.get("case_name", f"case_{case_idx}")
        selected_days = select_representative_validation_days(
            measured_df=measured_df,
            user_id=user_id,
            n_days=n_days,
            analysis_mode=analysis_mode,
            case_name=case_name,
        )
        selected_days_rows.append(selected_days.copy())
        if len(selected_days) == 0 or "day_type" not in selected_days.columns:
            continue

        train_excl = train_df.loc[train_df["user_id"].astype(str) != str(user_id)].copy()
        if len(train_excl) == 0:
            for _, day_row in selected_days.iterrows():
                summary_rows.append({
                    "case_name": case_name, "user_id": user_id,
                    "day": day_row.get("day"), "day_type": day_row.get("day_type"),
                    "method": "generated", "status": "skipped_empty_leave_user_out_train_df",
                })
            continue

        method_payloads: Dict[str, Dict[str, object]] = {}
        for method_name, builder in method_builders.items():
            method_rng = np.random.default_rng(int(rng_master.integers(0, 1_000_000_000)))
            try:
                payload = builder(case_row.copy(), train_excl.copy(), method_rng)
                if not isinstance(payload, dict):
                    payload = {"status": "skipped_invalid_builder_payload"}
            except Exception as exc:
                payload = {"status": "failed_builder_exception", "skip_reason": str(exc)}
            payload.setdefault("status", "ok")
            method_payloads[method_name] = payload

        # ── C layer: estimate tau_proxy once per building (all days) ─────────
        _tau_result: Optional[Dict[str, object]] = None
        if run_c_tau_validation:
            try:
                _user_all = measured_df.loc[
                    measured_df["user_id"].astype(str) == str(user_id)
                ].copy()
                if len(_user_all) > 0:
                    _q_thresh_c = get_on_thresh(
                        _user_all["Q"], abs_kw=on_thresh_abs_kw, rel_frac=on_thresh_rel_frac
                    )
                    _dt_min_c = _infer_dt_hours(
                        pd.to_datetime(_user_all["time"], errors="coerce")
                    ) * 60.0
                    if not np.isfinite(_dt_min_c) or _dt_min_c <= 0:
                        _dt_min_c = 15.0
                    _tau_result = estimate_tau_proxy(
                        all_days_df=_user_all,
                        t_in_col="Tin", t_amb_col="Tamb", q_col="Q",
                        on_thresh_kw=_q_thresh_c, dt_min=_dt_min_c,
                        min_off_steps=tau_min_off_steps,
                        min_off_segs=tau_min_off_segs,
                        min_r2=tau_min_r2,
                    )
            except Exception as exc:
                warnings.warn(f"estimate_tau_proxy failed for {case_name}: {exc}")

        # Add tau rows: one per method per case
        if run_c_tau_validation and _tau_result is not None:
            _tau_fit_val = np.nan
            if holdout_meta_df is not None and len(holdout_meta_df):
                _hmatch = holdout_meta_df.loc[
                    holdout_meta_df["user_id"].astype(str) == str(user_id)
                ]
                if len(_hmatch):
                    _tau_fit_val = float(_hmatch.iloc[0].get("tau_fit", np.nan))
                    if not np.isfinite(_tau_fit_val):
                        _tau_fit_val = np.nan

            _tau_proxy = float(_tau_result.get("tau_proxy", np.nan))
            _tau_reliable = bool(_tau_result.get("reliable", False))

            for method_name, payload in method_payloads.items():
                if payload.get("status") == "ok" and isinstance(payload.get("building"), dict):
                    bld = payload["building"]
                    _R = float(bld.get("R", np.nan))
                    _C = float(bld.get("C", np.nan))
                    _tau_gen = (_R * _C) if (np.isfinite(_R) and np.isfinite(_C)) else np.nan
                    tau_rows.append({
                        "case_id": case_name,
                        "method": method_name,
                        **_tau_result,
                        "tau_fit": _tau_fit_val,
                        "tau_gen": _tau_gen,
                        "tau_rel_err_vs_proxy": (
                            abs(_tau_gen - _tau_proxy) / _tau_proxy
                            if np.isfinite(_tau_gen) and np.isfinite(_tau_proxy) and _tau_proxy > 0 and _tau_reliable
                            else np.nan
                        ),
                        "tau_fit_rel_err_vs_proxy": (
                            abs(_tau_fit_val - _tau_proxy) / _tau_proxy
                            if np.isfinite(_tau_fit_val) and np.isfinite(_tau_proxy) and _tau_proxy > 0 and _tau_reliable
                            else np.nan
                        ),
                    })

        for _, day_row in selected_days.iterrows():
            if pd.notna(day_row.get("skip_reason")) and str(day_row.get("skip_reason")).strip():
                summary_rows.append({
                    "case_name": case_name, "user_id": user_id,
                    "day": day_row.get("day"), "day_type": day_row.get("day_type"),
                    "method": "generated", "status": "skipped_day_selection_reason",
                    "skip_reason": day_row.get("skip_reason"),
                })
                continue

            measured_day = _select_day_measurements(measured_df, user_id, pd.Timestamp(day_row["day"]))
            measured_day = measured_day.dropna(subset=["time", "Tamb", "S", "Q"]).copy()
            if len(measured_day) == 0:
                for method_name in method_builders:
                    summary_rows.append({
                        "case_name": case_name, "user_id": user_id,
                        "day": day_row["day"], "day_type": day_row["day_type"],
                        "method": method_name, "status": "skipped_no_day_measurements",
                    })
                continue

            weather_input = _prepare_day_weather(measured_day)
            combined_compare = measured_day.set_index("time")[["Q", "Tin", "Tamb", "S"]].copy()
            per_method_metrics: Dict[str, Dict[str, object]] = {}

            for method_name, payload in method_payloads.items():
                base_row: Dict[str, object] = {
                    "case_name": case_name,
                    "user_id": user_id,
                    "day": pd.Timestamp(day_row["day"]),
                    "day_type": day_row["day_type"],
                    "method": method_name,
                    "n_points": int(day_row.get("n_points", len(measured_day))),
                    "daily_q_sum": float(day_row.get("daily_q_sum", np.nan)),
                    "daily_q_max": float(day_row.get("daily_q_max", np.nan)),
                    "daily_Tamb_mean": float(day_row.get("daily_Tamb_mean", np.nan)),
                    "daily_S_sum": float(day_row.get("daily_S_sum", np.nan)),
                }
                if payload.get("status") != "ok":
                    summary_rows.append({**base_row, "status": payload.get("status"), "skip_reason": payload.get("skip_reason")})
                    continue

                building = payload.get("building")
                if not isinstance(building, dict):
                    summary_rows.append({**base_row, "status": "skipped_missing_building_payload"})
                    continue

                try:
                    sim_profile = simulate_fn(building, weather_input, **simulation_kwargs)
                except Exception as exc:
                    summary_rows.append({**base_row, "status": "failed_simulation_exception", "skip_reason": str(exc)})
                    continue

                if "datetime" in sim_profile.columns:
                    sim_profile = sim_profile.set_index("datetime")
                compare_df = combined_compare.join(sim_profile[["heat_kw", "indoor_temp_C"]], how="inner").dropna(subset=["Q", "heat_kw"])
                if len(compare_df) == 0:
                    summary_rows.append({**base_row, "status": "skipped_no_time_overlap"})
                    continue

                metrics, compare_df = compare_day_profiles(compare_df)
                summary_row: Dict[str, object] = {
                    **base_row,
                    **metrics,
                    "status": "ok",
                    "R": float(building.get("R", np.nan)),
                    "C": float(building.get("C", np.nan)),
                    "A": float(building.get("A", np.nan)),
                    "Qint": float(building.get("Qint", np.nan)),
                    "Tin_min_sim": float(_safe_numeric(compare_df["indoor_temp_C"]).min()),
                    "Tin_max_sim": float(_safe_numeric(compare_df["indoor_temp_C"]).max()),
                    "Tin_mean_sim": float(_safe_numeric(compare_df["indoor_temp_C"]).mean()),
                    "Tin_comfort_violation_share": float(
                        ((_safe_numeric(compare_df["indoor_temp_C"]) < 18.0) | (_safe_numeric(compare_df["indoor_temp_C"]) > 25.0)).mean()
                    ),
                    "selection_method_note": payload.get("note"),
                }
                summary_rows.append(summary_row)
                per_method_metrics[method_name] = summary_row
                compare_export = compare_df.copy()
                compare_export[f"{method_name}_heat_kw"] = compare_export["heat_kw"]
                compare_export[f"{method_name}_Tin_sim"] = compare_export["indoor_temp_C"]
                combined_compare = combined_compare.join(
                    compare_export[[f"{method_name}_heat_kw", f"{method_name}_Tin_sim"]],
                    how="left",
                )
                time_slug = pd.Timestamp(day_row["day"]).strftime("%Y%m%d")
                ts_path = rep_dir / f"{_slug(case_name)}__{time_slug}__{_slug(day_row['day_type'])}__{_slug(method_name)}_timeseries.csv"
                compare_export.reset_index().to_csv(ts_path, index=False)

                # ── A + B layer ──────────────────────────────────────────────
                if run_a_energy_validation or run_b_switching_validation:
                    try:
                        _on_thresh_day = get_on_thresh(
                            compare_df["Q"], abs_kw=on_thresh_abs_kw, rel_frac=on_thresh_rel_frac
                        )
                        _dt_min_day = float(metrics.get("dt_hours", 0.25)) * 60.0
                        _dm = compute_day_metrics(
                            measured_q=compare_df["Q"],
                            simulated_q=compare_df["heat_kw"],
                            dt_min=_dt_min_day,
                            on_thresh_kw=_on_thresh_day,
                            run_b=run_b_switching_validation,
                        )
                        day_metrics_rows.append({
                            "case_id": case_name,
                            "method": method_name,
                            "day": pd.Timestamp(day_row["day"]),
                            "day_type": day_row["day_type"],
                            **_dm,
                        })
                    except Exception as exc:
                        warnings.warn(f"compute_day_metrics failed for {case_name}/{method_name}: {exc}")

                # ── D layer ──────────────────────────────────────────────────
                if run_d_ramp_validation and "S" in compare_df.columns:
                    try:
                        _on_thresh_ramp = get_on_thresh(
                            compare_df["Q"], abs_kw=on_thresh_abs_kw, rel_frac=on_thresh_rel_frac
                        )
                        _dt_min_ramp = float(metrics.get("dt_hours", 0.25)) * 60.0
                        _ramp = compute_ramp_residuals(
                            measured_day_df=compare_df,
                            building_params={
                                "R": building.get("R", np.nan),
                                "C": building.get("C", np.nan),
                                "A": building.get("A", np.nan),
                                "Qint": building.get("Qint", 0.0),
                            },
                            t_in_col="Tin", t_amb_col="Tamb", q_col="Q", qg_col="S",
                            dt_min=_dt_min_ramp,
                            ramp_window=switching_ramp_window_steps,
                            on_thresh_kw=_on_thresh_ramp,
                        )
                        ramp_rows.append({
                            "case_id": case_name,
                            "method": method_name,
                            "day": pd.Timestamp(day_row["day"]),
                            "day_type": day_row["day_type"],
                            **_ramp,
                        })
                    except Exception as exc:
                        warnings.warn(f"compute_ramp_residuals failed for {case_name}/{method_name}: {exc}")

            comparison_base = {
                "case_name": case_name,
                "user_id": user_id,
                "day": pd.Timestamp(day_row["day"]),
                "day_type": day_row["day_type"],
            }
            comp_row = _default_comparison_row(comparison_base)
            for method_name, result in per_method_metrics.items():
                for metric in ["mae_kw", "rmse_kw", "shape_rmse_normalized", "daily_kwh_error_pct", "peak_kw_error_pct"]:
                    comp_row[f"{method_name}_{metric}"] = result.get(metric, np.nan)
            mae_pairs = {m: r.get("mae_kw", np.nan) for m, r in per_method_metrics.items() if np.isfinite(r.get("mae_kw", np.nan))}
            shape_pairs = {m: r.get("shape_rmse_normalized", np.nan) for m, r in per_method_metrics.items() if np.isfinite(r.get("shape_rmse_normalized", np.nan))}
            comp_row["best_method_by_mae"] = min(mae_pairs, key=mae_pairs.get) if mae_pairs else None
            comp_row["generated_best_method_by_mae"] = comp_row["best_method_by_mae"] == "generated"
            comp_row["best_method_by_shape"] = min(shape_pairs, key=shape_pairs.get) if shape_pairs else None
            comp_row["generated_best_method_by_shape"] = comp_row["best_method_by_shape"] == "generated"
            comparison_rows.append(comp_row)

            plot_cols = [col for col in combined_compare.columns if col == "Q" or col.endswith("_heat_kw")]
            if "Q" in plot_cols and len(plot_cols) >= 2:
                plot_path = rep_dir / f"{_slug(case_name)}__{pd.Timestamp(day_row['day']).strftime('%Y%m%d')}__{_slug(day_row['day_type'])}_comparison.png"
                _save_day_plot(
                    combined_compare,
                    plot_path,
                    title=f"{case_name} | {day_row['day_type']} | {pd.Timestamp(day_row['day']).date()}",
                )

    selected_days_df = pd.concat(selected_days_rows, ignore_index=True) if selected_days_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    day_metrics_df = pd.DataFrame(day_metrics_rows)
    tau_df = pd.DataFrame(tau_rows)
    ramp_df = pd.DataFrame(ramp_rows)

    selected_days_df.to_csv(rep_dir / "selected_validation_days.csv", index=False)
    if len(summary_df):
        summary_df.loc[summary_df["method"] == "generated"].to_csv(rep_dir / "generated_day_validation_summary.csv", index=False)
        summary_df.loc[summary_df["method"] != "generated"].to_csv(rep_dir / "baseline_day_validation_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(rep_dir / "generated_day_validation_summary.csv", index=False)
        pd.DataFrame().to_csv(rep_dir / "baseline_day_validation_summary.csv", index=False)
    comparison_df.to_csv(rep_dir / "representative_day_comparison.csv", index=False)
    if len(day_metrics_df):
        day_metrics_df.to_csv(rep_dir / "day_metrics.csv", index=False)
    if len(tau_df):
        tau_df.to_csv(rep_dir / "tau_validation.csv", index=False)
    if len(ramp_df):
        ramp_df.to_csv(rep_dir / "ramp_residuals.csv", index=False)

    return {
        "selected_validation_days_df": selected_days_df,
        "representative_day_validation_summary_df": summary_df,
        "representative_day_validation_comparison_df": comparison_df,
        "day_metrics_df": day_metrics_df,
        "tau_df": tau_df,
        "ramp_df": ramp_df,
    }
