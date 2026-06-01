"""
RC thermal simulation: heating profile, annual summary, screen weather.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DT_SECONDS,
    EXTREME_COLD_OUTDOOR_C,
    HEATING_MONTHS,
    PROFILE_YEAR,
    SHOULDER_SETPOINT,
    STANDARD_YEARS,
    WINTER_SETPOINT,
    CUSTOM_DAY_END_HOUR,
    CUSTOM_DAY_SETPOINT_C,
    CUSTOM_DAY_START_HOUR,
    CUSTOM_INITIAL_TEMP_C,
    CUSTOM_OTHER_SETPOINT_C,
    CUSTOM_SUMMER_MONTHS,
)


# ---------------------------------------------------------------------------
# Thermostat helpers (V3/V4 default: winter/shoulder logic)
# ---------------------------------------------------------------------------

def seasonal_setpoint(
    month: int,
    outdoor_temp: float,
    heating_months: List[int] = HEATING_MONTHS,
    winter_setpoint: float = WINTER_SETPOINT,
    shoulder_setpoint: float = SHOULDER_SETPOINT,
    extreme_cold: float = EXTREME_COLD_OUTDOOR_C,
) -> Optional[float]:
    """
    Return heating setpoint or None (heating off).

    - Heating months: winter_setpoint
    - Non-heating months: shoulder_setpoint only when outdoor < extreme_cold threshold
    """
    if month in heating_months:
        return winter_setpoint
    if outdoor_temp < extreme_cold:
        return shoulder_setpoint
    return None


# ---------------------------------------------------------------------------
# Core analytical RC simulation (V3/V4 default thermostat)
# ---------------------------------------------------------------------------

def simulate_heating_profile(
    R: float,
    C: float,
    A: float,
    Qint: float,
    area: float,
    weather_df: pd.DataFrame,
    initial_temp: float = 20.0,
    heating_months: List[int] = HEATING_MONTHS,
    winter_setpoint: float = WINTER_SETPOINT,
    shoulder_setpoint: float = SHOULDER_SETPOINT,
    extreme_cold: float = EXTREME_COLD_OUTDOOR_C,
) -> pd.DataFrame:
    """
    Simulate one building's heating demand using analytical RC discrete update.

    Weather DataFrame must have columns: datetime, ta (°C), qg (W/m²).
    Returns DataFrame indexed by datetime with heat_kw, heat_kwh, indoor_temp_C.
    """
    R = max(float(R), 1e-6)
    C = max(float(C), 1e-6)
    A = float(A)
    Qint = float(Qint)
    area = max(float(area), 1e-6)

    wdf = weather_df.copy()
    if "datetime" in wdf.columns:
        wdf = wdf.set_index("datetime")
    wdf.index = pd.to_datetime(wdf.index)
    wdf = wdf.sort_index()

    ta = wdf["ta"].to_numpy(dtype=float)
    qg = wdf["qg"].to_numpy(dtype=float)
    index = wdf.index

    # Infer timestep
    if len(index) >= 2:
        dt_sec = float((index[1] - index[0]).total_seconds())
    else:
        dt_sec = float(DT_SECONDS)
    dt_hours = dt_sec / 3600.0

    tau_sec = max(R * C, 1e-6)
    alpha = float(np.exp(-dt_sec / tau_sec))

    tin = float(initial_temp)
    rows = []

    for i in range(len(index)):
        ta_i = float(ta[i])
        qg_i = float(qg[i])
        month = int(index[i].month)
        solar_kw = A * qg_i / 1000.0  # qg in W/m², A in m² → kW
        sp = seasonal_setpoint(
            month, ta_i,
            heating_months=heating_months,
            winter_setpoint=winter_setpoint,
            shoulder_setpoint=shoulder_setpoint,
            extreme_cold=extreme_cold,
        )

        base_eq = ta_i + R * (solar_kw + Qint)
        tin_free = base_eq + (tin - base_eq) * alpha
        q_heat = 0.0

        if sp is not None and tin_free < sp:
            denom = max((1.0 - alpha) * R, 1e-9)
            q_heat = max((sp - alpha * tin - (1.0 - alpha) * base_eq) / denom, 0.0)
            base_eq_h = ta_i + R * (solar_kw + Qint + q_heat)
            tin_next = base_eq_h + (tin - base_eq_h) * alpha
        else:
            tin_next = tin_free

        rows.append({
            "datetime":       index[i],
            "ta":             ta_i,
            "indoor_temp_C":  float(tin_next),
            "heat_kw":        float(q_heat),
        })
        tin = tin_next

    profile = pd.DataFrame(rows)
    profile["heat_kwh"] = profile["heat_kw"] * dt_hours
    profile["heat_kwh_per_m2"] = profile["heat_kwh"] / area
    return profile


def summarize_profile(
    profile: pd.DataFrame,
    area: float,
    heating_months: List[int] = HEATING_MONTHS,
) -> Dict[str, float]:
    """
    Compute annual energy, summer share, summer timestep share, and peak from
    a simulate_heating_profile output.
    """
    area = max(float(area), 1e-6)
    dt_seconds = DT_SECONDS
    if len(profile) >= 2:
        dt_series = pd.to_datetime(profile["datetime"]).diff().dt.total_seconds().dropna()
        if len(dt_series) > 0:
            dt_seconds = float(dt_series.median())
    dt_hours = dt_seconds / 3600.0

    annual_kwh = float(profile["heat_kw"].sum()) * dt_hours
    annual_kwh_per_m2 = annual_kwh / area

    months = pd.to_datetime(profile["datetime"]).dt.month
    summer_mask = ~months.isin(heating_months)
    summer_kwh = float(profile.loc[summer_mask, "heat_kw"].sum()) * dt_hours
    summer_kwh_share = summer_kwh / max(annual_kwh, 1e-9)
    summer_timestep_share = float(
        (profile.loc[summer_mask, "heat_kw"] > 1e-9).sum() / max(len(profile), 1)
    )
    peak_kw = float(profile["heat_kw"].max()) if len(profile) else 0.0

    return {
        "annual_kwh": annual_kwh,
        "annual_kwh_per_m2": annual_kwh_per_m2,
        "summer_kwh": summer_kwh,
        "summer_kwh_share": summer_kwh_share,
        "summer_timestep_share": summer_timestep_share,
        "peak_kw": peak_kw,
    }


def simulate_standard_year_summary(
    R: float,
    C: float,
    A: float,
    Qint: float,
    area: float,
    weather_by_year: Dict[int, pd.DataFrame],
    years: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Simulate across all standard years and return mean summary statistics.
    """
    target_years = years if years is not None else list(weather_by_year.keys())
    summaries = []
    for year in target_years:
        if year not in weather_by_year:
            continue
        profile = simulate_heating_profile(R=R, C=C, A=A, Qint=Qint, area=area, weather_df=weather_by_year[year])
        summaries.append(summarize_profile(profile, area=area))

    if not summaries:
        return {
            "annual_kwh_per_m2_mean": np.nan,
            "annual_kwh_mean": np.nan,
            "summer_kwh_share_mean": np.nan,
            "summer_timestep_share_mean": np.nan,
            "peak_kw_mean": np.nan,
        }

    return {
        "annual_kwh_per_m2_mean": float(np.mean([s["annual_kwh_per_m2"] for s in summaries])),
        "annual_kwh_mean":        float(np.mean([s["annual_kwh"] for s in summaries])),
        "summer_kwh_share_mean":  float(np.mean([s["summer_kwh_share"] for s in summaries])),
        "summer_timestep_share_mean": float(np.mean([s["summer_timestep_share"] for s in summaries])),
        "peak_kw_mean":           float(np.mean([s["peak_kw"] for s in summaries])),
    }


def prepare_weather_arrays(
    weather_df: pd.DataFrame,
    heating_months: List[int] = HEATING_MONTHS,
    winter_setpoint: float = WINTER_SETPOINT,
    shoulder_setpoint: float = SHOULDER_SETPOINT,
    extreme_cold: float = EXTREME_COLD_OUTDOOR_C,
) -> Dict[str, Any]:
    """Prepare sorted weather and thermostat arrays for summary-only RC simulation."""
    wdf = weather_df.copy()
    if "datetime" in wdf.columns:
        wdf = wdf.set_index("datetime")
    wdf.index = pd.to_datetime(wdf.index)
    wdf = wdf.sort_index()

    index = wdf.index
    ta = wdf["ta"].to_numpy(dtype=float)
    qg = wdf["qg"].to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=int)
    if len(index) >= 2:
        dt_seconds = float((index[1] - index[0]).total_seconds())
        summary_dt_seconds = float(pd.Series(index).diff().dt.total_seconds().dropna().median())
    else:
        dt_seconds = float(DT_SECONDS)
        summary_dt_seconds = float(DT_SECONDS)
    setpoint = np.array(
        [
            np.nan
            if (value := seasonal_setpoint(
                int(month_i),
                float(ta_i),
                heating_months=heating_months,
                winter_setpoint=winter_setpoint,
                shoulder_setpoint=shoulder_setpoint,
                extreme_cold=extreme_cold,
            )) is None
            else float(value)
            for month_i, ta_i in zip(month, ta)
        ],
        dtype=float,
    )
    return {
        "datetime": index,
        "ta": ta,
        "qg": qg,
        "month": month,
        "setpoint": setpoint,
        "summer_mask": ~np.isin(month, np.asarray(heating_months, dtype=int)),
        "dt_seconds": dt_seconds,
        "dt_hours": dt_seconds / 3600.0,
        # summarize_profile uses the median interval when accumulating kWh.
        "summary_dt_seconds": summary_dt_seconds,
        "summary_dt_hours": summary_dt_seconds / 3600.0,
    }


def simulate_community_summary_vectorized(
    selected_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    initial_temp: float = 20.0,
    heating_months: List[int] = HEATING_MONTHS,
    winter_setpoint: float = WINTER_SETPOINT,
    shoulder_setpoint: float = SHOULDER_SETPOINT,
    extreme_cold: float = EXTREME_COLD_OUTDOOR_C,
    return_profiles: bool = False,
):
    """
    Simulate full RC state recursion for multiple buildings without profile DataFrames.

    Summary-only avoids per-timestep DataFrame construction; it does not skip
    indoor-temperature updates or alter thermostat behavior.
    """
    required = {"R", "C", "A", "Qint", "Area"}
    missing = required - set(selected_df.columns)
    if missing:
        raise ValueError(f"selected_df missing required columns: {sorted(missing)}")

    inputs = selected_df.reset_index(drop=True).copy()
    weather = prepare_weather_arrays(
        weather_df,
        heating_months=heating_months,
        winter_setpoint=winter_setpoint,
        shoulder_setpoint=shoulder_setpoint,
        extreme_cold=extreme_cold,
    )
    n_buildings = len(inputs)
    n_steps = len(weather["datetime"])

    R = np.maximum(inputs["R"].to_numpy(dtype=float), 1e-6)
    C = np.maximum(inputs["C"].to_numpy(dtype=float), 1e-6)
    A = inputs["A"].to_numpy(dtype=float)
    Qint = inputs["Qint"].to_numpy(dtype=float)
    area = np.maximum(inputs["Area"].to_numpy(dtype=float), 1e-6)
    alpha = np.exp(-float(weather["dt_seconds"]) / np.maximum(R * C, 1e-6))
    tin = np.full(n_buildings, float(initial_temp), dtype=float)
    annual_heat_kw_steps = np.zeros(n_buildings, dtype=float)
    summer_heat_kw_steps = np.zeros(n_buildings, dtype=float)
    summer_heating_steps = np.zeros(n_buildings, dtype=float)
    peak_heat_kw = np.zeros(n_buildings, dtype=float)
    min_indoor_temp = np.full(n_buildings, float(initial_temp), dtype=float)
    comfort_violation_degree_hours = np.zeros(n_buildings, dtype=float)
    comfort_violation_steps = np.zeros(n_buildings, dtype=float)
    community_peak_kw = 0.0
    heat_kw_matrix = np.zeros((n_steps, n_buildings), dtype=float) if return_profiles else None
    indoor_temp_matrix = np.zeros((n_steps, n_buildings), dtype=float) if return_profiles else None

    for step in range(n_steps):
        ta_i = float(weather["ta"][step])
        qg_i = float(weather["qg"][step])
        setpoint_i = float(weather["setpoint"][step])
        solar_kw = A * qg_i / 1000.0
        base_eq = ta_i + R * (solar_kw + Qint)
        tin_free = base_eq + (tin - base_eq) * alpha
        q_heat = np.zeros(n_buildings, dtype=float)
        tin_next = tin_free.copy()

        if np.isfinite(setpoint_i):
            heating_mask = tin_free < setpoint_i
            denom = np.maximum((1.0 - alpha) * R, 1e-9)
            q_heat[heating_mask] = np.maximum(
                (
                    setpoint_i
                    - alpha[heating_mask] * tin[heating_mask]
                    - (1.0 - alpha[heating_mask]) * base_eq[heating_mask]
                )
                / denom[heating_mask],
                0.0,
            )
            base_eq_h = ta_i + R * (solar_kw + Qint + q_heat)
            tin_next[heating_mask] = (
                base_eq_h[heating_mask]
                + (tin[heating_mask] - base_eq_h[heating_mask]) * alpha[heating_mask]
            )
            comfort_delta = np.maximum(setpoint_i - tin_next, 0.0)
            comfort_violation_degree_hours += comfort_delta * float(weather["summary_dt_hours"])
            comfort_violation_steps += (comfort_delta > 1e-9).astype(float)

        annual_heat_kw_steps += q_heat
        if bool(weather["summer_mask"][step]):
            summer_heat_kw_steps += q_heat
            summer_heating_steps += (q_heat > 1e-9).astype(float)
        peak_heat_kw = np.maximum(peak_heat_kw, q_heat)
        community_peak_kw = max(community_peak_kw, float(q_heat.sum()))
        min_indoor_temp = np.minimum(min_indoor_temp, tin_next)
        if return_profiles:
            heat_kw_matrix[step, :] = q_heat
            indoor_temp_matrix[step, :] = tin_next
        tin = tin_next

    summary_dt_hours = float(weather["summary_dt_hours"])
    annual_heat_kwh = annual_heat_kw_steps * summary_dt_hours
    summer_heat_kwh = summer_heat_kw_steps * summary_dt_hours
    building_summary_df = inputs.copy()
    building_summary_df["annual_heat_kwh"] = annual_heat_kwh
    building_summary_df["annual_heat_kwh_per_m2"] = annual_heat_kwh / area
    building_summary_df["summer_heat_kwh"] = summer_heat_kwh
    building_summary_df["summer_heat_share"] = summer_heat_kwh / np.maximum(annual_heat_kwh, 1e-9)
    building_summary_df["summer_timestep_share"] = summer_heating_steps / max(n_steps, 1)
    building_summary_df["peak_heat_kw"] = peak_heat_kw
    building_summary_df["min_indoor_temp_C"] = min_indoor_temp
    building_summary_df["comfort_violation_degree_hours"] = comfort_violation_degree_hours
    building_summary_df["comfort_violation_timestep_share"] = comfort_violation_steps / max(n_steps, 1)

    total_annual_kwh = float(annual_heat_kwh.sum())
    total_summer_kwh = float(summer_heat_kwh.sum())
    community_summary = {
        "n_buildings": int(n_buildings),
        "n_timesteps": int(n_steps),
        "annual_heat_kwh": total_annual_kwh,
        "annual_heat_kwh_per_m2": total_annual_kwh / max(float(area.sum()), 1e-9),
        "summer_heat_kwh": total_summer_kwh,
        "summer_heat_share": total_summer_kwh / max(total_annual_kwh, 1e-9),
        "peak_heat_kw": community_peak_kw,
        "dt_seconds": float(weather["dt_seconds"]),
    }
    if not return_profiles:
        return building_summary_df, community_summary
    return building_summary_df, community_summary, {
        "datetime": weather["datetime"],
        "heat_kw": heat_kw_matrix,
        "indoor_temp_C": indoor_temp_matrix,
    }


def simulate_community_standard_year_summary_vectorized(
    selected_df: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    years: Optional[List[int]] = None,
    initial_temp: float = 20.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Return per-building mean standard-year summaries using vectorized full RC simulation.

    Each weather year starts from the same initial indoor temperature, and
    annual energy, summer shares, timestep shares, and peak demand are averaged
    across years in the same way as `simulate_standard_year_summary(...)`.
    """
    target_years = years if years is not None else list(weather_by_year.keys())
    summaries = []
    for year in target_years:
        if year not in weather_by_year:
            continue
        building_summary, _ = simulate_community_summary_vectorized(
            selected_df=selected_df,
            weather_df=weather_by_year[year],
            initial_temp=initial_temp,
            return_profiles=False,
        )
        summaries.append(building_summary)

    out = selected_df.reset_index(drop=True).copy()
    if not summaries:
        for column in (
            "annual_kwh_per_m2_mean",
            "annual_kwh_mean",
            "summer_kwh_share_mean",
            "summer_timestep_share_mean",
            "peak_kw_mean",
        ):
            out[column] = np.nan
        return out, {"n_buildings": int(len(out)), "n_weather_years": 0}

    out["annual_kwh_per_m2_mean"] = np.mean(
        [summary["annual_heat_kwh_per_m2"].to_numpy(dtype=float) for summary in summaries],
        axis=0,
    )
    out["annual_kwh_mean"] = np.mean(
        [summary["annual_heat_kwh"].to_numpy(dtype=float) for summary in summaries],
        axis=0,
    )
    out["summer_kwh_share_mean"] = np.mean(
        [summary["summer_heat_share"].to_numpy(dtype=float) for summary in summaries],
        axis=0,
    )
    out["summer_timestep_share_mean"] = np.mean(
        [summary["summer_timestep_share"].to_numpy(dtype=float) for summary in summaries],
        axis=0,
    )
    out["peak_kw_mean"] = np.mean(
        [summary["peak_heat_kw"].to_numpy(dtype=float) for summary in summaries],
        axis=0,
    )
    community_summary = {
        "n_buildings": int(len(out)),
        "n_weather_years": int(len(summaries)),
        "annual_kwh_mean": float(out["annual_kwh_mean"].sum()),
        "annual_kwh_per_m2_mean": float(
            out["annual_kwh_mean"].sum() / max(float(out["Area"].sum()), 1e-9)
        ),
        "summer_kwh_share_mean": float(out["summer_kwh_share_mean"].mean()),
        "peak_kw_mean": float(out["peak_kw_mean"].sum()),
    }
    return out, community_summary


def make_screen_weather(
    weather_by_year: Dict[int, pd.DataFrame],
    profile_year: int = PROFILE_YEAR,
    n_months: int = 6,
) -> pd.DataFrame:
    """
    Return a compact screening weather slice (heating-season months only)
    to speed up candidate screening.
    """
    from src.config import HEATING_MONTHS

    df = weather_by_year[profile_year].copy()
    if "datetime" in df.columns:
        months = pd.to_datetime(df["datetime"]).dt.month
    else:
        months = pd.to_datetime(df.index).month
    screen = df.loc[months.isin(HEATING_MONTHS)].copy().reset_index(drop=True)
    return screen


def estimate_std_energy(
    R: float,
    C: float,
    A: float,
    Qint: float,
    area: float,
    weather_by_year: Dict[int, pd.DataFrame],
    profile_year: int = PROFILE_YEAR,
) -> float:
    """Fast energy estimate via screen weather (heating months only, scaled to full year)."""
    screen_weather = make_screen_weather(weather_by_year=weather_by_year, profile_year=profile_year)
    profile = simulate_heating_profile(R=R, C=C, A=A, Qint=Qint, area=area, weather_df=screen_weather)
    summary = summarize_profile(profile=profile, area=area)
    full_year_len = len(weather_by_year[profile_year])
    screen_len = max(len(screen_weather), 1)
    annual_scale = full_year_len / screen_len
    return float(summary["annual_kwh_per_m2"] * annual_scale)


# ---------------------------------------------------------------------------
# Optional typical-weather proxy simulation
# ---------------------------------------------------------------------------

def _typical_weather_months(mode: str, selected_months: Optional[List[int]]) -> List[int]:
    if selected_months is not None:
        return [int(month) for month in selected_months]
    if mode == "heating_season":
        return list(HEATING_MONTHS)
    if mode == "full_year_typical_days":
        return list(range(1, 13))
    raise ValueError("mode must be either 'heating_season' or 'full_year_typical_days'.")


def _daily_weather_inputs(
    weather_by_year: Dict[int, pd.DataFrame],
    selected_months: List[int],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    feature_rows: List[Dict[str, float]] = []
    daily_weather: Dict[str, pd.DataFrame] = {}
    for year, weather_df in sorted(weather_by_year.items()):
        weather = weather_df.copy()
        if "datetime" in weather.columns:
            weather["datetime"] = pd.to_datetime(weather["datetime"])
        else:
            weather = weather.reset_index().rename(columns={weather.index.name or "index": "datetime"})
            weather["datetime"] = pd.to_datetime(weather["datetime"])
        weather = weather.loc[weather["datetime"].dt.month.isin(selected_months)].copy()
        weather["_weather_day"] = weather["datetime"].dt.normalize()
        for day, part in weather.groupby("_weather_day", sort=True):
            part = part.drop(columns="_weather_day").sort_values("datetime").reset_index(drop=True)
            if len(part) == 0:
                continue
            key = f"{int(year)}_{pd.Timestamp(day):%Y%m%d}"
            ta = pd.to_numeric(part["ta"], errors="coerce")
            qg = pd.to_numeric(part["qg"], errors="coerce")
            if not ta.notna().any() or not qg.notna().any():
                continue
            daily_weather[key] = part
            mean_ta = float(ta.mean())
            feature_rows.append({
                "day_key": key,
                "source_year": int(year),
                "date": pd.Timestamp(day),
                "month": int(pd.Timestamp(day).month),
                "mean_ta": mean_ta,
                "min_ta": float(ta.min()),
                "max_ta": float(ta.max()),
                "solar_sum": float(qg.sum()),
                "heating_degree_day": float(max(WINTER_SETPOINT - mean_ta, 0.0)),
            })
    return pd.DataFrame(feature_rows), daily_weather


def _representative_clusters(
    daily_features: pd.DataFrame,
    n_clusters: int,
    include_extreme_days: bool,
    n_extreme_days: int,
) -> List[Dict[str, Any]]:
    feature_cols = ["mean_ta", "min_ta", "max_ta", "solar_sum", "heating_degree_day"]
    n_days = len(daily_features)
    if n_days == 0:
        return []
    target_clusters = max(1, min(int(n_clusters), n_days))
    extreme_count = min(max(int(n_extreme_days), 0), target_clusters) if include_extreme_days else 0
    extremes = (
        daily_features.nsmallest(extreme_count, "min_ta").index.tolist()
        if extreme_count else []
    )
    remaining = daily_features.drop(index=extremes)
    remaining_clusters = min(target_clusters - len(extremes), len(remaining))
    groups: List[Tuple[pd.Index, bool]] = []
    for idx in extremes:
        groups.append((pd.Index([idx]), True))

    if remaining_clusters > 0:
        values = remaining[feature_cols].to_numpy(dtype=float)
        scale = np.nanstd(values, axis=0)
        scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
        normalized = (values - np.nanmean(values, axis=0)) / scale
        centers = [0]
        while len(centers) < remaining_clusters:
            dist = np.min(
                np.stack([np.sum((normalized - normalized[pos]) ** 2, axis=1) for pos in centers]),
                axis=0,
            )
            dist[centers] = -1.0
            centers.append(int(np.argmax(dist)))
        center_values = normalized[centers].copy()
        assignments = np.zeros(len(remaining), dtype=int)
        for _ in range(20):
            distances = np.stack(
                [np.sum((normalized - center) ** 2, axis=1) for center in center_values],
                axis=1,
            )
            new_assignments = np.argmin(distances, axis=1)
            if np.array_equal(assignments, new_assignments) and _ > 0:
                break
            assignments = new_assignments
            for cluster_id in range(remaining_clusters):
                members = normalized[assignments == cluster_id]
                if len(members):
                    center_values[cluster_id] = members.mean(axis=0)
        for cluster_id in range(remaining_clusters):
            member_positions = np.flatnonzero(assignments == cluster_id)
            if len(member_positions):
                groups.append((remaining.index[member_positions], False))

    clusters: List[Dict[str, Any]] = []
    for cluster_id, (member_index, is_extreme) in enumerate(groups, start=1):
        members = daily_features.loc[member_index]
        values = members[feature_cols].to_numpy(dtype=float)
        scale = np.nanstd(daily_features[feature_cols].to_numpy(dtype=float), axis=0)
        scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
        center = values.mean(axis=0)
        distances = np.sum(((values - center) / scale) ** 2, axis=1)
        representative = members.iloc[int(np.argmin(distances))]
        clusters.append({
            "scenario_id": f"scenario_{cluster_id:02d}",
            "day_key": str(representative["day_key"]),
            "source_year": int(representative["source_year"]),
            "representative_date": pd.Timestamp(representative["date"]),
            "month": int(representative["month"]),
            "mean_ta": float(representative["mean_ta"]),
            "min_ta": float(representative["min_ta"]),
            "max_ta": float(representative["max_ta"]),
            "solar_sum": float(representative["solar_sum"]),
            "heating_degree_day": float(representative["heating_degree_day"]),
            "n_days": int(len(members)),
            "weight": float(len(members) / n_days),
            "is_extreme": bool(is_extreme),
        })
    return clusters


def build_typical_weather_scenarios(
    weather_by_year: Dict[int, pd.DataFrame],
    mode: str = "heating_season",
    n_clusters: int = 12,
    selected_months: Optional[List[int]] = None,
    include_extreme_days: bool = True,
    n_extreme_days: int = 1,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build an optional representative-day weather approximation for proxy use.

    Representative scenarios are real observed days. Full standard-year
    simulation remains the benchmark path for shortlisted candidates and
    validation.

    `random_state` is currently metadata-only because representative-day
    clustering is deterministic. It is reserved for future stochastic
    clustering or backend implementations and keeps API/config/cache metadata
    consistent.
    """
    months = _typical_weather_months(mode, selected_months)
    daily_features, daily_weather = _daily_weather_inputs(weather_by_year, months)
    if len(daily_features) == 0:
        raise ValueError("No daily weather data were available for typical-weather scenarios.")
    clusters = _representative_clusters(
        daily_features=daily_features,
        n_clusters=n_clusters,
        include_extreme_days=include_extreme_days,
        n_extreme_days=n_extreme_days,
    )
    scenario_table = pd.DataFrame(clusters)
    representative_days = {
        row.scenario_id: daily_weather[str(row.day_key)].copy()
        for row in scenario_table.itertuples(index=False)
    }
    source_years = sorted(int(year) for year in weather_by_year.keys())
    resolution_seconds = None
    sample = next(iter(representative_days.values()))
    if len(sample) >= 2:
        resolution_seconds = float(
            (pd.to_datetime(sample["datetime"]).iloc[1] - pd.to_datetime(sample["datetime"]).iloc[0]).total_seconds()
        )
    return {
        "scenario_table": scenario_table,
        "representative_days": representative_days,
        "metadata": {
            "mode": mode,
            "selected_months": months,
            "n_clusters": int(n_clusters),
            "include_extreme_days": bool(include_extreme_days),
            "n_extreme_days": int(n_extreme_days),
            "random_state": random_state,
            "source_years": source_years,
            "source_year_count": len(source_years),
            "n_source_days": int(len(daily_features)),
            "resolution_seconds": resolution_seconds,
            "feature_version": "daily_weather_v1",
            "feature_columns": ["mean_ta", "min_ta", "max_ta", "solar_sum", "heating_degree_day"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def _typical_weather_cache_file(
    weather_by_year: Dict[int, pd.DataFrame],
    cache_path: Optional[str | Path],
    mode: str,
    n_clusters: int,
    selected_months: Optional[List[int]],
    include_extreme_days: bool,
    n_extreme_days: int,
    random_state: Optional[int],
) -> Path:
    if cache_path is not None:
        return Path(cache_path)
    months = _typical_weather_months(mode, selected_months)
    resolution_seconds = {}
    for year, weather_df in weather_by_year.items():
        if len(weather_df) >= 2:
            timestamps = (
                pd.to_datetime(weather_df["datetime"])
                if "datetime" in weather_df.columns
                else pd.to_datetime(weather_df.index)
            )
            resolution_seconds[str(year)] = float((timestamps[1] - timestamps[0]).total_seconds())
    signature = {
        "source_years": sorted(int(year) for year in weather_by_year.keys()),
        "source_lengths": {str(year): int(len(df)) for year, df in weather_by_year.items()},
        "resolution_seconds": resolution_seconds,
        "mode": mode,
        "selected_months": months,
        "n_clusters": int(n_clusters),
        "include_extreme_days": bool(include_extreme_days),
        "n_extreme_days": int(n_extreme_days),
        "random_state": random_state,
        "feature_version": "daily_weather_v1",
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path("cache") / "typical_weather" / f"typical_weather_{digest}.pkl"


def load_or_build_typical_weather_scenarios(
    weather_by_year: Dict[int, pd.DataFrame],
    mode: str = "heating_season",
    n_clusters: int = 12,
    selected_months: Optional[List[int]] = None,
    include_extreme_days: bool = True,
    n_extreme_days: int = 1,
    random_state: Optional[int] = None,
    cache_path: Optional[str | Path] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Load or cache representative-day weather scenarios for optional proxy acceleration.

    `random_state` is currently metadata-only because representative-day
    clustering is deterministic; it is included for API/config/cache
    consistency and future stochastic backends.
    """
    target_path = _typical_weather_cache_file(
        weather_by_year=weather_by_year,
        cache_path=cache_path,
        mode=mode,
        n_clusters=n_clusters,
        selected_months=selected_months,
        include_extreme_days=include_extreme_days,
        n_extreme_days=n_extreme_days,
        random_state=random_state,
    )
    if use_cache and target_path.exists():
        with target_path.open("rb") as cache_file:
            payload = pickle.load(cache_file)
        if isinstance(payload, dict) and "scenario_table" in payload and "representative_days" in payload:
            return payload
    payload = build_typical_weather_scenarios(
        weather_by_year=weather_by_year,
        mode=mode,
        n_clusters=n_clusters,
        selected_months=selected_months,
        include_extreme_days=include_extreme_days,
        n_extreme_days=n_extreme_days,
        random_state=random_state,
    )
    if use_cache:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload["metadata"]["cache_path"] = str(target_path)
        with target_path.open("wb") as cache_file:
            pickle.dump(payload, cache_file)
    return payload


def simulate_typical_weather_summary(
    R: float,
    C: float,
    A: float,
    Qint: float,
    area: float,
    typical_weather_scenarios: Dict[str, Any],
    warmup_repeats: int = 2,
) -> Dict[str, Any]:
    """
    Approximate annual heating metrics from weighted real representative days.

    Each day is simulated repeatedly before collecting its final cycle because
    the indoor-temperature state needs warm-up. This is for fast proxy
    estimation, not final validation claims.
    """
    scenario_table = typical_weather_scenarios.get("scenario_table")
    representative_days = typical_weather_scenarios.get("representative_days", {})
    metadata = typical_weather_scenarios.get("metadata", {})
    if not isinstance(scenario_table, pd.DataFrame) or len(scenario_table) == 0:
        raise ValueError("typical_weather_scenarios must contain a non-empty scenario_table.")
    source_year_count = max(int(metadata.get("source_year_count", 1)), 1)
    annual_kwh = 0.0
    peak_values: List[float] = []
    for scenario in scenario_table.itertuples(index=False):
        weather_day = representative_days.get(scenario.scenario_id)
        if not isinstance(weather_day, pd.DataFrame) or len(weather_day) == 0:
            raise ValueError(f"Missing representative weather data for {scenario.scenario_id}.")
        initial_temp = 20.0
        profile = pd.DataFrame()
        for _ in range(max(int(warmup_repeats), 0) + 1):
            profile = simulate_heating_profile(
                R=R, C=C, A=A, Qint=Qint, area=area,
                weather_df=weather_day, initial_temp=initial_temp,
            )
            initial_temp = float(profile["indoor_temp_C"].iloc[-1])
        day_kwh = float(profile["heat_kwh"].sum())
        annual_kwh += day_kwh * float(scenario.n_days) / source_year_count
        peak_values.append(float(profile["heat_kw"].max()))
    annual_per_m2 = annual_kwh / max(float(area), 1e-6)
    peak_fast = float(max(peak_values)) if peak_values else np.nan
    return {
        "annual_kwh_per_m2_mean": float(annual_per_m2),
        "annual_kwh_mean": float(annual_kwh),
        "summer_kwh_share_mean": np.nan,
        "summer_timestep_share_mean": np.nan,
        "peak_kw_mean": peak_fast,
        "peak_kw_fast": peak_fast,
        "typical_weather_mode": metadata.get("mode", "heating_season"),
        "typical_weather_n_scenarios": int(len(scenario_table)),
        "typical_weather_warmup_repeats": int(warmup_repeats),
    }


def estimate_std_energy_typical(
    R: float,
    C: float,
    A: float,
    Qint: float,
    area: float,
    typical_weather_scenarios: Dict[str, Any],
    warmup_repeats: int = 2,
) -> float:
    """Return approximate annual kWh/m2 from optional typical-weather proxy simulation."""
    summary = simulate_typical_weather_summary(
        R=R, C=C, A=A, Qint=Qint, area=area,
        typical_weather_scenarios=typical_weather_scenarios,
        warmup_repeats=warmup_repeats,
    )
    return float(summary["annual_kwh_per_m2_mean"])


# ---------------------------------------------------------------------------
# Custom thermostat simulation (Cell 32 / validation use)
# ---------------------------------------------------------------------------

def ensure_datetime_index(
    df: pd.DataFrame,
    datetime_cols: Tuple[str, ...] = ("datetime", "time", "timestamp", "date"),
) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out.sort_index()
    for col in datetime_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
            return out.set_index(col).sort_index()
    raise ValueError(
        f"weather_df must have a DatetimeIndex or one of these datetime columns: {datetime_cols}"
    )


def infer_weather_columns(
    weather_df: pd.DataFrame,
    temp_candidates: Tuple[str, ...] = ("ta", "T_out", "Tout", "outdoor_temp", "temperature", "temp_air"),
    solar_candidates: Tuple[str, ...] = ("qg", "ghi", "GHI", "solar", "solar_radiation", "irradiance", "radiation"),
) -> Tuple[str, str]:
    temp_col = next((c for c in temp_candidates if c in weather_df.columns), None)
    solar_col = next((c for c in solar_candidates if c in weather_df.columns), None)
    if temp_col is None or solar_col is None:
        raise ValueError(
            f"Could not infer weather columns. Available: {list(weather_df.columns)}"
        )
    return temp_col, solar_col


def infer_dt_seconds(index: pd.DatetimeIndex, default: int = DT_SECONDS) -> float:
    dt = index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt) or dt <= 0:
        return float(default)
    return float(dt)


def build_custom_setpoint(
    index: pd.DatetimeIndex,
    *,
    summer_months: Tuple[int, ...] = CUSTOM_SUMMER_MONTHS,
    day_start: int = CUSTOM_DAY_START_HOUR,
    day_end: int = CUSTOM_DAY_END_HOUR,
    day_setpoint: float = CUSTOM_DAY_SETPOINT_C,
    other_setpoint: float = CUSTOM_OTHER_SETPOINT_C,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (setpoint_array, is_summer_array); summer timesteps get NaN → heating off."""
    hours = np.asarray(index.hour)
    is_summer = np.isin(np.asarray(index.month), list(summer_months))
    setpoint = np.full(len(index), np.nan, dtype=float)
    is_day = (hours >= day_start) & (hours < day_end)
    setpoint[~is_summer & is_day] = day_setpoint
    setpoint[~is_summer & ~is_day] = other_setpoint
    return setpoint, is_summer


def simulate_heating_profile_custom(
    building: Dict[str, Any],
    weather_df: pd.DataFrame,
    *,
    summer_months: Tuple[int, ...] = CUSTOM_SUMMER_MONTHS,
    initial_temp: float = CUSTOM_INITIAL_TEMP_C,
    day_start: int = CUSTOM_DAY_START_HOUR,
    day_end: int = CUSTOM_DAY_END_HOUR,
    day_setpoint: float = CUSTOM_DAY_SETPOINT_C,
    other_setpoint: float = CUSTOM_OTHER_SETPOINT_C,
) -> pd.DataFrame:
    """
    Simulate using custom (occupied-hours) thermostat schedule.
    Requires building dict with R, C, A, Qint, (optionally) Area.
    """
    weather = ensure_datetime_index(weather_df)
    temp_col, solar_col = infer_weather_columns(weather)

    ta = weather[temp_col].to_numpy(dtype=float)
    solar_raw = weather[solar_col].to_numpy(dtype=float)
    solar_kw_per_m2 = solar_raw / 1000.0

    index = weather.index
    dt_sec = infer_dt_seconds(index)
    dt_hours = dt_sec / 3600.0

    setpoint, is_summer = build_custom_setpoint(
        index,
        summer_months=summer_months,
        day_start=day_start,
        day_end=day_end,
        day_setpoint=day_setpoint,
        other_setpoint=other_setpoint,
    )

    R = max(float(building["R"]), 1e-6)
    C = max(float(building["C"]), 1e-6)
    A = float(building["A"])
    Qint = float(building["Qint"])
    area = max(float(building.get("Area", np.nan) or np.nan), 1e-6)

    tau_sec = max(R * C, 1e-6)
    alpha = float(np.exp(-dt_sec / tau_sec))
    tin = float(initial_temp)

    rows = []
    for i, dt in enumerate(index):
        ta_i = float(ta[i])
        sp = float(setpoint[i]) if np.isfinite(setpoint[i]) else np.nan
        solar_gain_kw = float(A * solar_kw_per_m2[i])

        base_eq = ta_i + R * (solar_gain_kw + Qint)
        tin_free = base_eq + (tin - base_eq) * alpha
        q_heat = 0.0

        if np.isfinite(sp) and tin_free < sp:
            denom = max((1.0 - alpha) * R, 1e-9)
            q_heat = max((sp - alpha * tin - (1.0 - alpha) * base_eq) / denom, 0.0)
            base_eq_h = ta_i + R * (solar_gain_kw + Qint + q_heat)
            tin_next = base_eq_h + (tin - base_eq_h) * alpha
        else:
            tin_next = tin_free

        rows.append({
            "datetime":        dt,
            "ta":              ta_i,
            "solar_raw":       float(solar_raw[i]),
            "solar_gain_kw":   solar_gain_kw,
            "setpoint_C":      sp,
            "is_summer":       bool(is_summer[i]),
            "free_float_temp_C": float(tin_free),
            "indoor_temp_C":   float(tin_next),
            "heat_kw":         float(q_heat),
            "heating_on":      bool(q_heat > 1e-9),
            "month":           int(dt.month),
        })
        tin = tin_next

    profile = pd.DataFrame(rows).set_index("datetime")
    profile["heat_kwh"] = profile["heat_kw"] * dt_hours
    profile["heat_kwh_per_m2"] = profile["heat_kwh"] / area
    return profile


def simulate_community_heating_custom(
    selected_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    *,
    case_name: str = "case",
    summer_months: Tuple[int, ...] = CUSTOM_SUMMER_MONTHS,
    initial_temp: float = CUSTOM_INITIAL_TEMP_C,
    day_start: int = CUSTOM_DAY_START_HOUR,
    day_end: int = CUSTOM_DAY_END_HOUR,
    day_setpoint: float = CUSTOM_DAY_SETPOINT_C,
    other_setpoint: float = CUSTOM_OTHER_SETPOINT_C,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Simulate all buildings in a generated community.

    Returns (community_summary_dict, building_summary_df, community_profile_df).
    """
    building_rows = []
    profile_list = []

    for i, (_, row) in enumerate(selected_df.reset_index(drop=True).iterrows(), start=1):
        building = row.to_dict()
        building_id = building.get("building_id", f"B{i:03d}")
        profile = simulate_heating_profile_custom(
            building, weather_df,
            summer_months=summer_months, initial_temp=initial_temp,
            day_start=day_start, day_end=day_end,
            day_setpoint=day_setpoint, other_setpoint=other_setpoint,
        )

        area = float(building.get("Area") or np.nan)
        annual_kwh = float(profile["heat_kwh"].sum())
        summer_kwh = float(profile.loc[profile["is_summer"], "heat_kwh"].sum())
        annual_kwh_per_m2 = annual_kwh / max(area, 1e-6) if np.isfinite(area) else np.nan
        summer_share_pct = 100.0 * summer_kwh / max(annual_kwh, 1e-9)

        building_rows.append({
            "case_name":               case_name,
            "building_id":             building_id,
            "Area":                    area,
            "EnergyLabel":             building.get("EnergyLabel"),
            "vintage":                 building.get("vintage"),
            "R":                       float(building["R"]),
            "C":                       float(building["C"]),
            "A":                       float(building["A"]),
            "Qint":                    float(building["Qint"]),
            "annual_heat_kwh":         annual_kwh,
            "annual_heat_mwh":         annual_kwh / 1000.0,
            "annual_heat_kwh_per_m2":  annual_kwh_per_m2,
            "summer_heat_kwh":         summer_kwh,
            "summer_heat_share_pct":   summer_share_pct,
            "peak_heat_kw":            float(profile["heat_kw"].max()),
        })

        profile_col = profile[["heat_kw", "heat_kwh"]].copy()
        profile_col.columns = [f"{building_id}_heat_kw", f"{building_id}_heat_kwh"]
        profile_list.append(profile_col)

    building_df = pd.DataFrame(building_rows)
    if profile_list:
        community_profile = pd.concat(profile_list, axis=1)
        community_profile["total_heat_kw"] = community_profile[[c for c in community_profile.columns if c.endswith("_heat_kw")]].sum(axis=1)
        community_profile["total_heat_kwh"] = community_profile[[c for c in community_profile.columns if c.endswith("_heat_kwh")]].sum(axis=1)
    else:
        community_profile = pd.DataFrame()

    community_summary = {
        "case_name":               case_name,
        "n_buildings":             len(building_df),
        "total_annual_kwh":        float(building_df["annual_heat_kwh"].sum()),
        "total_annual_mwh":        float(building_df["annual_heat_mwh"].sum()),
        "mean_annual_kwh_per_m2":  float(building_df["annual_heat_kwh_per_m2"].mean()),
        "peak_community_kw":       float(community_profile["total_heat_kw"].max()) if len(community_profile) else np.nan,
    }
    return community_summary, building_df, community_profile
