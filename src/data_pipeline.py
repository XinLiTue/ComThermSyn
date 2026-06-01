"""
Data loading and preparation: RCA building parameters, weather data, merging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    BUILDING_DATA_PATH,
    HOUSE_INFO_PATH,
    STANDARD_YEARS,
    WEATHER_PATH,
    WEATHER_RESAMPLE_RULE,
)


def load_merged_dataset(
    building_path: Optional[Path] = None,
    house_info_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge identified RC parameters with house meta-data (area, year, energy label).

    Returns a DataFrame with columns including R, C, A, Qint, year, Area, EnergyLabel.
    """
    bp = Path(building_path or BUILDING_DATA_PATH)
    hip = Path(house_info_path or HOUSE_INFO_PATH)

    rca = pd.read_parquet(bp)

    # Rename raw columns to standard names if needed
    col_map = {}
    if "R_C_per_kW" in rca.columns and "R" not in rca.columns:
        col_map["R_C_per_kW"] = "R"
    if "C_kWh_per_C" in rca.columns and "C" not in rca.columns:
        col_map["C_kWh_per_C"] = "C"
    if "A_m2" in rca.columns and "A" not in rca.columns:
        col_map["A_m2"] = "A"
    if col_map:
        rca = rca.rename(columns=col_map)

    house = pd.read_excel(hip)
    if "participant_id" in house.columns and "user_id" not in house.columns:
        house = house.rename(columns={"participant_id": "user_id"})

    merged = rca.merge(house[["user_id", "Area", "year", "EnergyLabel"]], on="user_id", how="inner")

    keep_cols = [c for c in ["user_id", "R", "C", "A", "Qint", "year", "Area", "EnergyLabel"] + list(rca.columns) if c in merged.columns]
    seen = set()
    ordered = []
    for c in keep_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    return merged[ordered].copy()


def load_weather_data(
    weather_path: Optional[Path] = None,
    resample_rule: str = WEATHER_RESAMPLE_RULE,
) -> pd.DataFrame:
    """
    Load weather CSV and resample to a uniform 15-minute grid.

    Adds year_weather and month columns.
    """
    path = Path(weather_path or WEATHER_PATH)
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    numeric_cols = [c for c in df.columns if c not in ("datetime",)]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.resample(resample_rule).interpolate(method="time").reset_index()
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["year_weather"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    return df


def split_weather_by_year(
    weather_df: pd.DataFrame,
    years: Optional[List[int]] = None,
) -> Dict[int, pd.DataFrame]:
    """Split a weather DataFrame by calendar year into a {year: df} dict."""
    target_years = years if years is not None else STANDARD_YEARS
    out: Dict[int, pd.DataFrame] = {}
    year_col = "year_weather" if "year_weather" in weather_df.columns else None
    for year in target_years:
        if year_col is not None:
            part = weather_df.loc[weather_df[year_col] == year].copy().reset_index(drop=True)
        else:
            part = weather_df.loc[pd.to_datetime(weather_df["datetime"]).dt.year == year].copy().reset_index(drop=True)
        out[year] = part
    return out
