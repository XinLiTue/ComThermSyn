"""
Feature engineering: energy label / vintage helpers, forward-metric computation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    HEATING_MONTHS,
    LABEL_BOUNDS,
    LABEL_ORDER,
    LABEL_SCORE,
    STANDARD_YEARS,
    VINTAGE_BINS,
    VINTAGE_EXPECTED_LABEL_SCORE,
    VINTAGE_LABELS,
)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def clean_energy_label(label: Any) -> Optional[str]:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    text = re.sub(r"\s+", "", str(label).strip().upper())
    if not text or text in ("NAN", "NONE", "NA", ""):
        return None
    return text


def label_bounds(label: Any) -> Tuple[float, float]:
    clean = clean_energy_label(label)
    if clean is not None and clean in LABEL_BOUNDS:
        return LABEL_BOUNDS[clean]
    return (0.0, 1e9)


def bounded_label_bounds(label: Any, *, margin: float = 0.15) -> Tuple[float, float]:
    lo, hi = label_bounds(label)
    span = max(hi - lo, 1.0)
    return max(lo - margin * span, 0.0), hi + margin * span


# ---------------------------------------------------------------------------
# Vintage helper
# ---------------------------------------------------------------------------

def vintage_from_year(year: Any) -> str:
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return VINTAGE_LABELS[-1]
    for i in range(len(VINTAGE_BINS) - 1):
        if VINTAGE_BINS[i] < y <= VINTAGE_BINS[i + 1]:
            return VINTAGE_LABELS[i]
    return VINTAGE_LABELS[-1]


# ---------------------------------------------------------------------------
# Forward metrics: run the standard-year simulation to get E, peaks, shares
# ---------------------------------------------------------------------------

def add_simulated_energy_metrics(
    df: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute E_std_annual_per_m2, summer_kwh_share, summer_timestep_share,
    and peak_heat_kw for every building in df using the standard-year simulation.
    """
    from src.simulation import simulate_standard_year_summary

    results = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if verbose and i % 10 == 0:
            print(f"Forward metrics computed for {i}/{total} buildings")
        try:
            summary = simulate_standard_year_summary(
                R=float(row["R"]),
                C=float(row["C"]),
                A=float(row["A"]),
                Qint=float(row["Qint"]),
                area=float(row["Area"]),
                weather_by_year=weather_by_year,
            )
        except Exception:
            summary = {
                "annual_kwh_per_m2_mean": np.nan,
                "annual_kwh_mean": np.nan,
                "summer_kwh_share_mean": np.nan,
                "summer_timestep_share_mean": np.nan,
                "peak_kw_mean": np.nan,
            }
        results.append(summary)

    if verbose:
        print(f"Forward metrics computed for {total}/{total} buildings")

    metrics = pd.DataFrame(results, index=df.index)
    out = df.copy()
    out["E_std_annual_per_m2"] = metrics["annual_kwh_per_m2_mean"]
    out["summer_kwh_share"] = metrics["summer_kwh_share_mean"]
    out["summer_timestep_share"] = metrics.get("summer_timestep_share_mean", np.nan)
    out["peak_heat_kw"] = metrics["peak_kw_mean"]

    if "C_per_area" not in out.columns:
        area = pd.to_numeric(out["Area"], errors="coerce").replace(0, np.nan)
        out["C_per_area"] = pd.to_numeric(out["C"], errors="coerce") / area

    return out


BASELINE_SIMULATION_METRIC_COLUMNS = (
    "E_std_annual_per_m2",
    "summer_kwh_share",
    "summer_timestep_share",
    "peak_heat_kw",
    "C_per_area",
)
BASELINE_SIMULATION_CACHE_SCHEMA_VERSION = "baseline_simulation_metrics_v1"
BASELINE_SIMULATION_FINGERPRINT_COLUMNS = (
    "user_id",
    "R",
    "C",
    "A",
    "Qint",
    "Area",
    "year",
    "EnergyLabel",
)


def _baseline_simulation_cache_metadata(
    df_raw: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
) -> Dict[str, Any]:
    fingerprint_columns = [
        column for column in BASELINE_SIMULATION_FINGERPRINT_COLUMNS if column in df_raw.columns
    ]
    if "C_per_area" in df_raw.columns:
        fingerprint_columns.append("C_per_area")
    fingerprint_frame = df_raw[fingerprint_columns].reset_index(drop=True)
    fingerprint_hash = hashlib.sha256()
    fingerprint_hash.update(json.dumps(fingerprint_columns).encode("utf-8"))
    fingerprint_hash.update(
        pd.util.hash_pandas_object(fingerprint_frame, index=False).values.tobytes()
    )
    metadata: Dict[str, Any] = {
        "schema_version": BASELINE_SIMULATION_CACHE_SCHEMA_VERSION,
        "row_count": int(len(df_raw)),
        "input_fingerprint": fingerprint_hash.hexdigest(),
        "fingerprint_columns": fingerprint_columns,
        "weather_years": sorted(int(year) for year in weather_by_year.keys()),
        "weather_row_counts": {
            str(year): int(len(weather_by_year[year])) for year in sorted(weather_by_year.keys())
        },
        "has_user_id": "user_id" in df_raw.columns,
    }
    if "user_id" in df_raw.columns:
        user_values = sorted(set(df_raw["user_id"].astype(str).tolist()))
        metadata["user_id_set_fingerprint"] = hashlib.sha256(
            json.dumps(user_values).encode("utf-8")
        ).hexdigest()
    return metadata


def _baseline_simulation_cache_mismatch_reason(
    cached_metadata: Dict[str, Any],
    expected_metadata: Dict[str, Any],
) -> Optional[str]:
    fields = (
        "schema_version",
        "row_count",
        "has_user_id",
        "user_id_set_fingerprint",
        "input_fingerprint",
        "weather_years",
        "weather_row_counts",
    )
    for field in fields:
        if cached_metadata.get(field) != expected_metadata.get(field):
            return f"{field} mismatch"
    return None


def add_or_load_baseline_simulation_metrics(
    df_raw: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    cache_path: str | Path,
    use_cache: bool = True,
    save_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add deterministic fitted-building standard-year metrics, loading a valid cache when available.

    This wraps `add_simulated_energy_metrics(...)` without changing its formulas
    or output columns. Cached metrics are reused only when fitted-building input
    fields and weather-shape metadata match the current request.
    """
    target_path = Path(cache_path)
    metadata_path = target_path.with_suffix(".metadata.json")
    expected_metadata = _baseline_simulation_cache_metadata(df_raw, weather_by_year)
    reject_reason: Optional[str] = None

    if use_cache:
        if not target_path.exists() or not metadata_path.exists():
            reject_reason = "cache file or metadata file missing"
        else:
            try:
                cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                reject_reason = _baseline_simulation_cache_mismatch_reason(
                    cached_metadata, expected_metadata
                )
                if reject_reason is None:
                    cached_metrics = pd.read_parquet(target_path)
                    if len(cached_metrics) != len(df_raw):
                        reject_reason = "cached metric row count mismatch"
                    else:
                        missing_metrics = [
                            column
                            for column in BASELINE_SIMULATION_METRIC_COLUMNS
                            if column not in cached_metrics.columns
                        ]
                        if missing_metrics:
                            reject_reason = f"cached metric columns missing: {missing_metrics}"
                        else:
                            out = df_raw.copy()
                            for column in BASELINE_SIMULATION_METRIC_COLUMNS:
                                out[column] = cached_metrics[column].to_numpy()
                            if verbose:
                                print("Baseline simulation metrics: loaded from cache")
                                print(f"Baseline simulation cache path: {target_path}")
                            return out
            except Exception as exc:
                reject_reason = f"cache read failed: {exc}"

    if verbose and reject_reason is not None:
        print(f"Baseline simulation cache rejected: {reject_reason}")
    out = add_simulated_energy_metrics(
        df_raw,
        weather_by_year=weather_by_year,
        verbose=verbose,
    )
    if verbose:
        print("Baseline simulation metrics: recomputed")
    if save_cache:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        out[list(BASELINE_SIMULATION_METRIC_COLUMNS)].reset_index(drop=True).to_parquet(
            target_path, index=False
        )
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if verbose:
            print(f"Baseline simulation cache saved to: {target_path}")
    return out


# ---------------------------------------------------------------------------
# Community input helpers (used by V4 generator)
# ---------------------------------------------------------------------------

def coerce_condition_frame(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    if "area" in out.columns and "Area" not in out.columns:
        out = out.rename(columns={"area": "Area"})
    required = {"year", "Area", "EnergyLabel"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"community/building inputs missing required columns: {sorted(missing)}")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["Area"] = pd.to_numeric(out["Area"], errors="coerce")
    out["EnergyLabel"] = out["EnergyLabel"].apply(clean_energy_label)
    out = out.dropna(subset=["year", "Area"]).copy()
    out = out[out["Area"] > 0].reset_index(drop=True)
    return out


def prepare_building_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add vintage, label scores, reno_signal, C_per_area to a building DataFrame."""
    df = coerce_condition_frame(df_in)
    for col in ["R", "C", "A", "Qint", "E_std_annual_per_m2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "C_per_area" not in df.columns:
        if "C" in df.columns:
            df["C_per_area"] = df["C"] / df["Area"]
        else:
            df["C_per_area"] = np.nan

    df["vintage"] = df["year"].apply(vintage_from_year)
    df["label_clean"] = df["EnergyLabel"].apply(clean_energy_label)
    df["vintage_expected_label_score"] = df["vintage"].map(VINTAGE_EXPECTED_LABEL_SCORE).astype(float)
    df["label_score_raw"] = df["label_clean"].map(LABEL_SCORE)
    df["label_score"] = df["label_score_raw"].fillna(df["vintage_expected_label_score"])
    df["reno_signal"] = df["label_score"] - df["vintage_expected_label_score"]
    return df


add_forward_metrics = add_simulated_energy_metrics
enrich_model_df = prepare_building_features
