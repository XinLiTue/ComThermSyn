"""Representative building-input selection for train/holdout diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd

from src.config import LABEL_SCORE
from src.feature_engineering import clean_energy_label, vintage_from_year


_ALLOWED_SELECTION_MODES = {
    "input_central",
    "physics_central",
    "random",
    "input_edge",
}
_INPUT_COLUMNS = ("year", "Area", "EnergyLabel")
_PHYSICS_COLUMNS = ("R", "C", "A", "Qint", "C_per_area", "E_std_annual_per_m2")
_LOG_SCALE_COLUMNS = {"R", "C", "A", "tau"}


@dataclass(frozen=True)
class BuildingInputSelectionConfig:
    n_buildings: int = 12
    selection_mode: str = "input_central"
    group_cols: tuple[str, ...] = ("vintage",)
    year_col: str = "year"
    area_col: str = "Area"
    label_col: str = "EnergyLabel"
    random_state: int = 42
    avoid_edge_quantile: float = 0.1


@dataclass
class SelectedBuildingInputs:
    source_name: str
    selected_full_df: pd.DataFrame
    synthesis_input_df: pd.DataFrame
    truth_df: pd.DataFrame
    selection_summary_df: pd.DataFrame


def _validate_mode(mode: str) -> None:
    if mode not in _ALLOWED_SELECTION_MODES:
        allowed = ", ".join(sorted(_ALLOWED_SELECTION_MODES))
        raise ValueError(f"selection_mode must be one of: {allowed}.")


def _add_available_group_columns(
    df: pd.DataFrame,
    group_cols: tuple[str, ...],
    year_col: str,
) -> pd.DataFrame:
    out = df.copy()
    if "vintage" in group_cols and "vintage" not in out.columns and year_col in out.columns:
        out["vintage"] = out[year_col].map(vintage_from_year)
    return out


def _label_score_series(df: pd.DataFrame, label_col: str) -> Optional[pd.Series]:
    if "label_score" in df.columns:
        return pd.to_numeric(df["label_score"], errors="coerce")
    if label_col not in df.columns:
        return None
    return df[label_col].map(clean_energy_label).map(LABEL_SCORE).astype(float)


def _centrality_features(
    df: pd.DataFrame,
    *,
    mode: str,
    year_col: str,
    area_col: str,
    label_col: str,
) -> dict[str, pd.Series]:
    features: dict[str, pd.Series] = {}
    for output_name, source_col in (("year", year_col), ("Area", area_col)):
        if source_col in df.columns:
            features[output_name] = pd.to_numeric(df[source_col], errors="coerce")
    label_score = _label_score_series(df, label_col)
    if label_score is not None:
        features["label_score"] = label_score

    if mode == "physics_central":
        for col in _PHYSICS_COLUMNS:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors="coerce")
        if "R" in df.columns and "C" in df.columns:
            features["tau"] = (
                pd.to_numeric(df["R"], errors="coerce")
                * pd.to_numeric(df["C"], errors="coerce")
            )
    return features


def _robust_absolute_z(values: pd.Series, *, log_scale: bool = False) -> pd.Series:
    work = pd.to_numeric(values, errors="coerce").astype(float)
    if log_scale:
        work = work.where(work > 0)
        work = np.log(work)
    valid = work.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    median = float(valid.median())
    iqr = float(valid.quantile(0.75) - valid.quantile(0.25))
    mad = float((valid - median).abs().median())
    scale = iqr if iqr > 0 else 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(0.0, index=values.index, dtype=float).where(work.notna())
    return (work - median).abs() / scale


def _group_columns(df: pd.DataFrame, group_cols: tuple[str, ...]) -> list[str]:
    return [col for col in group_cols if col in df.columns]


def _group_items(df: pd.DataFrame, group_cols: list[str]):
    if not group_cols:
        return [("__all__", df)]
    by = group_cols[0] if len(group_cols) == 1 else group_cols
    return list(df.groupby(by, dropna=False, sort=True))


def _set_edge_candidates(
    out: pd.DataFrame,
    group_cols: list[str],
    upper_tail_fraction: float,
) -> pd.Series:
    edge = pd.Series(False, index=out.index, dtype=bool)
    quantile = float(np.clip(1.0 - upper_tail_fraction, 0.0, 1.0))
    for _, group in _group_items(out, group_cols):
        scores = group["centrality_score"].dropna()
        if len(scores) <= 1 or float(scores.max()) == float(scores.min()):
            continue
        threshold = float(scores.quantile(quantile))
        edge.loc[group.index] = group["centrality_score"] >= threshold
    return edge


def add_building_centrality_score(
    df: pd.DataFrame,
    *,
    mode: str = "input_central",
    group_cols: tuple[str, ...] = ("vintage",),
    year_col: str = "year",
    area_col: str = "Area",
    label_col: str = "EnergyLabel",
) -> pd.DataFrame:
    """Add robust within-group centrality metrics; lower scores are more central."""
    _validate_mode(mode)
    scoring_mode = "input_central" if mode in {"random", "input_edge"} else mode
    out = _add_available_group_columns(df, group_cols, year_col)
    group_columns = _group_columns(out, group_cols)
    features = _centrality_features(
        out,
        mode=scoring_mode,
        year_col=year_col,
        area_col=area_col,
        label_col=label_col,
    )
    out["centrality_score"] = 0.0
    for _, group in _group_items(out, group_columns):
        z_values = [
            _robust_absolute_z(series.loc[group.index], log_scale=name in _LOG_SCALE_COLUMNS)
            for name, series in features.items()
        ]
        if z_values:
            score_frame = pd.concat(z_values, axis=1)
            out.loc[group.index, "centrality_score"] = score_frame.mean(axis=1).fillna(0.0)
    if group_columns:
        by = group_columns[0] if len(group_columns) == 1 else group_columns
        out["centrality_rank"] = (
            out.groupby(by, dropna=False)["centrality_score"]
            .rank(method="first", ascending=True)
            .astype(int)
        )
    else:
        out["centrality_rank"] = out["centrality_score"].rank(method="first").astype(int)
    out["is_edge_candidate"] = _set_edge_candidates(out, group_columns, 0.1)
    return out


def build_synthesis_input_from_selected(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Return generator-safe diagnostic inputs without fitted physical truth."""
    columns = [
        "input_id",
        "source_name",
        "source_user_id",
        "year",
        "Area",
        "EnergyLabel",
        "vintage",
    ]
    return selected_df[[col for col in columns if col in selected_df.columns]].copy()


def build_truth_table_from_selected(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Return available post-hoc truth columns for diagnostic comparisons."""
    out = selected_df.copy()
    if "tau" not in out.columns and {"R", "C"}.issubset(out.columns):
        out["tau"] = (
            pd.to_numeric(out["R"], errors="coerce")
            * pd.to_numeric(out["C"], errors="coerce")
        )
    columns = [
        "input_id",
        "source_name",
        "source_user_id",
        "user_id",
        "year",
        "Area",
        "EnergyLabel",
        "vintage",
        "R",
        "C",
        "A",
        "Qint",
        "C_per_area",
        "tau",
        "E_std_annual_per_m2",
    ]
    return out[[col for col in columns if col in out.columns]].copy()


def _group_key(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, tuple) else (value,)
    return tuple("<NA>" if pd.isna(item) else str(item) for item in values)


def _row_group_keys(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series([("__all__",)] * len(df), index=df.index)
    return df[group_cols].apply(lambda row: _group_key(tuple(row.tolist())), axis=1)


def _allocate_counts(weights: dict[tuple[str, ...], float], total: int) -> dict[tuple[str, ...], int]:
    if not weights or total <= 0:
        return {}
    positive = {key: max(float(value), 0.0) for key, value in weights.items()}
    weight_sum = sum(positive.values())
    if weight_sum <= 0:
        return {}
    raw = {key: total * value / weight_sum for key, value in positive.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _target_weights(
    available: pd.Series,
    target_group_counts: dict | pd.Series | None,
) -> dict[tuple[str, ...], float]:
    if target_group_counts is None:
        return {key: float(value) for key, value in available.items()}
    source = target_group_counts.items() if hasattr(target_group_counts, "items") else []
    targets = {_group_key(key): float(value) for key, value in source}
    return {key: targets.get(key, 0.0) for key in available.index}


def _ordered_indices(
    part: pd.DataFrame,
    *,
    mode: str,
    rng: np.random.Generator,
) -> list[Any]:
    if mode == "random":
        return list(rng.permutation(part.index.to_numpy()))
    if mode == "input_edge":
        return list(
            part.sort_values(
                ["centrality_score", "centrality_rank"],
                ascending=[False, False],
                kind="stable",
            ).index
        )
    return list(
        part.sort_values(
            ["is_edge_candidate", "centrality_score", "centrality_rank"],
            ascending=[True, True, True],
            kind="stable",
        ).index
    )


def select_representative_building_inputs(
    df: pd.DataFrame,
    config: BuildingInputSelectionConfig,
    source_name: str,
    target_group_counts: dict | pd.Series | None = None,
) -> SelectedBuildingInputs:
    """Select stable diagnostic rows while keeping synthesis inputs separate from truth."""
    _validate_mode(config.selection_mode)
    if config.n_buildings < 0:
        raise ValueError("n_buildings must be non-negative.")
    scored = add_building_centrality_score(
        df,
        mode=config.selection_mode,
        group_cols=config.group_cols,
        year_col=config.year_col,
        area_col=config.area_col,
        label_col=config.label_col,
    )
    group_columns = _group_columns(scored, config.group_cols)
    scored["__selection_group_key"] = _row_group_keys(scored, group_columns)
    scored["is_edge_candidate"] = _set_edge_candidates(
        scored,
        group_columns,
        config.avoid_edge_quantile,
    )
    n_select = min(config.n_buildings, len(scored))
    if config.n_buildings > len(scored):
        warnings.warn(
            f"Requested {config.n_buildings} rows from {source_name}, but only {len(scored)} are available."
        )
    available_counts = scored["__selection_group_key"].value_counts(sort=False)
    allocations = _allocate_counts(
        _target_weights(available_counts, target_group_counts),
        n_select,
    )
    rng = np.random.default_rng(config.random_state)
    chosen: list[Any] = []
    for key in sorted(allocations):
        part = scored.loc[scored["__selection_group_key"] == key]
        chosen.extend(_ordered_indices(part, mode=config.selection_mode, rng=rng)[: allocations[key]])
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < n_select:
        remaining = scored.drop(index=chosen)
        chosen.extend(
            _ordered_indices(remaining, mode=config.selection_mode, rng=rng)[: n_select - len(chosen)]
        )
    selected = scored.loc[chosen[:n_select]].copy().reset_index(drop=True)
    selected["source_name"] = source_name
    if "source_user_id" not in selected.columns and "user_id" in selected.columns:
        selected["source_user_id"] = selected["user_id"]
    selected.insert(0, "input_id", [f"{source_name}_{i + 1:03d}" for i in range(len(selected))])
    selected["selection_mode"] = config.selection_mode
    selected["selection_order"] = np.arange(1, len(selected) + 1)
    selected = selected.drop(columns=["__selection_group_key"], errors="ignore")
    summary_columns = [
        "input_id",
        "source_name",
        "source_user_id",
        *group_columns,
        config.year_col,
        config.area_col,
        config.label_col,
        "centrality_score",
        "centrality_rank",
        "is_edge_candidate",
        "selection_mode",
        "selection_order",
    ]
    summary = selected[[col for col in dict.fromkeys(summary_columns) if col in selected.columns]].copy()
    return SelectedBuildingInputs(
        source_name=source_name,
        selected_full_df=selected,
        synthesis_input_df=build_synthesis_input_from_selected(selected),
        truth_df=build_truth_table_from_selected(selected),
        selection_summary_df=summary,
    )
