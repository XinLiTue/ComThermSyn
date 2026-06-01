"""
Export helpers: flatten artifacts, save CSV/JSON, build manifest.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils import save_df as _save_df_path, save_jsonable as _save_jsonable_path


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def expand_case_size_artifacts_payloads(
    case_artifacts: Dict[Tuple[int, int], Dict[str, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """Flatten {(size, case_id): payload} into concatenated DataFrames."""
    generated_rows: List[pd.DataFrame] = []
    candidate_rows: List[pd.DataFrame] = []
    real_rows: List[pd.DataFrame] = []
    payload_summary_rows: List[Dict] = []

    for (community_size, case_id), payload in sorted(case_artifacts.items()):
        case_name = f"size{community_size}_case{case_id}"
        payload_summary_rows.append({
            "community_size": community_size,
            "case_id": case_id,
            "case_name": case_name,
            "payload_keys": sorted(payload.keys()),
        })
        for key in ["generated_df", "real_df", "score_df", "community_score_df"]:
            obj = payload.get(key)
            if not isinstance(obj, pd.DataFrame) or len(obj) == 0:
                continue
            tagged = obj.copy()
            tagged["community_size"] = community_size
            tagged["case_id"] = case_id
            tagged["case_name"] = case_name
            if key == "generated_df":
                generated_rows.append(tagged)
            elif key in {"score_df", "community_score_df"}:
                tagged["score_source_key"] = key
                candidate_rows.append(tagged)
            elif key == "real_df":
                real_rows.append(tagged)

    return {
        "generated_df": pd.concat(generated_rows, ignore_index=True) if generated_rows else pd.DataFrame(),
        "candidate_scores_df": pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame(),
        "real_df": pd.concat(real_rows, ignore_index=True) if real_rows else pd.DataFrame(),
        "payload_summary_df": pd.DataFrame(payload_summary_rows),
    }


def flatten_annual_building_summaries(
    building_summaries: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames = []
    for case_name, bdf in building_summaries.items():
        if not isinstance(bdf, pd.DataFrame) or len(bdf) == 0:
            continue
        tmp = bdf.copy()
        tmp["case_name"] = case_name
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flatten_profiles_by_size(
    profiles_by_size: Dict[int, Dict[Tuple, pd.DataFrame]],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for community_size, prof_map in profiles_by_size.items():
        for (role, weather_kind), prof_df in prof_map.items():
            if not isinstance(prof_df, pd.DataFrame) or len(prof_df) == 0:
                continue
            tmp = prof_df.reset_index().copy()
            time_col = tmp.columns[0]
            tmp = tmp.rename(columns={time_col: "time"})
            tmp["community_size"] = community_size
            tmp["role"] = role
            tmp["weather_kind"] = weather_kind
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Manifest tracking
# ---------------------------------------------------------------------------

def new_manifest(export_root: Path, identity_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    export_timestamp = datetime.now().isoformat(timespec="seconds")
    identity_metadata = _jsonable_dict(identity_metadata or {})
    return {
        "timestamp": export_timestamp,
        "export_timestamp": export_timestamp,
        "export_root": str(export_root),
        **identity_metadata,
        "saved_files": [],
        "missing_items": [],
        "notes": [],
    }


def record_saved(manifest: Dict[str, Any], path: Path, obj: Any, source_name: str) -> None:
    rows = cols = None
    if isinstance(obj, pd.DataFrame):
        rows, cols = int(len(obj)), list(obj.columns)
    elif isinstance(obj, dict):
        rows, cols = len(obj), list(obj.keys())
    elif isinstance(obj, list):
        rows = len(obj)
    manifest["saved_files"].append({"path": str(path), "source_variable": source_name, "rows": rows, "columns": cols})


def record_missing(manifest: Dict[str, Any], name: str, note: str = "variable not found") -> None:
    manifest["missing_items"].append({"name": name, "note": note})


def save_df_to_manifest(
    manifest: Dict[str, Any],
    df: Optional[pd.DataFrame],
    path: Path,
    source_name: str,
) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        record_missing(manifest, source_name, f"expected DataFrame, got {type(df).__name__ if df is not None else 'None'}")
        return
    _save_df_path(df, path)
    record_saved(manifest, path, df, source_name)


def save_jsonable_to_manifest(
    manifest: Dict[str, Any],
    obj: Any,
    path: Path,
    source_name: str,
) -> None:
    _save_jsonable_path(obj, path)
    record_saved(manifest, path, obj, source_name)


# ---------------------------------------------------------------------------
# Top-level export pipeline
# ---------------------------------------------------------------------------

def run_export_pipeline(
    export_root: Path,
    *,
    case_size_artifacts: Optional[Dict] = None,
    case_size_summary_df: Optional[pd.DataFrame] = None,
    case_size_detail_df: Optional[pd.DataFrame] = None,
    annual_case_df: Optional[pd.DataFrame] = None,
    annual_building_summaries: Optional[Dict[str, pd.DataFrame]] = None,
    profiles_by_size: Optional[Dict] = None,
    sanity_report_df: Optional[pd.DataFrame] = None,
    conditional_validation_df: Optional[pd.DataFrame] = None,
    monotonicity_report_df: Optional[pd.DataFrame] = None,
    pit_validation_detail_df: Optional[pd.DataFrame] = None,
    pit_validation_summary_df: Optional[pd.DataFrame] = None,
    case_validation_summary_df: Optional[pd.DataFrame] = None,
    input_config: Optional[Dict] = None,
    identity_metadata: Optional[Dict] = None,
    fallback_generated_df: Optional[pd.DataFrame] = None,
    fallback_score_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Save all pipeline artifacts to ``export_root`` under inputs/, results/, validation/.

    Returns the manifest dict.
    """
    export_root = Path(export_root)
    inputs_dir = export_root / "inputs"
    results_dir = export_root / "results"
    validation_dir = export_root / "validation"
    for d in [export_root, inputs_dir, results_dir, validation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = new_manifest(export_root, identity_metadata=identity_metadata)

    # -- inputs --
    if input_config is not None:
        save_jsonable_to_manifest(manifest, input_config, export_root / "input_config.json", "input_config")

    if case_size_artifacts:
        expanded = expand_case_size_artifacts_payloads(case_size_artifacts)
        if len(expanded["real_df"]):
            bldg_export = expanded["real_df"].copy().rename(columns={
                "year": "construction_year", "Area": "floor_area", "EnergyLabel": "energy_label"
            })
            if "building_id" not in bldg_export.columns:
                bldg_export["building_id"] = [f"input_{i+1:02d}" for i in range(len(bldg_export))]
            save_df_to_manifest(manifest, bldg_export, inputs_dir / "building_inputs.csv", "case_size_artifacts.real_df")
    elif fallback_generated_df is not None:
        save_df_to_manifest(manifest, fallback_generated_df, inputs_dir / "building_inputs.csv", "fallback_generated_df")

    # -- results --
    if case_size_artifacts:
        expanded = expand_case_size_artifacts_payloads(case_size_artifacts)
        if len(expanded["generated_df"]):
            save_df_to_manifest(manifest, expanded["generated_df"], results_dir / "generated_rcaq.csv", "case_size_artifacts.generated_df")
        elif fallback_generated_df is not None:
            save_df_to_manifest(manifest, fallback_generated_df, results_dir / "generated_rcaq.csv", "fallback_generated_df")
        else:
            record_missing(manifest, "generated_rcaq.csv", "no generated dataframe found")

        if len(expanded["candidate_scores_df"]):
            save_df_to_manifest(manifest, expanded["candidate_scores_df"], results_dir / "candidate_scores.csv", "case_size_artifacts.score_df")
        elif fallback_score_df is not None:
            save_df_to_manifest(manifest, fallback_score_df, results_dir / "candidate_scores.csv", "fallback_score_df")

        if len(expanded["payload_summary_df"]):
            save_df_to_manifest(manifest, expanded["payload_summary_df"], results_dir / "case_size_artifact_manifest.csv", "case_size_artifacts")
    elif fallback_generated_df is not None:
        save_df_to_manifest(manifest, fallback_generated_df, results_dir / "generated_rcaq.csv", "fallback_generated_df")
        if fallback_score_df is not None:
            save_df_to_manifest(manifest, fallback_score_df, results_dir / "candidate_scores.csv", "fallback_score_df")

    for df_obj, file_name, src_name in [
        (case_size_summary_df, "case_size_summary.csv", "case_size_summary_df"),
        (case_size_detail_df, "case_size_detail.csv", "case_size_detail_df"),
        (annual_case_df, "annual_community_summary.csv", "annual_case_df"),
    ]:
        if isinstance(df_obj, pd.DataFrame):
            save_df_to_manifest(manifest, df_obj, results_dir / file_name, src_name)
        else:
            record_missing(manifest, file_name, f"{src_name} not available")

    if annual_building_summaries:
        flat_bldg = flatten_annual_building_summaries(annual_building_summaries)
        if len(flat_bldg):
            save_df_to_manifest(manifest, flat_bldg, results_dir / "annual_building_summary.csv", "annual_building_summaries")

    if profiles_by_size:
        flat_prof = flatten_profiles_by_size(profiles_by_size)
        if len(flat_prof):
            save_df_to_manifest(manifest, flat_prof, results_dir / "selected_24h_profiles.csv", "profiles_by_size")

    # -- validation --
    for df_obj, file_name, src_name in [
        (sanity_report_df, "sanity_check.csv", "sanity_report_df"),
        (conditional_validation_df, "conditional_validation.csv", "conditional_validation_df"),
        (monotonicity_report_df, "monotonicity_validation.csv", "monotonicity_report_df"),
        (pit_validation_detail_df, "pit_validation.csv", "pit_validation_detail_df"),
        (case_validation_summary_df, "case_validation.csv", "case_validation_summary_df"),
    ]:
        if isinstance(df_obj, pd.DataFrame):
            save_df_to_manifest(manifest, df_obj, validation_dir / file_name, src_name)
        else:
            record_missing(manifest, file_name, f"{src_name} not available")

    if pit_validation_summary_df is not None and len(pit_validation_summary_df):
        save_df_to_manifest(manifest, pit_validation_summary_df, validation_dir / "pit_validation_summary.csv", "pit_validation_summary_df")

    save_jsonable_to_manifest(manifest, manifest.copy(), export_root / "manifest.json", "manifest")
    return manifest


def _jsonable_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        json.dumps(value)
        return value
    except TypeError:
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            try:
                json.dumps(item)
                cleaned[str(key)] = item
            except TypeError:
                cleaned[str(key)] = str(item)
        return cleaned
