from __future__ import annotations

import json
import math
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_MD_MISSING = "—"


def normalize_run_context(run_context: dict | None) -> dict[str, Any]:
    """Return JSON-safe run context with modern and legacy context keys aligned."""
    if not isinstance(run_context, dict):
        return {}

    normalized = {str(key): _clean_value(value) for key, value in run_context.items()}
    development_context = normalized.get("development_context")
    ai_conversation = normalized.get("ai_conversation")

    if _context_value_present(development_context):
        if not _context_value_present(ai_conversation):
            normalized["ai_conversation"] = development_context
    elif _context_value_present(ai_conversation):
        normalized["development_context"] = ai_conversation

    return _json_serializable_dict(normalized)


def save_run(
    locals_dict: dict,
    *,
    run_identity: dict | None = None,
    run_root: str | Path = "run_results",
) -> dict[str, Any]:
    """Save a compact, best-effort summary of a notebook run."""
    try:
        return _save_run_impl(locals_dict, run_identity=run_identity, run_root=run_root)
    except Exception as exc:
        print(f"Run logging skipped: {exc}")
        return {}


def _save_run_impl(
    locals_dict: dict,
    *,
    run_identity: dict | None,
    run_root: str | Path,
) -> dict[str, Any]:
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    timestamp_dt = datetime.now()
    final_save_timestamp = timestamp_dt.isoformat(timespec="seconds")
    safe_identity = _json_serializable_dict(run_identity or {})
    run_context = _extract_run_context(locals_dict)
    run_id = safe_identity.get("run_id") or _next_run_id(run_root, timestamp_dt)
    task_id = safe_identity.get("task_id") or run_context.get("task_id")
    artifact_id = safe_identity.get("artifact_id") or _build_artifact_id(task_id, run_id)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / f"{run_id}_summary.md"

    git_hash = _git_hash()
    run_duration_sec = _run_duration(locals_dict)

    config = _extract_config(locals_dict)
    runtime_timings_df = _extract_runtime_timings(locals_dict)
    metrics: dict[str, Any] = {}

    for extractor in (
        _extract_dataset_split,
        _extract_generation_quality,
        _extract_sanity_check,
        _extract_layer_a,
        _extract_layer_c,
        _extract_layer_d,
        _extract_representative_day_counts,
    ):
        try:
            metrics.update(extractor(locals_dict))
        except Exception:
            pass

    run_meta = {
        "run_id": run_id,
        "artifact_id": artifact_id,
        "timestamp": final_save_timestamp,
        "final_save_timestamp": final_save_timestamp,
        "git_hash": git_hash,
        "task_id": task_id,
        "project_log_entry_id": safe_identity.get("project_log_entry_id")
        or run_context.get("project_log_entry_id")
        or task_id,
        "project_log_path": safe_identity.get("project_log_path")
        or run_context.get("project_log_path")
        or "PROJECT_LOG.md",
        "identity_timestamp": safe_identity.get("identity_timestamp"),
        "prelim_run_token": safe_identity.get("prelim_run_token"),
        "task": run_context.get("task"),
        "development_context": run_context.get("development_context"),
        "ai_conversation": run_context.get("ai_conversation"),
        "code_change_summary": run_context.get("code_change_summary"),
        "expected_effect": run_context.get("expected_effect"),
        "run_context": run_context,
    }
    run_meta = _json_serializable_dict(run_meta)

    record = {
        "run_id": run_id,
        "artifact_id": artifact_id,
        "timestamp": final_save_timestamp,
        "final_save_timestamp": final_save_timestamp,
        "git_hash": git_hash,
        "run_duration_sec": run_duration_sec,
        "run_dir": _posix(run_dir),
        "summary_path": _posix(summary_path),
        "run_identity": safe_identity,
        "run_meta": run_meta,
        "run_context": _json_safe(run_context),
        "metrics": _json_safe(metrics),
        "config": _json_safe(config),
    }

    summary_path.write_text(
        _render_markdown(
            run_meta,
            run_duration_sec,
            run_context,
            config,
            metrics,
            runtime_timings_df=runtime_timings_df,
        ),
        encoding="utf-8",
    )

    with (run_root / "run_results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved run directory: {_posix(run_dir)}")
    print(f"Saved summary:       {_posix(summary_path)}")
    print("Suggested git commit:")
    print(f'  git commit -m "log experiment run {run_id}"')
    return run_meta


def get_git_hash() -> str:
    return _git_hash()


def _context_value_present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "manual_fill"}


def validate_run_context(run_context: dict, allow_incomplete: bool = False) -> None:
    if not isinstance(run_context, dict):
        raise ValueError("RUN_CONTEXT must be a dictionary.")
    normalized_context = normalize_run_context(run_context)
    required = [
        "task_id",
        "project_log_path",
        "project_log_entry_id",
        "task",
        "code_change_summary",
        "expected_effect",
    ]
    missing = []
    for field in required:
        if not _context_value_present(normalized_context.get(field)):
            missing.append(field)
    if not _context_value_present(normalized_context.get("development_context")):
        missing.append("development_context or ai_conversation")
    if missing and not allow_incomplete:
        raise ValueError(
            "RUN_CONTEXT is incomplete. Fill these fields or set "
            f"ALLOW_INCOMPLETE_RUN_CONTEXT=True: {', '.join(missing)}"
        )


def build_run_identity(run_context: dict, allow_incomplete: bool = False) -> dict[str, Any]:
    """Build a lightweight preliminary identity from RUN_CONTEXT only.

    This function must not receive or inspect the full notebook namespace.
    It does not allocate the persistent sequential run_id; save_run() does that.
    """
    normalized_context = normalize_run_context(run_context)
    validate_run_context(normalized_context, allow_incomplete=allow_incomplete)
    task_id = normalized_context.get("task_id")
    project_log_entry_id = normalized_context.get("project_log_entry_id", task_id)
    project_log_path = normalized_context.get("project_log_path", "PROJECT_LOG.md")
    identity_timestamp = datetime.now().isoformat(timespec="seconds")
    git_hash = get_git_hash()
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "manual_fill")).strip("_")
    prelim_stamp = identity_timestamp.replace(":", "").replace("-", "")
    prelim_run_token = f"prelim_{safe_task_id}_{prelim_stamp}"
    identity = {
        "task_id": str(task_id),
        "project_log_entry_id": str(project_log_entry_id),
        "project_log_path": str(project_log_path),
        "identity_timestamp": str(identity_timestamp),
        "git_hash": str(git_hash),
        "prelim_run_token": str(prelim_run_token),
        "task": normalized_context.get("task"),
        "development_context": normalized_context.get("development_context"),
        "ai_conversation": normalized_context.get("ai_conversation"),
        "code_change_summary": normalized_context.get("code_change_summary"),
        "expected_effect": normalized_context.get("expected_effect"),
        "run_context": normalized_context,
    }
    return _json_serializable_dict(identity)


def update_streamlit_manifests_with_run_meta(
    streamlit_root: str | Path,
    run_meta: dict[str, Any],
) -> None:
    streamlit_root = Path(streamlit_root)
    safe_run_meta = _json_serializable_dict(run_meta or {})
    run_context = normalize_run_context(safe_run_meta.get("run_context"))
    if run_context:
        safe_run_meta["run_context"] = run_context
        for field in [
            "task_id",
            "project_log_entry_id",
            "task",
            "development_context",
            "ai_conversation",
            "code_change_summary",
            "expected_effect",
        ]:
            if run_context.get(field) is not None:
                safe_run_meta[field] = run_context[field]
    for manifest_path in [
        streamlit_root / "manifest.json",
        streamlit_root / "validation" / "manifest.json",
    ]:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = _read_json_dict(manifest_path)
        manifest.update(safe_run_meta)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated Streamlit manifest with run metadata: {_posix(manifest_path)}")


def _next_run_id(run_root: Path, timestamp_dt: datetime) -> str:
    max_seq = 0
    pattern = re.compile(r"^run_(\d+)")
    for path in run_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            try:
                max_seq = max(max_seq, int(match.group(1)))
            except Exception:
                pass
    return f"run_{max_seq + 1:03d}_{timestamp_dt:%Y-%m-%d_%H-%M}"


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _run_duration(locals_dict: dict) -> float | None:
    try:
        run_start = locals_dict.get("_run_start")
        if run_start is None:
            return None
        return _round(time.time() - float(run_start))
    except Exception:
        return None


def _extract_config(locals_dict: dict) -> dict[str, Any]:
    fields = {
        "analysis_mode": "ANALYSIS_MODE",
        "validation_mode": "VALIDATION_MODE",
        "community_size": "COMMUNITY_SIZES_TO_RUN",
        "n_cases_per_size": "N_CASES_PER_SIZE",
        "n_population_candidates": "N_POPULATION_CANDIDATES_CASE_SIZE",
        "top_k_proxy": "TOP_K_PROXY_CASE_SIZE",
        "top_k_full": "TOP_K_FULL_CASE_SIZE",
        "run_case_size": "RUN_CASE_SIZE",
        "run_sanity_check": "RUN_SANITY_CHECK",
        "run_case_validation": "RUN_CASE_VALIDATION",
        "representative_n_days": "REPRESENTATIVE_N_DAYS",
    }
    config: dict[str, Any] = {}
    for out_name, source_name in fields.items():
        try:
            config[out_name] = _clean_value(locals_dict.get(source_name))
        except Exception:
            config[out_name] = None
    return config


def _extract_run_context(locals_dict: dict) -> dict[str, Any]:
    run_context = locals_dict.get("RUN_CONTEXT", {})
    return normalize_run_context(run_context)


def _extract_runtime_timings(locals_dict: dict) -> pd.DataFrame:
    """Return valid recorded runtime rows for markdown summaries only."""
    runtime_df = locals_dict.get("runtime_timings_df")
    columns = ["section", "seconds", "note"]
    if not isinstance(runtime_df, pd.DataFrame) or runtime_df.empty:
        return pd.DataFrame(columns=columns)
    if not {"section", "seconds"}.issubset(runtime_df.columns):
        return pd.DataFrame(columns=columns)
    out = runtime_df.copy()
    if "note" not in out.columns:
        out["note"] = None
    out = out[columns].copy()
    out["seconds"] = pd.to_numeric(out["seconds"], errors="coerce")
    out = out.loc[out["seconds"].notna()].reset_index(drop=True)
    return out


def _extract_dataset_split(locals_dict: dict) -> dict[str, Any]:
    out = {
        "n_total": None,
        "n_train": None,
        "n_test": None,
        "tested_vintages": None,
        "untested_vintages": None,
    }
    try:
        model_df = locals_dict.get("model_df")
        if isinstance(model_df, pd.DataFrame):
            out["n_total"] = int(len(model_df))
    except Exception:
        pass
    try:
        model_df_train = locals_dict.get("model_df_train")
        if isinstance(model_df_train, pd.DataFrame):
            out["n_train"] = int(len(model_df_train))
    except Exception:
        pass
    try:
        model_df_test = locals_dict.get("model_df_test")
        if isinstance(model_df_test, pd.DataFrame):
            out["n_test"] = int(len(model_df_test))
    except Exception:
        pass
    try:
        model_df = locals_dict.get("model_df")
        model_df_test = locals_dict.get("model_df_test")
        if (
            isinstance(model_df, pd.DataFrame)
            and isinstance(model_df_test, pd.DataFrame)
            and "vintage" in model_df.columns
            and "vintage" in model_df_test.columns
        ):
            tested = _sorted_values(model_df_test["vintage"].dropna().unique())
            all_vintages = set(model_df["vintage"].dropna().unique())
            tested_set = set(model_df_test["vintage"].dropna().unique())
            out["tested_vintages"] = tested
            out["untested_vintages"] = _sorted_values(all_vintages - tested_set)
    except Exception:
        pass
    return out


def _extract_generation_quality(locals_dict: dict) -> dict[str, Any]:
    columns = [
        "abs_mean_E_error",
        "median_E_error",
        "total_kWh_error_pct",
        "wasserstein_E",
        "wasserstein_R",
        "wasserstein_C",
        "wasserstein_A",
        "wasserstein_Qint",
        "label_ordering_score",
        "label_ordering_ok",
        "renovation_R_ok",
        "score_full",
        "coverage_ratio_full",
    ]
    out = {col: None for col in columns}
    try:
        df = locals_dict.get("case_size_summary_df")
        if isinstance(df, pd.DataFrame) and len(df):
            row = df.iloc[0]
            for col in columns:
                if col in df.columns:
                    out[col] = _clean_value(row[col])
    except Exception:
        pass
    return out


def _extract_sanity_check(locals_dict: dict) -> dict[str, Any]:
    out = {"sanity_n_rows": None, "sanity_n_pass": None, "sanity_pass_rate": None}
    try:
        summary = locals_dict.get("sanity_summary_dict", {})
        if summary:
            out["sanity_n_rows"]    = summary.get("n_rows")
            out["sanity_n_pass"]    = summary.get("n_pass")
            out["sanity_pass_rate"] = _round(summary.get("pass_rate"))
            return out
        df = locals_dict.get("sanity_report_df")
        if isinstance(df, pd.DataFrame) and len(df):
            out["sanity_n_rows"] = int(len(df))
    except Exception:
        pass
    return out


def _extract_layer_a(locals_dict: dict) -> dict[str, Any]:
    out = {
        "A_generated_daily_kwh_err_mean": None,
        "A_generated_peak_kw_err_mean": None,
        "A_fitted_baseline_daily_kwh_err_mean": None,
        "A_fitted_baseline_peak_kw_err_mean": None,
    }
    try:
        df = locals_dict.get("day_metrics_df")
        if not isinstance(df, pd.DataFrame) or "method" not in df.columns:
            return out
        mappings = [
            ("generated", "daily_kwh_err", "A_generated_daily_kwh_err_mean"),
            ("generated", "peak_kw_err", "A_generated_peak_kw_err_mean"),
            ("fitted_baseline", "daily_kwh_err", "A_fitted_baseline_daily_kwh_err_mean"),
            ("fitted_baseline", "peak_kw_err", "A_fitted_baseline_peak_kw_err_mean"),
        ]
        for method, column, key in mappings:
            if column in df.columns:
                values = pd.to_numeric(df.loc[df["method"] == method, column], errors="coerce")
                if values.notna().any():
                    out[key] = _round(values.mean())
    except Exception:
        pass
    return out


def _extract_layer_c(locals_dict: dict) -> dict[str, Any]:
    out = {"C_reliable_cases": None, "C_total_cases": None, "C_gen_tau_rel_err": None}
    try:
        df = locals_dict.get("tau_df")
        if not isinstance(df, pd.DataFrame):
            return out
        if "reliable" in df.columns:
            if "case_id" in df.columns:
                out["C_reliable_cases"] = int(
                    df[df["reliable"] == True]["case_id"].nunique()  # noqa: E712
                )
            else:
                out["C_reliable_cases"] = int((df["reliable"] == True).sum())  # noqa: E712
        if "case_id" in df.columns:
            out["C_total_cases"] = int(df["case_id"].nunique())
        if {"method", "reliable", "tau_rel_err_vs_proxy"}.issubset(df.columns):
            mask = (df["method"] == "generated") & (df["reliable"] == True)  # noqa: E712
            values = pd.to_numeric(df.loc[mask, "tau_rel_err_vs_proxy"], errors="coerce")
            if values.notna().any():
                out["C_gen_tau_rel_err"] = _round(values.mean())
    except Exception:
        pass
    return out


def _extract_layer_d(locals_dict: dict) -> dict[str, Any]:
    out = {
        "D_generated_ramp_rmse_residual_mean": None,
        "D_fitted_baseline_ramp_rmse_residual_mean": None,
    }
    try:
        df = locals_dict.get("ramp_df")
        if (
            not isinstance(df, pd.DataFrame)
            or "method" not in df.columns
            or "ramp_rmse_residual" not in df.columns
        ):
            return out
        for method, key in [
            ("generated", "D_generated_ramp_rmse_residual_mean"),
            ("fitted_baseline", "D_fitted_baseline_ramp_rmse_residual_mean"),
        ]:
            values = pd.to_numeric(df.loc[df["method"] == method, "ramp_rmse_residual"], errors="coerce")
            if values.notna().any():
                out[key] = _round(values.mean())
    except Exception:
        pass
    return out


def _extract_representative_day_counts(locals_dict: dict) -> dict[str, Any]:
    out = {
        "selected_representative_validation_buildings": None,
        "case_validation_summary_rows": None,
        "day_metrics_rows": None,
        "tau_rows": None,
        "ramp_rows": None,
    }
    try:
        case_df = locals_dict.get("_case_df")
        if isinstance(case_df, pd.DataFrame):
            out["selected_representative_validation_buildings"] = int(len(case_df))
    except Exception:
        pass
    for local_name, metric_name in [
        ("case_validation_summary_df", "case_validation_summary_rows"),
        ("day_metrics_df", "day_metrics_rows"),
        ("tau_df", "tau_rows"),
        ("ramp_df", "ramp_rows"),
    ]:
        try:
            df = locals_dict.get(local_name)
            if isinstance(df, pd.DataFrame):
                out[metric_name] = int(len(df))
        except Exception:
            pass
    return out


def _is_pass(v: Any) -> bool:
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    if isinstance(v, bool):
        return v
    try:
        return float(v) > 0
    except Exception:
        return False


def _clean_value(value: Any) -> Any:
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    try:
        if value is None or value is pd.NA:
            return None
    except Exception:
        if value is None:
            return None

    if isinstance(value, (np.ndarray, pd.Index)):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, set):
        return [_json_safe(v) for v in _sorted_values(value)]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return _round(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _json_serializable_dict(value: dict[str, Any]) -> dict[str, Any]:
    cleaned = _json_safe(value)
    if not isinstance(cleaned, dict):
        cleaned = {}
    try:
        json.dumps(cleaned)
        return cleaned
    except TypeError:
        return {str(k): str(v) for k, v in cleaned.items()}


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _build_artifact_id(task_id: Any, run_id: str) -> str:
    task = str(task_id or "unknown_task")
    task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task).strip("_") or "unknown_task"
    return f"{task}_{run_id}"


def _round(value: Any) -> float | None:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 3)
    except Exception:
        return None


def _sorted_values(values: Any) -> list[Any]:
    cleaned = [_json_safe(v) for v in values]
    try:
        return sorted(cleaned)
    except Exception:
        return sorted(cleaned, key=lambda x: str(x))


def _md_value(value: Any) -> str:
    value = _json_safe(value)
    if value is None:
        return _MD_MISSING
    if isinstance(value, (list, tuple)):
        return ", ".join(_md_value(v) for v in value) if value else _MD_MISSING
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _posix(path: Path) -> str:
    return path.as_posix()


def _render_markdown(
    run_meta: dict[str, Any],
    run_duration_sec: float | None,
    run_context: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
    runtime_timings_df: pd.DataFrame | None = None,
) -> str:
    run_context_rows = [
        ("Task ID", run_context.get("task_id")),
        ("Project log entry ID", run_context.get("project_log_entry_id")),
        ("Project log path", run_context.get("project_log_path")),
        ("Task", run_context.get("task")),
        ("Development context", run_context.get("development_context")),
        ("AI conversation", run_context.get("ai_conversation")),
        ("Code change summary", run_context.get("code_change_summary")),
        ("Expected effect", run_context.get("expected_effect")),
    ]
    config_rows = [
        ("Analysis mode", config.get("analysis_mode")),
        ("Validation mode", config.get("validation_mode")),
        ("Community size", config.get("community_size")),
        ("Number of cases per size", config.get("n_cases_per_size")),
        ("Number of population candidates", config.get("n_population_candidates")),
        ("Top-k proxy", config.get("top_k_proxy")),
        ("Top-k full", config.get("top_k_full")),
        ("Run case size", config.get("run_case_size")),
        ("Run sanity check", config.get("run_sanity_check")),
        ("Run case validation", config.get("run_case_validation")),
        ("Representative number of days", config.get("representative_n_days")),
    ]
    dataset_rows = [
        ("Total buildings", metrics.get("n_total")),
        ("Training buildings", metrics.get("n_train")),
        ("Test buildings", metrics.get("n_test")),
        ("Tested vintages", metrics.get("tested_vintages")),
        ("Untested vintages", metrics.get("untested_vintages")),
    ]
    generation_rows = [
        ("Absolute mean energy error", metrics.get("abs_mean_E_error")),
        ("Median energy error", metrics.get("median_E_error")),
        ("Total kWh error percent", metrics.get("total_kWh_error_pct")),
        ("Wasserstein E", metrics.get("wasserstein_E")),
        ("Wasserstein R", metrics.get("wasserstein_R")),
        ("Wasserstein C", metrics.get("wasserstein_C")),
        ("Wasserstein A", metrics.get("wasserstein_A")),
        ("Wasserstein Qint", metrics.get("wasserstein_Qint")),
        ("Label ordering score", metrics.get("label_ordering_score")),
        ("Label ordering OK", metrics.get("label_ordering_ok")),
        ("Renovation R OK", metrics.get("renovation_R_ok")),
        ("Full score", metrics.get("score_full")),
        ("Full coverage ratio", metrics.get("coverage_ratio_full")),
    ]
    sanity_rows = [
        ("Number of rows", metrics.get("sanity_n_rows")),
        ("Number passed", metrics.get("sanity_n_pass")),
        ("Pass rate", metrics.get("sanity_pass_rate")),
    ]
    layer_a_rows = [
        ("Generated daily kWh error mean", metrics.get("A_generated_daily_kwh_err_mean")),
        ("Generated peak kW error mean", metrics.get("A_generated_peak_kw_err_mean")),
        ("Fitted baseline daily kWh error mean", metrics.get("A_fitted_baseline_daily_kwh_err_mean")),
        ("Fitted baseline peak kW error mean", metrics.get("A_fitted_baseline_peak_kw_err_mean")),
    ]
    layer_c_rows = [
        ("Reliable cases", metrics.get("C_reliable_cases")),
        ("Total cases", metrics.get("C_total_cases")),
        ("Generated tau relative error", metrics.get("C_gen_tau_rel_err")),
    ]
    layer_d_rows = [
        ("Generated ramp RMSE residual mean", metrics.get("D_generated_ramp_rmse_residual_mean")),
        (
            "Fitted baseline ramp RMSE residual mean",
            metrics.get("D_fitted_baseline_ramp_rmse_residual_mean"),
        ),
    ]
    representative_day_rows = [
        (
            "Number of selected representative validation buildings",
            metrics.get("selected_representative_validation_buildings"),
        ),
        ("Rows in case_validation_summary_df", metrics.get("case_validation_summary_rows")),
        ("Rows in day_metrics_df", metrics.get("day_metrics_rows")),
        ("Rows in tau_df", metrics.get("tau_rows")),
        ("Rows in ramp_df", metrics.get("ramp_rows")),
    ]

    lines = [
        f"# Run Summary: {run_meta.get('run_id')}",
        "",
        "## Header",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Task ID | {_md_value(run_meta.get('task_id'))} |",
        f"| Project log entry ID | {_md_value(run_meta.get('project_log_entry_id'))} |",
        f"| Run ID | {_md_value(run_meta.get('run_id'))} |",
        f"| Artifact ID | {_md_value(run_meta.get('artifact_id'))} |",
        f"| Git hash | {_md_value(run_meta.get('git_hash'))} |",
        f"| Identity timestamp | {_md_value(run_meta.get('identity_timestamp'))} |",
        f"| Final save timestamp | {_md_value(run_meta.get('final_save_timestamp'))} |",
        f"| Timestamp | {_md_value(run_meta.get('timestamp'))} |",
        f"| Run duration (sec) | {_md_value(run_duration_sec)} |",
        "",
        "## Run Context",
        "",
        _md_table("Field", "Value", run_context_rows),
        "",
    ]
    if Path("PROJECT_LOG.md").exists():
        lines.extend(["Project log: `PROJECT_LOG.md`", ""])
    lines.extend(
        [
            "## Manual Notes After Run",
            "",
            "- Notes:",
            "- Errors this run:",
            "- Comparison with previous run:",
            "",
            "## Config",
            "",
            _md_table("Field", "Value", config_rows),
            "",
            "## Dataset Split",
            "",
            _md_table("Metric", "Value", dataset_rows),
            "",
            "## Generation Quality",
            "",
            "Note: values extracted from row 0 of `case_size_summary_df`.",
            "",
            _md_table("Metric", "Value", generation_rows),
            "",
            "## Sanity Check",
            "",
            _md_table("Metric", "Value", sanity_rows),
            "",
            "## Layer A - Energy Error",
            "",
            _md_table("Metric", "Value", layer_a_rows),
            "",
            "## Layer C - Tau",
            "",
            _md_table("Metric", "Value", layer_c_rows),
            "",
            "## Layer D - Ramp Residuals",
            "",
            _md_table("Metric", "Value", layer_d_rows),
            "",
            "## Representative-Day Validation",
            "",
            _md_table("Metric", "Value", representative_day_rows),
            "",
        ]
    )
    if isinstance(runtime_timings_df, pd.DataFrame) and len(runtime_timings_df):
        lines.extend(
            [
                "## Runtime Timings",
                "",
                "| Section | Seconds | Notes |",
                "|---|---:|---|",
            ]
        )
        for timing in runtime_timings_df.itertuples(index=False):
            section = _md_value(timing.section).replace("|", "\\|")
            seconds = _md_value(_round(timing.seconds)).replace("|", "\\|")
            note = _md_value(timing.note).replace("|", "\\|")
            lines.append(f"| {section} | {seconds} | {note} |")
        lines.append("")
    return "\n".join(lines)


def _md_table(left_header: str, right_header: str, rows: list[tuple[str, Any]]) -> str:
    lines = [f"| {left_header} | {right_header} |", "|---|---|"]
    for label, value in rows:
        lines.append(f"| {label} | {_md_value(value)} |")
    return "\n".join(lines)
