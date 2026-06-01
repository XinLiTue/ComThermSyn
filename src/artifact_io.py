"""Public-safe artifact I/O helpers for deployment runtimes."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_FORBIDDEN_TERMS = (
    "user_id",
    "participant_id",
    "source_user_id",
    "address",
    "postcode",
    "postal",
    "zip",
    "truth",
    "holdout",
    "train_index",
    "test_index",
    "source_file",
    "private",
    "non_extreme_users",
    "houseinformation",
    "c:\\",
)

PUBLIC_SAFETY_DECLARATION_KEYS = {
    "private_data_included",
    "contains_user_ids",
    "contains_truth_tables",
}


def _normalise_terms(forbidden_terms: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(str(term).lower() for term in (forbidden_terms or DEFAULT_FORBIDDEN_TERMS))


def _contains_forbidden_term(value: Any, forbidden_terms: tuple[str, ...]) -> str | None:
    text = str(value).lower()
    return next((term for term in forbidden_terms if term in text), None)


def remove_private_columns(
    df: pd.DataFrame,
    forbidden_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of *df* with sensitive-looking columns removed."""
    forbidden_terms = _normalise_terms(forbidden_columns)
    columns_to_drop = [
        column
        for column in df.columns
        if _contains_forbidden_term(column, forbidden_terms) is not None
    ]
    return df.drop(columns=columns_to_drop, errors="ignore").copy()


def _validate_key(key: Any, path: str, forbidden_terms: tuple[str, ...]) -> None:
    key_text = str(key)
    if key_text in PUBLIC_SAFETY_DECLARATION_KEYS:
        return
    unsafe_term = _contains_forbidden_term(key_text, forbidden_terms)
    if unsafe_term:
        raise ValueError(f"Unsafe artifact key at {path}: {key_text!r} contains {unsafe_term!r}.")


def _validate_value(value: Any, path: str, forbidden_terms: tuple[str, ...]) -> None:
    if isinstance(value, pd.DataFrame):
        for column in value.columns:
            unsafe_term = _contains_forbidden_term(column, forbidden_terms)
            if unsafe_term:
                raise ValueError(
                    f"Unsafe DataFrame column at {path}: {column!r} contains {unsafe_term!r}."
                )
        return
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _validate_key(key, f"{path}.{key}", forbidden_terms)
            _validate_value(nested_value, f"{path}.{key}", forbidden_terms)
        return
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            _validate_value(item, f"{path}[{index}]", forbidden_terms)
        return
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if path.endswith(".privacy_note") or path.endswith(".artifact_root"):
            return
        lowered = value.lower()
        if ":\\" in lowered or lowered.startswith("/") and ("users/" in lowered or "home/" in lowered):
            raise ValueError(f"Unsafe string metadata at {path}: value looks like a local path.")
        unsafe_term = _contains_forbidden_term(value, forbidden_terms)
        if unsafe_term:
            raise ValueError(f"Unsafe string metadata at {path}: value contains {unsafe_term!r}.")


def _load_json_if_possible(path: Path) -> Any | None:
    try:
        return load_json(path)
    except Exception:
        return None


def validate_public_artifact_safety(
    payload_or_path: Any,
    forbidden_terms: Iterable[str] | None = None,
) -> None:
    """Validate practical public-artifact safety.

    The checker scans DataFrame columns, mapping keys, simple nested metadata,
    and JSON-like files/directories. It is intentionally conservative but not a
    brittle deep privacy scanner.
    """
    terms = _normalise_terms(forbidden_terms)
    if isinstance(payload_or_path, (str, Path)):
        path = Path(payload_or_path)
        unsafe_term = _contains_forbidden_term(path.name, terms)
        if unsafe_term:
            raise ValueError(f"Unsafe artifact path name {path!s}: contains {unsafe_term!r}.")
        if path.is_dir():
            for child in path.rglob("*"):
                unsafe_term = _contains_forbidden_term(child.name, terms)
                if unsafe_term:
                    raise ValueError(f"Unsafe artifact path name {child!s}: contains {unsafe_term!r}.")
                if child.suffix.lower() == ".json":
                    loaded = _load_json_if_possible(child)
                    if loaded is not None:
                        _validate_value(loaded, str(child), terms)
            return
        if path.suffix.lower() == ".json" and path.exists():
            _validate_value(load_json(path), str(path), terms)
            return
    _validate_value(payload_or_path, "artifact", terms)


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pickle(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_deployment_artifacts(
    artifacts: dict[str, Any],
    artifact_root: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save public deployment artifacts.

    The helper supports both the Step 1 mock payload and the Step 2
    ``public_sampler_v1`` payload. DataFrames for Step 2 are persisted as
    parquet files; the runtime loader accepts either artifact shape.
    """
    artifact_root = Path(artifact_root)
    model_dir = artifact_root / "model"
    weather_dir = artifact_root / "weather"
    runtime_config = artifacts.get("runtime_config", {})
    input_schema = artifacts.get("input_schema", {})
    model_metadata = dict(artifacts.get("model_metadata", {}))
    if metadata:
        model_metadata.update(metadata)
    mock_model_payload = artifacts.get("mock_model_payload")
    deployment_model_payload = artifacts.get("deployment_model_payload")
    typical_weather_scenarios = artifacts.get("typical_weather_scenarios", {})
    typical_day_weights = artifacts.get("typical_day_weights", {})
    public_sampling_baselines = artifacts.get("public_sampling_baselines")
    synthetic_residual_pool = artifacts.get("synthetic_residual_pool")
    expected_energy_baseline = artifacts.get("expected_energy_baseline")

    paths = {
        "runtime_config": str(model_dir / "runtime_config.json"),
        "input_schema": str(model_dir / "input_schema.json"),
        "model_metadata": str(model_dir / "model_metadata.json"),
        "typical_weather_scenarios": str(weather_dir / "typical_weather_scenarios.json"),
    }
    payload_for_validation = {
        "runtime_config": runtime_config,
        "input_schema": input_schema,
        "model_metadata": model_metadata,
        "typical_weather_scenarios": typical_weather_scenarios,
    }
    if mock_model_payload is not None:
        paths["mock_model_payload"] = str(model_dir / "mock_model_payload.pkl")
        payload_for_validation["mock_model_payload"] = mock_model_payload
    if deployment_model_payload is not None:
        paths["deployment_model_payload"] = str(model_dir / "deployment_model_payload.pkl")
        payload_for_validation["deployment_model_payload"] = deployment_model_payload
    if public_sampling_baselines is not None:
        paths["public_sampling_baselines"] = str(model_dir / "public_sampling_baselines.parquet")
        payload_for_validation["public_sampling_baselines"] = public_sampling_baselines
    if synthetic_residual_pool is not None:
        paths["synthetic_residual_pool"] = str(model_dir / "synthetic_residual_pool.parquet")
        payload_for_validation["synthetic_residual_pool"] = synthetic_residual_pool
    if expected_energy_baseline is not None:
        paths["expected_energy_baseline_public"] = str(model_dir / "expected_energy_baseline_public.parquet")
        payload_for_validation["expected_energy_baseline"] = expected_energy_baseline
    if typical_day_weights:
        paths["typical_day_weights"] = str(weather_dir / "typical_day_weights.json")
        payload_for_validation["typical_day_weights"] = typical_day_weights

    validate_public_artifact_safety(payload_for_validation)
    save_json(paths["runtime_config"], runtime_config)
    save_json(paths["input_schema"], input_schema)
    save_json(paths["model_metadata"], model_metadata)
    if mock_model_payload is not None:
        save_pickle(paths["mock_model_payload"], mock_model_payload)
    if deployment_model_payload is not None:
        save_pickle(paths["deployment_model_payload"], deployment_model_payload)
    if public_sampling_baselines is not None:
        target = Path(paths["public_sampling_baselines"])
        target.parent.mkdir(parents=True, exist_ok=True)
        public_sampling_baselines.to_parquet(target, index=False)
    if synthetic_residual_pool is not None:
        target = Path(paths["synthetic_residual_pool"])
        target.parent.mkdir(parents=True, exist_ok=True)
        synthetic_residual_pool.to_parquet(target, index=False)
    if expected_energy_baseline is not None:
        target = Path(paths["expected_energy_baseline_public"])
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_energy_baseline.to_parquet(target, index=False)
    save_json(paths["typical_weather_scenarios"], typical_weather_scenarios)
    if typical_day_weights:
        save_json(paths["typical_day_weights"], typical_day_weights)
    validate_public_artifact_safety(artifact_root)
    return paths


def load_deployment_artifacts(artifact_root: str | Path) -> dict[str, Any]:
    """Load Step 1 mock or Step 2 ``public_sampler_v1`` artifacts."""
    artifact_root = Path(artifact_root)
    model_dir = artifact_root / "model"
    weather_dir = artifact_root / "weather"
    artifacts: dict[str, Any] = {
        "runtime_config": load_json(model_dir / "runtime_config.json"),
        "input_schema": load_json(model_dir / "input_schema.json"),
        "model_metadata": load_json(model_dir / "model_metadata.json"),
        "artifact_root": str(artifact_root),
    }
    mock_path = model_dir / "mock_model_payload.pkl"
    if mock_path.exists():
        artifacts["mock_model_payload"] = load_pickle(mock_path)
    deployment_payload_path = model_dir / "deployment_model_payload.pkl"
    if deployment_payload_path.exists():
        artifacts["deployment_model_payload"] = load_pickle(deployment_payload_path)
    public_sampling_path = model_dir / "public_sampling_baselines.parquet"
    if public_sampling_path.exists():
        artifacts["public_sampling_baselines"] = pd.read_parquet(public_sampling_path)
    residual_pool_path = model_dir / "synthetic_residual_pool.parquet"
    if residual_pool_path.exists():
        artifacts["synthetic_residual_pool"] = pd.read_parquet(residual_pool_path)
    expected_energy_path = model_dir / "expected_energy_baseline_public.parquet"
    if expected_energy_path.exists():
        artifacts["expected_energy_baseline"] = pd.read_parquet(expected_energy_path)
    weather_path = weather_dir / "typical_weather_scenarios.json"
    if weather_path.exists():
        artifacts["typical_weather_scenarios"] = load_json(weather_path)
    typical_day_weights_path = weather_dir / "typical_day_weights.json"
    if typical_day_weights_path.exists():
        artifacts["typical_day_weights"] = load_json(typical_day_weights_path)
    validate_public_artifact_safety(artifacts)
    return artifacts
