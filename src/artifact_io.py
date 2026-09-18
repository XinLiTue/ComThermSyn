from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_ARTIFACT_TOKENS = (
    "address",
    "postcode",
    "participant",
    "private_id",
    "source_user",
    "user_id",
    "validation_target",
    "holdout_target",
    "final_test",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    for item in manifest.get("files", []):
        relative_path = Path(str(item["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe manifest path: {relative_path}")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing deployment artifact: {path}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"artifact hash mismatch: {relative_path}")


def load_deployment_artifacts(artifact_root: Path) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    model_root = root / "model"
    weather_root = root / "weather"
    manifest = _read_json(model_root / "model_manifest.json")
    _verify_manifest(root, manifest)

    correlation_payload = _read_json(model_root / "copula_correlation.json")
    artifacts = {
        "artifact_root": root,
        "model_manifest": manifest,
        "model_metadata": _read_json(model_root / "model_metadata.json"),
        "runtime_config": _read_json(model_root / "runtime_config.json"),
        "input_schema": _read_json(model_root / "input_schema.json"),
        "parameter_draws": pd.read_parquet(model_root / "parameter_draws.parquet"),
        "residual_marginals": pd.read_parquet(model_root / "residual_marginals.parquet"),
        "q_period_scales": pd.read_parquet(model_root / "q_period_scales.parquet"),
        "copula_correlation": np.asarray(correlation_payload["matrix"], dtype=float),
        "copula_components": tuple(correlation_payload["components"]),
        "weather_scenarios": _read_json(weather_root / "typical_weather_scenarios.json"),
        "typical_day_weights": _read_json(weather_root / "typical_day_weights.json"),
    }
    rank_template = model_root / "anonymous_residual_rank_template.csv"
    if rank_template.is_file():
        artifacts["residual_rank_template"] = pd.read_csv(rank_template)
    validate_public_artifact_safety(artifacts)
    return artifacts


def validate_public_artifact_safety(artifacts: dict[str, Any]) -> None:
    root = Path(artifacts["artifact_root"])
    for path in root.rglob("*"):
        if path.is_file():
            lowered = path.name.lower()
            if any(token in lowered for token in FORBIDDEN_ARTIFACT_TOKENS):
                raise ValueError(f"forbidden deployment artifact name: {path.name}")
            if path.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
                raise ValueError(f"executable serialization is not allowed: {path.name}")

    for name, value in artifacts.items():
        if isinstance(value, pd.DataFrame):
            lowered_columns = [str(column).lower() for column in value.columns]
            unsafe = [
                column
                for column in lowered_columns
                if any(token in column for token in FORBIDDEN_ARTIFACT_TOKENS)
            ]
            if unsafe:
                raise ValueError(f"private columns in public artifact {name}: {unsafe}")

    draws = artifacts["parameter_draws"]
    required_draw_columns = {
        "parameter_draw_id",
        "component",
        "period",
        "predictor_center",
        "coefficient",
        "pooled_mean",
        "sparse_period_offset",
    }
    if not required_draw_columns.issubset(draws.columns):
        raise ValueError("parameter draw artifact has an incompatible schema")
    if draws["parameter_draw_id"].nunique() != 50:
        raise ValueError("the frozen online model requires exactly 50 parameter draws")

    correlation = np.asarray(artifacts["copula_correlation"], dtype=float)
    if correlation.shape != (4, 4):
        raise ValueError("Copula correlation must be 4x4")
    if not np.allclose(correlation, correlation.T, atol=1e-10):
        raise ValueError("Copula correlation must be symmetric")
    if np.linalg.eigvalsh(correlation).min() <= 0:
        raise ValueError("Copula correlation must be positive definite")
    if "residual_rank_template" in artifacts:
        from .e4_runtime import _validate_template

        _validate_template(artifacts["residual_rank_template"])
