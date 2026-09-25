from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from src.artifact_io import load_deployment_artifacts


PUBLIC_BUNDLE_VERSION = "task107_dual_e4_public_v1_20260924"
DEFAULT_SAMPLING_MODE = "e4_rank_lhs_v1"
DEFAULT_SELECTION_RULE = "whole_community_robust_medoid_v1"
AVAILABLE_SAMPLING_MODES = [
    "e4_rank_lhs_v1",
    "e4_complete_row_v1",
    "lhs_fast",
    "full_mc",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def promote_candidate_bundle(candidate_root: Path, public_root: Path) -> None:
    candidate_root = candidate_root.resolve()
    public_root = public_root.resolve()
    source_manifest_path = candidate_root / "model" / "model_manifest.json"

    # Verify every protected source file and the public-safety contract before copying it.
    load_deployment_artifacts(candidate_root)
    source_manifest = _read_json(source_manifest_path)
    protected_files = [Path(str(item["path"])) for item in source_manifest["files"]]
    required_rank_template = Path("model/anonymous_residual_rank_template.csv")
    if required_rank_template not in protected_files:
        raise ValueError("candidate bundle does not protect the anonymous rank template")

    for relative_path in protected_files:
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe candidate manifest path: {relative_path}")
        source_path = candidate_root / relative_path
        destination_path = public_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    runtime_path = public_root / "model" / "runtime_config.json"
    runtime_config = _read_json(runtime_path)
    runtime_config.update(
        {
            "sampling_mode": DEFAULT_SAMPLING_MODE,
            "selection_rule": DEFAULT_SELECTION_RULE,
            "e4_max_buildings": 50,
            "available_sampling_modes": AVAILABLE_SAMPLING_MODES,
        }
    )
    _write_json(runtime_path, runtime_config)

    public_manifest = dict(source_manifest)
    public_manifest.update(
        {
            "bundle_version": PUBLIC_BUNDLE_VERSION,
            "default_sampling_mode": DEFAULT_SAMPLING_MODE,
            "promoted_from_bundle_version": source_manifest["bundle_version"],
            "promoted_from_manifest_sha256": _sha256(source_manifest_path),
            "files": [
                {"path": path.as_posix(), "sha256": _sha256(public_root / path)}
                for path in protected_files
            ],
        }
    )
    _write_json(public_root / "model" / "model_manifest.json", public_manifest)

    promoted = load_deployment_artifacts(public_root)
    if promoted["runtime_config"]["sampling_mode"] != DEFAULT_SAMPLING_MODE:
        raise ValueError("promoted bundle did not retain the E4 default")
    if "residual_rank_template" not in promoted:
        raise ValueError("promoted bundle is missing the anonymous rank template")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the accepted Task107 bundle to public.")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEPLOY_ROOT / "artifacts_candidates" / "task107_dual_e4_v1",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=DEPLOY_ROOT / "artifacts_public",
    )
    args = parser.parse_args()
    promote_candidate_bundle(args.candidate_root, args.public_root)
    print(args.public_root.resolve() / "model" / "model_manifest.json")


if __name__ == "__main__":
    main()
