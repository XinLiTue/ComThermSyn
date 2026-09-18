from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from src.artifact_io import load_deployment_artifacts
from src.deploy_runtime import build_streamlit_output_package, run_deployment_synthesis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one standalone community RCAQ case.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEPLOY_ROOT / "artifacts_candidates" / "task107_dual_e4_v1",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["e4_rank_lhs_v1", "e4_complete_row_v1", "lhs_fast", "full_mc"],
        default="e4_rank_lhs_v1",
    )
    args = parser.parse_args()
    artifacts = load_deployment_artifacts(args.artifact_root)
    artifacts = dict(artifacts)
    artifacts["runtime_config"] = dict(artifacts["runtime_config"])
    artifacts["runtime_config"]["sampling_mode"] = args.sampling_mode
    result = run_deployment_synthesis(pd.read_csv(args.input_csv), artifacts=artifacts)
    manifest = build_streamlit_output_package(result, args.output_dir)
    print(f"selected={manifest['selected_community_id']}")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
