"""
src/run_id.py
-------------
Generate the next run ID.

Format:  run_001_2026-05-18_14-32
         run_XXX_YYYY-MM-DD_HH-MM

Usage:
    from src.run_id import next_run_id
    run_id, run_dir = next_run_id()
"""

from datetime import datetime
from pathlib import Path

RESULTS_ROOT = Path("run_results")


def next_run_id() -> tuple[str, Path]:
    """Return (run_id, run_dir) and create the run folder."""
    RESULTS_ROOT.mkdir(exist_ok=True)
    existing = sorted(RESULTS_ROOT.glob("run_[0-9][0-9][0-9]_*"))
    num = len(existing) + 1
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = f"run_{num:03d}_{now}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(exist_ok=True)
    return run_id, run_dir