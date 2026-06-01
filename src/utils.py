"""
Shared utility helpers used across multiple src modules.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# Private alias kept for internal use inside validation modules
_safe_numeric = safe_numeric


def slug(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "na"


_slug = slug


def infer_dt_hours(times: pd.Series) -> float:
    dt = pd.to_datetime(times, errors="coerce").sort_values().dropna().diff().dt.total_seconds() / 3600.0
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 1.0
    return float(dt.median())


def save_df(df: pd.DataFrame, path: Path, mkdir: bool = True) -> None:
    if mkdir:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_jsonable(obj: Any, path: Path, mkdir: bool = True) -> None:
    import json

    if mkdir:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
