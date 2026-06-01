"""
Validation orchestration helpers: generated pool sampling, result aggregation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.validation_internal import _with_core_columns


def sample_generated_pool(
    case_size_artifacts: Optional[Dict] = None,
    fallback_dfs: Optional[List[Tuple[Optional[pd.DataFrame], str]]] = None,
) -> pd.DataFrame:
    """
    Build a pool of generated buildings from case_size_artifacts or fallback frames.

    Parameters
    ----------
    case_size_artifacts : {(size, case_id): payload} mapping, optional
    fallback_dfs        : [(DataFrame_or_None, source_name), ...] checked in order
    """
    if case_size_artifacts:
        pool_frames = []
        for payload in case_size_artifacts.values():
            gen_df = payload.get("generated_df")
            if gen_df is not None and len(gen_df):
                pool_frames.append(gen_df.copy())
        if pool_frames:
            out = pd.concat(pool_frames, ignore_index=True)
            out = _with_core_columns(out)
            out.attrs["pool_source"] = "case_size_artifacts"
            return out

    if fallback_dfs:
        for fb_df, source_name in fallback_dfs:
            if fb_df is not None and len(fb_df):
                out = _with_core_columns(fb_df.copy())
                out.attrs["pool_source"] = source_name
                return out

    return pd.DataFrame()
