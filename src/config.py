"""
Centralised configuration: all project constants and run-mode settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent
BUILDING_DATA_PATH = DATA_DIR / "saved_rca_results" / "non_extreme_users.parquet"
WEATHER_PATH = DATA_DIR / "weather.csv"
HOUSE_INFO_PATH = DATA_DIR / "HouseInformation.xlsx"
MEASURED_CASE_DATA_PATH = (
    DATA_DIR
    / "DACS-Data"
    / "JupyterCode"
    / "processed_v1"
    / "P2_2024_10_to_2025_03__train_strict_core_weather.parquet"
)
VALIDATION_OUTPUT_DIR = DATA_DIR / "validation_outputs"
STREAMLIT_OUTPUT_DIR = DATA_DIR / "streamlit_demo_outputs"

# ---------------------------------------------------------------------------
# Weather / simulation constants
# ---------------------------------------------------------------------------
STANDARD_YEARS: List[int] = [2024, 2025]
PROFILE_YEAR: int = 2024
DT_SECONDS: int = 900          # 15-minute timestep used in simulation
WEATHER_RESAMPLE_RULE: str = "15min"

# Heating-season thermostat (V3/V4 default)
HEATING_MONTHS: List[int] = [1, 2, 3, 4, 10, 11, 12]
WINTER_SETPOINT: float = 20.0           # °C, main heating-season setpoint
EXTREME_COLD_OUTDOOR_C: float = 2.0    # °C, outdoor threshold for shoulder heating
SHOULDER_SETPOINT: float = 15.0        # °C, setpoint in shoulder/summer extreme cold

# ---------------------------------------------------------------------------
# Building vintage classification
# ---------------------------------------------------------------------------
VINTAGE_BINS: List[int] = [0, 1964, 1974, 1982, 1991, 1999, 3000]
VINTAGE_LABELS: List[str] = [
    "pre1965",
    "1965-1974",
    "1975-1982",
    "1983-1991",
    "1992-1999",
    "post2000",
]

# ---------------------------------------------------------------------------
# Energy label classification
# ---------------------------------------------------------------------------
LABEL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "A++++": (0.0, 25.0),
    "A+++":  (0.0, 50.0),
    "A++":   (0.0, 75.0),
    "A+":    (0.0, 100.0),
    "A":     (0.0, 125.0),
    "B":     (125.0, 160.0),
    "C":     (160.0, 190.0),
    "D":     (190.0, 250.0),
    "E":     (250.0, 290.0),
    "F":     (290.0, 335.0),
    "G":     (335.0, 1e9),
}

LABEL_ORDER: List[str] = [
    "A++++", "A+++", "A++", "A+", "A", "B", "C", "D", "E", "F", "G",
]

LABEL_SCORE: Dict[str, float] = {
    "A++++": 11.0,
    "A+++":  10.0,
    "A++":   9.0,
    "A+":    8.0,
    "A":     7.0,
    "B":     6.0,
    "C":     5.0,
    "D":     4.0,
    "E":     3.0,
    "F":     2.0,
    "G":     1.0,
}

# ---------------------------------------------------------------------------
# RCA / model column names
# ---------------------------------------------------------------------------
RCA_COLS: List[str] = ["R", "C", "A", "Qint"]
PARAM_TARGETS: List[str] = ["R", "C_per_area", "A", "Qint"]

# ---------------------------------------------------------------------------
# V4 community generator constants
# ---------------------------------------------------------------------------
LABEL_OFFICIAL_WEIGHT: Dict[str, float] = {
    "A++++": 1.0, "A+++": 1.0, "A++": 1.0, "A+": 1.0, "A": 1.0,
    "B": 1.0, "C": 1.0,
    "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0,
}

VINTAGE_EXPECTED_LABEL: Dict[str, str] = {
    "pre1965":   "D",
    "1965-1974": "C",
    "1975-1982": "C",
    "1983-1991": "B",
    "1992-1999": "A",
    "post2000":  "A+",
}
VINTAGE_EXPECTED_LABEL_SCORE: Dict[str, float] = {
    vintage: LABEL_SCORE[label]
    for vintage, label in VINTAGE_EXPECTED_LABEL.items()
}

DEFAULT_RENOVATION_LIFT: float = 1.5
RNG_SEED: int = 42
GROUP_FEATURES_CPA: List[str] = ["R", "C_per_area", "A", "Qint"]
GROUP_FEATURES_RAW_C: List[str] = ["R", "C", "A", "Qint"]
LABEL_RANK: Dict[str, int] = {label: idx for idx, label in enumerate(LABEL_ORDER)}

# ---------------------------------------------------------------------------
# Notebook run-mode defaults (can be overridden at runtime)
# ---------------------------------------------------------------------------
ANALYSIS_MODE: str = "fast"

CASE_SIZE_CONFIG: Dict[str, dict] = {
    "debug": {
        "community_sizes": (30,),
        "n_cases_per_size": 1,
        "n_population_candidates": 12,
        "top_k_proxy": 4,
        "top_k_full": 1,
    },
    "fast": {
        "community_sizes": (30,),
        "n_cases_per_size": 3,
        "n_population_candidates": 24,
        "top_k_proxy": 5,
        "top_k_full": 1,
    },
    "full": {
        "community_sizes": (20, 50, 100),
        "n_cases_per_size": 3,
        "n_population_candidates": 48,
        "top_k_proxy": 8,
        "top_k_full": 1,
    },
}

VALIDATION_MODE_CONFIGS: Dict[str, dict] = {
    "debug": {"n_repeats": 2,  "n_population_candidates": 32},
    "fast":  {"n_repeats": 4,  "n_population_candidates": 64},
    "full":  {"n_repeats": 10, "n_population_candidates": 120},
}

ENERGY_PRIORITY_COLS: List[str] = [
    "E_std_annual_per_m2",
    "E_std_full_per_m2",
    "annual_heat_kwh_per_m2",
    "E_proxy_annual_per_m2",
]

TAU_COARSE_GROUPS: Dict[str, str] = {
    "pre1965":   "pre1975",
    "1965-1974": "pre1975",
    "1975-1991": "1975-2005",
    "1992-2005": "1975-2005",
    "2006-2014": "2006plus",
    "2015+":     "2006plus",
}

LABEL_GROUP_ORDER: List[str] = ["A-family", "B", "C", "D-G"]

# ---------------------------------------------------------------------------
# Custom thermostat defaults (Cell 32 / simulation_custom)
# ---------------------------------------------------------------------------
CUSTOM_SUMMER_MONTHS: Tuple[int, ...] = (6, 7, 8)
CUSTOM_DAY_START_HOUR: int = 9
CUSTOM_DAY_END_HOUR: int = 17
CUSTOM_DAY_SETPOINT_C: float = 18.0
CUSTOM_OTHER_SETPOINT_C: float = 20.0
CUSTOM_INITIAL_TEMP_C: float = 19.0


def get_active_case_size_config(mode: Optional[str] = None) -> dict:
    m = mode or ANALYSIS_MODE
    return CASE_SIZE_CONFIG.get(m, CASE_SIZE_CONFIG["fast"]).copy()


def get_validation_mode_config(mode: Optional[str] = None) -> dict:
    m = mode or ANALYSIS_MODE
    return VALIDATION_MODE_CONFIGS.get(m, VALIDATION_MODE_CONFIGS["fast"]).copy()
