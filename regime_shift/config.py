from pathlib import Path
from typing import Dict, List


def find_project_root(start_path: Path) -> Path:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    for path in [current, *current.parents]:
        if (
            (path / "pyproject.toml").exists()
            and (path / "data").exists()
            and (path / "notebooks").exists()
        ):
            return path

    raise RuntimeError(
        f"Could not find project root from {start_path}. "
        "Expected a folder containing pyproject.toml, data/, and notebooks/."
    )


PROJECT_ROOT = find_project_root(Path(__file__))
SRC_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
EXT_ROOT = DATA_ROOT / "external"
INT_ROOT = DATA_ROOT / "interim"
PROC_ROOT = DATA_ROOT / "processed"

MODEL_ROOT = PROJECT_ROOT / "models"
REPORT_ROOT = PROJECT_ROOT / "reports"
FIG_ROOT = REPORT_ROOT / "figures"
NOTE_ROOT = PROJECT_ROOT / "notebooks"

for root in [
    DATA_ROOT,
    RAW_ROOT,
    EXT_ROOT,
    INT_ROOT,
    PROC_ROOT,
    MODEL_ROOT,
    REPORT_ROOT,
    FIG_ROOT,
]:
    root.mkdir(parents=True, exist_ok=True)

MERGED_PATH = PROC_ROOT / "07_merged_data.csv"
DICT_PATH = PROC_ROOT / "07_data_dictionary.csv"
MERGE_PATH = PROC_ROOT / "07_merge_summary.csv"

EDA_PATH = REPORT_ROOT / "03_eda_summary.csv"
BREAK_PATH = REPORT_ROOT / "03_structural_breaks.csv"
REGIME_PATH = REPORT_ROOT / "03_regime_statistics.csv"
UNIT_PATH = REPORT_ROOT / "03_unit_root_tests.csv"
CURVE_PATH = REPORT_ROOT / "03_phillips_curve_results.csv"
ROLL_PATH = REPORT_ROOT / "03_rolling_correlation_summary.csv"

FEAT_PATH = PROC_ROOT / "08_feature_engineered_data.csv"
MODEL_READY_PATH = PROC_ROOT / "09_model_ready_data.csv"
WAGE_MODEL_READY_PATH = PROC_ROOT / "09_wage_model_ready_data.csv"
PRICE_MODEL_READY_PATH = PROC_ROOT / "09_price_model_ready_data.csv"
INTERACTION_MODEL_READY_PATH = PROC_ROOT / "09_model_interaction_data.csv"
DESIGN_MODEL_READY_PATH = PROC_ROOT / "09_model_design_data.csv"

IMPORT_PATH = REPORT_ROOT / "04_feature_importance_summary.csv"
COLINEAR_PATH = REPORT_ROOT / "04_colinearity_check.csv"
VALID_PATH = REPORT_ROOT / "04_feature_validation.csv"

REGIME_DATES: Dict[str, str] = {
    "pre_start": "2000-01-01",
    "pre_end": "2019-12-31",
    "shock_start": "2020-01-01",
    "shock_end": "2021-05-31",
    "post_start": "2021-06-01",
    "post_end": "2026-12-31",
}

BREAK_DATE = "2021-09-01"

PRE_LABEL = "Pre-2020"
SHOCK_LABEL = "Pandemic Shock (2020-2021H1)"
POST_LABEL = "Post-Pandemic (2021-06+)"

REGIME_LABELS = {
    "pre": PRE_LABEL,
    "shock": SHOCK_LABEL,
    "post": POST_LABEL,
}

LABOR_VARS: List[str] = [
    "job_openings_level",
    "unemployment_rate",
    "unemployment_level",
    "hires_rate",
    "quits_rate",
    "layoffs_discharges_rate",
    "total_separations_rate",
    "prime_age_urate",
    "u6_rate",
    "avg_weeks_unemployed",
    "lfpr",
    "epop_ratio",
]

WAGE_VARS: List[str] = [
    "ahe_private",
    "awe_private",
    "ahe_manufacturing",
    "eci_total",
    "eci_wages",
]

PRICE_VARS: List[str] = [
    "cpi_all",
    "cpi_core",
    "cpi_shelter",
    "cpi_less_shelter",
    "pce_price",
    "pce_core",
    "pce_trimmed_12m",
    "breakeven_5y",
    "forward_5y5y",
    "exp_5yr_cpi",
]

POLICY_VARS: List[str] = [
    "fed_funds",
    "treasury_3m",
    "treasury_2y",
    "treasury_10y",
]

FIN_VARS: List[str] = [
    "aaa_yield",
    "baa_yield",
    "hy_oas",
    "bbb_oas",
    "consumer_sentiment",
    "m2",
]

ACTIVITY_VARS: List[str] = [
    "payrolls_nonfarm",
    "payrolls_private",
    "ip_total",
    "capacity_util",
    "real_retail",
    "housing_starts",
]

MAIN_PRED = "jolts_ratio"

TARGET_HORIZONS: List[int] = [3, 6, 12]

TARGET_PREFIX_MAP: Dict[str, str] = {
    "ahe_private_12m_pct": "wage_target",
    "cpi_all_12m_pct": "cpi_target",
    "pce_price_12m_pct": "pce_target",
}

WAGE_TARGET_COLS = [f"wage_target_{h}" for h in TARGET_HORIZONS]
PRICE_TARGET_COLS = [f"cpi_target_{h}" for h in TARGET_HORIZONS] + [
    f"pce_target_{h}" for h in TARGET_HORIZONS
]
TARGET_COLS = WAGE_TARGET_COLS + PRICE_TARGET_COLS

WAGE_TARGET = "wage_target_12"
PRICE_TARGET = "cpi_target_12"

LAG_LIST: List[int] = [1, 3, 6, 12, 24]
ROLL_LIST: List[int] = [3, 6, 12, 24]
CORR_LIST: List[int] = [0, 6, 12, 24]

ADF_ALPHA = 0.05
ADF_LAG = "AIC"
ADF_REG = "c"

STD_BY_REGIME = True
CORR_MAX = 0.95
MISS_MAX = 0.50
OBS_MIN = 12
FWD_WINDOW = 12

FIG_SIZE = (14, 8)
FIG_WIDE = (16, 10)
FIG_TALL = (12, 10)
SAVE_DPI = 300
PLOT_STYLE = "seaborn-v0_8-darkgrid"

ATL_MAP = {
    "atl_.": "atl_wage_3m_ga",
    "atl_..1": "atl_wage_6m_ga",
    "atl_..2": "atl_wage_12m_ga",
    "atl_..3": "atl_wage_diffusion_3m",
    "atl_..4": "atl_wage_diffusion_6m",
    "atl_..5": "atl_wage_diffusion_12m",
    "atl_..6": "atl_wage_median_3m",
    "atl_..7": "atl_wage_median_6m",
    "atl_..8": "atl_wage_median_12m",
    "atl_..9": "atl_wage_25th_3m",
    "atl_..10": "atl_wage_25th_6m",
    "atl_..11": "atl_wage_25th_12m",
    "atl_..12": "atl_wage_75th_3m",
    "atl_..13": "atl_wage_75th_6m",
    "atl_..14": "atl_wage_75th_12m",
    "atl_..15": "atl_wage_diffusion_levels",
}

PHIL_YEARS = ["15", "16", "17", "18", "19"]

SHOW_DEC = 4
STAT_ALPHA = 0.05
SCORE_NAME = "RMSE"