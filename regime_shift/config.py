
import os
from pathlib import Path
from datetime import datetime

# Paths and Roots
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for directory in [INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MERGED_DATA_PATH = PROCESSED_DATA_DIR / "07_merged_data.csv"
DATA_DICTIONARY_PATH = PROCESSED_DATA_DIR / "07_data_dictionary.csv"
MERGE_SUMMARY_PATH = PROCESSED_DATA_DIR / "07_merge_summary.csv"

EDA_SUMMARY_PATH = REPORTS_DIR / "03_eda_summary.csv"
STRUCTURAL_BREAK_RESULTS_PATH = REPORTS_DIR / "03_structural_breaks.csv"
REGIME_STATISTICS_PATH = REPORTS_DIR / "03_regime_statistics.csv"
UNIT_ROOT_RESULTS_PATH = REPORTS_DIR / "03_unit_root_tests.csv"
PHILLIPS_CURVE_RESULTS_PATH = REPORTS_DIR / "03_phillips_curve_results.csv"
ROLLING_CORRELATION_SUMMARY_PATH = REPORTS_DIR / "03_rolling_correlation_summary.csv"

FEATURE_ENGINEERED_DATA_PATH = PROCESSED_DATA_DIR / "08_feature_engineered_data.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "04_feature_importance_summary.csv"
MULTICOLLINEARITY_REPORT_PATH = REPORTS_DIR / "04_multicollinearity_check.csv"
FEATURE_VALIDATION_PATH = REPORTS_DIR / "04_feature_validation.csv"

REGIME_BREAKS = {
    "pre_2020_start": "2000-01-01",
    "pre_2020_end": "2019-12-31",
    "pandemic_shock_start": "2020-01-01",
    "pandemic_shock_end": "2020-12-31",
    "post_pandemic_start": "2021-01-01",
    "post_pandemic_end": "2026-12-31",
}

QUANDT_BREAKPOINT = "2021-09-01"

LABOR_MARKET_VARIABLES = [
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

WAGE_VARIABLES = [
    "ahe_private",
    "awe_private",
    "ahe_manufacturing",
    "eci_total",
    "eci_wages",
]

INFLATION_VARIABLES = [
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

MONETARY_POLICY_VARIABLES = [
    "fed_funds",
    "treasury_3m",
    "treasury_2y",
    "treasury_10y",
]

FINANCIAL_CONDITIONS_VARIABLES = [
    "aaa_yield",
    "baa_yield",
    "hy_oas",
    "bbb_oas",
    "consumer_sentiment",
    "m2",
]

ACTIVITY_VARIABLES = [
    "payrolls_nonfarm",
    "payrolls_private",
    "ip_total",
    "capacity_util",
    "real_retail",
    "housing_starts",
]

PRIMARY_PREDICTOR_NAME = "jolts_unemployment_ratio"
FORWARD_WAGE_TARGET = "ahe_12m_forward"
FORWARD_INFLATION_TARGET = "cpi_12m_forward"

LAG_HORIZONS = [1, 3, 6, 12, 24]
ROLLING_WINDOW_MONTHS = [12, 24]
CORRELATION_LAG_WINDOWS = [0, 6, 12, 24]

ADF_SIGNIFICANCE_LEVEL = 0.05
ADF_MAX_LAG = "AIC"
ADF_REGRESSION_TYPE = "c"  # constant only

CHOW_BREAK_DATE = "2021-09-01"

STANDARDIZE_WITHIN_REGIME = True
REGIME_1_STANDARDIZATION = "pre_2020"
REGIME_2_STANDARDIZATION = "post_2021"

MULTICOLLINEARITY_THRESHOLD = 0.95

MAX_MISSING_PCT = 0.50
MIN_REQUIRED_OBSERVATIONS = 12

FORWARD_WINDOW_MONTHS = 12

FIGSIZE_DEFAULT = (14, 8)
FIGSIZE_WIDE = (16, 10)
FIGSIZE_TALL = (12, 10)
DPI_SAVE = 300
PLOT_STYLE = "seaborn-v0_8-darkgrid" 

ATLANTA_WAGE_TRACKER_COLS = {
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

# Philadelphia Fed vintage data years
PHILADELPHIA_VINTAGE_YEARS = ["15", "16", "17", "18", "19"]

DISPLAY_DECIMALS = 4
STATISTICAL_SIGNIFICANCE_LEVEL = 0.05
FORECAST_EVALUATION_METRIC = "RMSE"

print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")
print(f"Processed Data Save Path: {PROCESSED_DATA_DIR}")
