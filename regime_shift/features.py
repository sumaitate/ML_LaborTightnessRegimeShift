from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import LAG_LIST, REGIME_DATES, ROLL_LIST, TARGET_HORIZONS, TARGET_PREFIX_MAP
except ImportError:
    from config import LAG_LIST, REGIME_DATES, ROLL_LIST, TARGET_HORIZONS, TARGET_PREFIX_MAP


def safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.where(series > 0))


def _as_month_start(series: pd.Series) -> pd.Series:
    date_data = pd.to_datetime(series, errors="coerce")
    return date_data.dt.to_period("M").dt.to_timestamp(how="start")


def _coerce_numeric(df: pd.DataFrame, col_list: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in col_list:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def target_feature_map() -> Dict[str, str]:
    return TARGET_PREFIX_MAP.copy()


def set_regime(
    df: pd.DataFrame,
    shock_start: str = REGIME_DATES["shock_start"],
    shock_end: str = REGIME_DATES["shock_end"],
    post_start: str = REGIME_DATES["post_start"],
) -> pd.DataFrame:
    result = df.copy()

    if "date" not in result.columns:
        raise ValueError("Date column is missing.")

    result["date"] = _as_month_start(result["date"])

    shock_start_ts = pd.Timestamp(shock_start)
    shock_end_ts = pd.Timestamp(shock_end)
    post_start_ts = pd.Timestamp(post_start)

    result["pre_regime"] = (result["date"] < shock_start_ts).astype(int)
    result["shock_regime"] = ((result["date"] >= shock_start_ts) & (result["date"] <= shock_end_ts)).astype(int)
    result["post_regime"] = (result["date"] >= post_start_ts).astype(int)

    result["regime"] = "Pre-2020"
    result.loc[result["shock_regime"].eq(1), "regime"] = "Pandemic Shock (2020-2021H1)"
    result.loc[result["post_regime"].eq(1), "regime"] = "Post-Pandemic (2021-06+)"

    result["split_role"] = "pre_train"
    result.loc[result["shock_regime"].eq(1), "split_role"] = "pandemic_test"
    result.loc[result["post_regime"].eq(1), "split_role"] = "post_test"

    return result


def add_base_feature(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    numeric_cols = [
        "job_openings_level",
        "unemployment_level",
        "unemployment_rate",
        "ahe_private",
        "eci_total",
        "cpi_all",
        "pce_price",
    ]
    result = _coerce_numeric(result, numeric_cols)

    if {"job_openings_level", "unemployment_level"}.issubset(result.columns):
        denom = result["unemployment_level"].where(result["unemployment_level"] > 0)
        result["jolts_ratio"] = result["job_openings_level"] / denom
        result["log_jolts_ratio"] = safe_log(result["jolts_ratio"])

    if "unemployment_rate" in result.columns:
        result["log_unemployment_rate"] = safe_log(result["unemployment_rate"])

    if "ahe_private" in result.columns:
        result["log_ahe_private"] = safe_log(result["ahe_private"])

    if "eci_total" in result.columns:
        result["log_eci_total"] = safe_log(result["eci_total"])

    if "cpi_all" in result.columns:
        result["log_cpi_all"] = safe_log(result["cpi_all"])

    if "pce_price" in result.columns:
        result["log_pce_price"] = safe_log(result["pce_price"])

    return result


def add_rate_feature(
    df: pd.DataFrame,
    window: int = 12,
    suffix: str = "pct",
) -> pd.DataFrame:
    result = df.copy()

    rate_map = {
        "log_ahe_private": f"ahe_private_{window}m_{suffix}",
        "log_eci_total": f"eci_total_{window}m_{suffix}",
        "log_cpi_all": f"cpi_all_{window}m_{suffix}",
        "log_pce_price": f"pce_price_{window}m_{suffix}",
    }

    for base_col, new_col in rate_map.items():
        if base_col in result.columns:
            result[new_col] = result[base_col].diff(window) * 100.0

    return result


def add_target_feature(
    df: pd.DataFrame,
    horizon_list: Optional[List[int]] = None,
    prefix_map: Optional[Dict[str, str]] = None,
    include_contemporaneous: bool = False,
    include_legacy_price_alias: bool = True,
) -> pd.DataFrame:
    result = df.copy()
    horizon_list = horizon_list or TARGET_HORIZONS
    prefix_map = prefix_map or target_feature_map()

    for base_col, prefix in prefix_map.items():
        if base_col not in result.columns:
            continue

        if include_contemporaneous:
            result[f"{prefix}_0"] = result[base_col]

        for horizon in horizon_list:
            result[f"{prefix}_{horizon}"] = result[base_col].shift(-horizon)

    if include_legacy_price_alias:
        for horizon in horizon_list:
            cpi_col = f"cpi_target_{horizon}"
            if cpi_col in result.columns:
                result[f"price_target_{horizon}"] = result[cpi_col]

    return result


def add_lag_feature(
    df: pd.DataFrame,
    col_list: List[str],
    lag_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    result = df.copy()
    lag_list = lag_list or LAG_LIST

    for col in col_list:
        if col not in result.columns:
            continue
        for lag in lag_list:
            result[f"{col}_lag_{lag}"] = result[col].shift(lag)

    return result


def add_change_feature(
    df: pd.DataFrame,
    col_list: List[str],
    step_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    result = df.copy()
    step_list = step_list or LAG_LIST

    for col in col_list:
        if col not in result.columns:
            continue
        for step in step_list:
            result[f"{col}_chg_{step}"] = result[col].diff(step)

    return result


def add_roll_feature(
    df: pd.DataFrame,
    col_list: List[str],
    win_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    result = df.copy()
    win_list = win_list or ROLL_LIST

    for col in col_list:
        if col not in result.columns:
            continue
        for win in win_list:
            result[f"{col}_mean_{win}"] = result[col].rolling(win).mean()
            result[f"{col}_std_{win}"] = result[col].rolling(win).std()

    return result


def add_persistence_feature(
    df: pd.DataFrame,
    horizon_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    result = df.copy()
    horizon_list = horizon_list or TARGET_HORIZONS

    base_candidates = [
        "ahe_private_12m_pct",
        "cpi_all_12m_pct",
        "pce_price_12m_pct",
    ]

    for col in base_candidates:
        if col not in result.columns:
            continue

        result[f"{col}_lag_1"] = result[col].shift(1)

        for horizon in horizon_list:
            result[f"{col}_lag_{horizon}"] = result[col].shift(horizon)

    return result


def add_state_feature(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "regime" not in result.columns:
        result = set_regime(result)

    pre_data = result.loc[result["regime"].eq("Pre-2020")].copy()

    if "cpi_all_12m_pct" in result.columns:
        ref = pre_data["cpi_all_12m_pct"].dropna()
        high_cut = ref.quantile(0.75) if not ref.empty else result["cpi_all_12m_pct"].quantile(0.75)
        result["high_inflation"] = (result["cpi_all_12m_pct"] >= high_cut).astype(int)

    if "unemployment_rate" in result.columns:
        ref = pre_data["unemployment_rate"].dropna()
        low_cut = ref.quantile(0.25) if not ref.empty else result["unemployment_rate"].quantile(0.25)
        result["tight_labor"] = (result["unemployment_rate"] <= low_cut).astype(int)

    if "hy_oas" in result.columns:
        ref = pre_data["hy_oas"].dropna()
        high_cut = ref.quantile(0.75) if not ref.empty else result["hy_oas"].quantile(0.75)
        result["credit_stress"] = (result["hy_oas"] >= high_cut).astype(int)

    return result


def add_interact_feature(
    df: pd.DataFrame,
    col_list: List[str],
    include_post_regime: bool = False,
) -> pd.DataFrame:
    result = df.copy()

    if "regime" not in result.columns:
        result = set_regime(result)

    state_cols = [col for col in ["high_inflation", "tight_labor", "credit_stress"] if col in result.columns]
    if include_post_regime and "post_regime" in result.columns:
        state_cols = state_cols + ["post_regime"]

    for col in col_list:
        if col not in result.columns:
            continue
        for state_col in state_cols:
            result[f"{col}_x_{state_col}"] = result[col] * result[state_col]

    if {"jolts_ratio", "unemployment_rate"}.issubset(result.columns):
        result["jolts_x_unemployment"] = result["jolts_ratio"] * result["unemployment_rate"]

    if {"log_jolts_ratio", "quits_rate"}.issubset(result.columns):
        result["log_jolts_x_quits"] = result["log_jolts_ratio"] * result["quits_rate"]

    return result


def add_feature_pipeline(
    df: pd.DataFrame,
    lag_cols: Optional[List[str]] = None,
    change_cols: Optional[List[str]] = None,
    roll_cols: Optional[List[str]] = None,
    interact_cols: Optional[List[str]] = None,
    include_post_regime_interactions: bool = False,
) -> pd.DataFrame:
    result = df.copy()
    result = set_regime(result)
    result = add_base_feature(result)
    result = add_rate_feature(result)
    result = add_target_feature(result)
    result = add_persistence_feature(result)
    result = add_state_feature(result)

    lag_cols = lag_cols or [
        "log_jolts_ratio",
        "jolts_ratio",
        "unemployment_rate",
        "quits_rate",
        "fed_funds",
        "consumer_sentiment",
        "hy_oas",
        "ahe_private_12m_pct",
        "cpi_all_12m_pct",
        "pce_price_12m_pct",
    ]
    change_cols = change_cols or [
        "log_jolts_ratio",
        "jolts_ratio",
        "unemployment_rate",
        "quits_rate",
        "fed_funds",
        "consumer_sentiment",
        "hy_oas",
    ]
    roll_cols = roll_cols or [
        "log_jolts_ratio",
        "jolts_ratio",
        "ahe_private_12m_pct",
        "cpi_all_12m_pct",
        "pce_price_12m_pct",
    ]
    interact_cols = interact_cols or [
        "log_jolts_ratio",
        "jolts_ratio",
        "quits_rate",
        "unemployment_rate",
    ]

    result = add_lag_feature(result, lag_cols)
    result = add_change_feature(result, change_cols)
    result = add_roll_feature(result, roll_cols)
    result = add_interact_feature(
        result,
        col_list=interact_cols,
        include_post_regime=include_post_regime_interactions,
    )

    return result


def drop_sparse(
    df: pd.DataFrame,
    max_pct: float = 60.0,
    keep_list: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    result = df.copy()
    keep_list = keep_list or []

    miss_pct = 100 * result.isna().mean()
    drop_list = [col for col in result.columns if miss_pct[col] > max_pct and col not in keep_list]
    result = result.drop(columns=drop_list, errors="ignore")

    return result, drop_list


def fill_feature(
    df: pd.DataFrame,
    limit: int = 3,
    skip_list: Optional[List[str]] = None,
    skip_target_cols: bool = True,
) -> pd.DataFrame:
    result = df.copy()
    skip_list = skip_list or []

    num_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    fill_cols = [col for col in num_cols if col not in skip_list]

    if skip_target_cols:
        fill_cols = [col for col in fill_cols if "target_" not in col]

    if fill_cols:
        result[fill_cols] = result[fill_cols].ffill(limit=limit)

    return result


def choose_core(df: pd.DataFrame) -> Dict[str, List[str]]:
    core_map = {
        "tightness": [
            "jolts_ratio",
            "log_jolts_ratio",
            "unemployment_rate",
            "quits_rate",
        ],
        "target": [
            "ahe_private_12m_pct",
            "cpi_all_12m_pct",
            "pce_price_12m_pct",
        ],
        "control": [
            "fed_funds",
            "consumer_sentiment",
            "hy_oas",
        ],
        "state": [
            "high_inflation",
            "tight_labor",
            "credit_stress",
        ],
    }

    out_map: Dict[str, List[str]] = {}
    for key, col_list in core_map.items():
        out_map[key] = [col for col in col_list if col in df.columns]

    return out_map


def choose_candidate(df: pd.DataFrame, core_map: Dict[str, List[str]]) -> List[str]:
    target_base = core_map.get("target", [])
    tightness_base = core_map.get("tightness", [])
    control_base = core_map.get("control", [])
    state_base = core_map.get("state", [])

    keep_cols = set(tightness_base + control_base + state_base)

    lag_base = tightness_base + target_base + control_base
    chg_base = tightness_base + control_base
    roll_base = tightness_base + target_base
    interact_base = ["log_jolts_ratio", "jolts_ratio", "quits_rate", "unemployment_rate"]

    for col in lag_base:
        for lag in [1, 3, 6, 12]:
            name = f"{col}_lag_{lag}"
            if name in df.columns:
                keep_cols.add(name)

    for col in chg_base:
        for step in [1, 3, 6, 12]:
            name = f"{col}_chg_{step}"
            if name in df.columns:
                keep_cols.add(name)

    for col in roll_base:
        for win in [3, 6, 12, 24]:
            mean_col = f"{col}_mean_{win}"
            std_col = f"{col}_std_{win}"
            if mean_col in df.columns:
                keep_cols.add(mean_col)
            if std_col in df.columns:
                keep_cols.add(std_col)

    for col in interact_base:
        for state_col in ["high_inflation", "tight_labor", "credit_stress", "post_regime"]:
            name = f"{col}_x_{state_col}"
            if name in df.columns:
                keep_cols.add(name)

    for col in ["jolts_x_unemployment", "log_jolts_x_quits"]:
        if col in df.columns:
            keep_cols.add(col)

    return sorted(keep_cols)


def compute_vif(df: pd.DataFrame, col_list: List[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    use_cols = [col for col in col_list if col in df.columns]
    data = df[use_cols].select_dtypes(include=[np.number]).dropna().copy()

    if data.shape[0] < 5 or data.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif"])

    data = data.loc[:, data.nunique(dropna=True) > 1]

    if data.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif"])

    x_data = pd.concat([pd.Series(1.0, index=data.index, name="const"), data], axis=1)

    vif_rows = []
    for idx, col in enumerate(x_data.columns):
        if col == "const":
            continue
        try:
            value = variance_inflation_factor(x_data.values, idx)
        except Exception:
            value = np.nan
        vif_rows.append({"feature": col, "vif": value})

    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)


def reduce_feature(
    df: pd.DataFrame,
    col_list: List[str],
    max_corr: float = 0.95,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    use_cols = [col for col in col_list if col in df.columns]
    data = df[use_cols].select_dtypes(include=[np.number]).copy()

    if data.empty:
        empty_pair = pd.DataFrame(columns=["left_col", "right_col", "correlation"])
        empty_vif = pd.DataFrame(columns=["feature", "vif"])
        return [], empty_pair, empty_vif

    keep_rank = {
        "log_jolts_ratio": 1,
        "jolts_ratio": 2,
        "unemployment_rate": 3,
        "quits_rate": 4,
        "fed_funds": 5,
        "consumer_sentiment": 6,
        "hy_oas": 7,
        "ahe_private_12m_pct": 8,
        "cpi_all_12m_pct": 9,
        "pce_price_12m_pct": 10,
        "high_inflation": 11,
        "tight_labor": 12,
        "credit_stress": 13,
        "post_regime": 14,
    }

    data = data.loc[:, data.nunique(dropna=True) > 1]
    if data.empty:
        empty_pair = pd.DataFrame(columns=["left_col", "right_col", "correlation"])
        empty_vif = pd.DataFrame(columns=["feature", "vif"])
        return [], empty_pair, empty_vif

    corr = data.corr().abs()
    drop_set = set()
    pair_rows = []

    cols = corr.columns.tolist()

    for left_idx in range(len(cols)):
        for right_idx in range(left_idx + 1, len(cols)):
            left_col = cols[left_idx]
            right_col = cols[right_idx]
            value = corr.iloc[left_idx, right_idx]

            if pd.isna(value) or value < max_corr:
                continue

            pair_rows.append(
                {
                    "left_col": left_col,
                    "right_col": right_col,
                    "correlation": float(value),
                }
            )

            if left_col in drop_set or right_col in drop_set:
                continue

            left_rank = keep_rank.get(left_col, 1000)
            right_rank = keep_rank.get(right_col, 1000)

            if left_rank < right_rank:
                drop_set.add(right_col)
            elif right_rank < left_rank:
                drop_set.add(left_col)
            else:
                left_miss = df[left_col].isna().mean()
                right_miss = df[right_col].isna().mean()
                if left_miss <= right_miss:
                    drop_set.add(right_col)
                else:
                    drop_set.add(left_col)

    keep_cols = [col for col in data.columns if col not in drop_set]
    vif_data = compute_vif(df, keep_cols)

    if pair_rows:
        pair_data = pd.DataFrame(pair_rows).sort_values("correlation", ascending=False).reset_index(drop=True)
    else:
        pair_data = pd.DataFrame(columns=["left_col", "right_col", "correlation"])

    return keep_cols, pair_data, vif_data


def leak_check(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, object]:
    leak_tags = ["target_", "_lead_", "future_", "actual_", "observed_", "_forward"]
    safe_tags = ["lag_", "_chg_", "_std_", "_mean_"]

    leak_cols = []

    for col in feature_cols:
        low_col = col.lower()
        if any(tag in low_col for tag in safe_tags):
            continue
        if any(tag in low_col for tag in leak_tags):
            leak_cols.append(col)

    return {
        "leak_cols": sorted(leak_cols),
        "has_leak": len(leak_cols) > 0,
    }


def feature_diagnostic(
    df: pd.DataFrame,
    feature_cols: List[str],
    regime_col: str = "regime",
) -> pd.DataFrame:
    rows = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        row = {
            "feature": col,
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(100 * df[col].isna().mean(), 2),
            "mean_all": df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
            "std_all": df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else np.nan,
        }

        if regime_col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            for regime_name in ["Pre-2020", "Pandemic Shock (2020-2021H1)", "Post-Pandemic (2021-06+)"]:
                reg_mask = df[regime_col] == regime_name
                reg_data = df.loc[reg_mask, col]
                row[f"mean_{regime_name}"] = reg_data.mean()
                row[f"std_{regime_name}"] = reg_data.std()

        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "missing_count",
                "missing_pct",
                "mean_all",
                "std_all",
            ]
        )

    return pd.DataFrame(rows).sort_values(["missing_pct", "feature"]).reset_index(drop=True)


def standardize_feature(
    df: pd.DataFrame,
    col_list: List[str],
    regime_col: str = "regime",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    result = df.copy()
    param_map: Dict[str, Dict[str, float]] = {}

    if regime_col not in result.columns:
        return result, param_map

    for col in col_list:
        if col not in result.columns:
            continue

        std_col = f"{col}_scale"
        result[std_col] = np.nan

        for regime_name in result[regime_col].dropna().unique():
            reg_mask = result[regime_col] == regime_name
            mean_value = result.loc[reg_mask, col].mean()
            std_value = result.loc[reg_mask, col].std()

            if pd.isna(std_value) or std_value == 0:
                std_value = 1.0

            key = f"{regime_name}_{col}"
            param_map[key] = {
                "mean": float(mean_value) if pd.notna(mean_value) else np.nan,
                "std": float(std_value),
            }

            result.loc[reg_mask, std_col] = (result.loc[reg_mask, col] - mean_value) / std_value

    return result, param_map


def build_model_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    id_cols: Optional[List[str]] = None,
    drop_target_na: bool = False,
    drop_scaled_cols: bool = True,
) -> pd.DataFrame:
    id_cols = id_cols or [
        "date",
        "regime",
        "split_role",
        "pre_regime",
        "shock_regime",
        "post_regime",
        "high_inflation",
        "tight_labor",
        "credit_stress",
    ]

    keep_ids = [col for col in id_cols if col in df.columns]
    keep_features = [col for col in feature_cols if col in df.columns]
    keep_targets = [col for col in target_cols if col in df.columns]

    leak_data = leak_check(df, keep_features)
    keep_features = [col for col in keep_features if col not in leak_data["leak_cols"]]

    if drop_scaled_cols:
        keep_features = [col for col in keep_features if not col.endswith("_scale")]

    ordered_cols: List[str] = []
    for col in keep_ids + keep_features + keep_targets:
        if col not in ordered_cols:
            ordered_cols.append(col)

    model_data = df.loc[:, ordered_cols].copy()
    model_data = model_data.loc[:, ~model_data.columns.duplicated()].copy()

    if drop_target_na and keep_targets:
        model_data = model_data.dropna(subset=keep_targets, how="all")

    core_need = [col for col in ["log_jolts_ratio", "unemployment_rate", "quits_rate"] if col in model_data.columns]
    if core_need:
        model_data = model_data.dropna(subset=core_need, how="any")

    return model_data.reset_index(drop=True)


def split_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    id_cols = id_cols or [
        "date",
        "regime",
        "split_role",
        "pre_regime",
        "shock_regime",
        "post_regime",
        "high_inflation",
        "tight_labor",
        "credit_stress",
    ]

    use_ids = [col for col in id_cols if col in df.columns]
    use_features = [col for col in feature_cols if col in df.columns]

    ordered_cols: List[str] = []
    for col in use_ids + use_features + [target_col]:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    result = df.loc[:, ordered_cols].copy()
    result = result.loc[:, ~result.columns.duplicated()].copy()
    result = result.dropna(subset=[target_col], how="any").reset_index(drop=True)

    return result