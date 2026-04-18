import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def clean_year(col: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, col)
    if not match:
        return None

    year = int(match.group(1))
    return 2000 + year if year < 20 else 1900 + year


def process_vintage(
    df: pd.DataFrame,
    col_pattern: str,
    prefix: str,
    keep_last: int = 5,
) -> Tuple[Optional[pd.DataFrame], List[int]]:
    work = df.copy()
    first_col = work.columns[0]
    work = work.rename(columns={first_col: "date_text"})

    work["date"] = pd.to_datetime(
        work["date_text"].astype(str).str.replace(":", "-", regex=False) + "-01",
        format="%Y-%m-%d",
        errors="coerce",
    )

    work = work.drop(columns=["date_text"])
    work = work.dropna(subset=["date"])
    work = work.sort_values("date").reset_index(drop=True)

    year_map: Dict[int, List[str]] = {}
    for col in work.columns:
        if col == "date":
            continue
        year = clean_year(col, col_pattern)
        if year is None:
            continue
        year_map.setdefault(year, []).append(col)

    if not year_map:
        logger.warning("No vintage columns were found.")
        return None, []

    keep_years = sorted(year_map.keys())[-keep_last:]
    keep_cols = ["date"] + [col for year in keep_years for col in year_map[year]]
    work = work[keep_cols].copy()

    rename_map = {col: f"{prefix}_{col}" for col in work.columns if col != "date"}
    work = work.rename(columns=rename_map)

    return work, keep_years


def load_file(
    file_path: Path,
    file_type: str = "csv",
    index_col: Optional[str] = None,
    **kwargs,
) -> Optional[pd.DataFrame]:
    if not file_path.exists():
        print("File couldn't be found, try again.")
        return None

    try:
        if file_type == "csv":
            df = pd.read_csv(file_path, index_col=index_col, **kwargs)
        elif file_type == "excel":
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("File type is not supported.")

        return df
    except Exception:
        print("File couldn't be read, try again.")
        return None


def set_date(
    df: pd.DataFrame,
    date_col: str = "date",
    month_start: bool = True,
) -> pd.DataFrame:
    work = df.copy()

    if date_col not in work.columns:
        raise ValueError("Date column is missing.")

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).copy()

    period = work[date_col].dt.to_period("M")

    if month_start:
        work[date_col] = period.dt.to_timestamp(how="start")
    else:
        work[date_col] = period.dt.to_timestamp(how="end")

    work = work.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    work = work.reset_index(drop=True)

    return work


def check_panel(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    work = set_date(df=df, date_col=date_col)
    full_date = pd.date_range(work[date_col].min(), work[date_col].max(), freq="MS")
    full_df = pd.DataFrame({date_col: full_date})
    merged = full_df.merge(work, on=date_col, how="left")

    return pd.DataFrame(
        {
            "metric": [
                "Rows",
                "Cols",
                "Start Date",
                "End Date",
                "Full Months",
                "Missing Months",
            ],
            "value": [
                len(work),
                work.shape[1],
                str(work[date_col].min().date()),
                str(work[date_col].max().date()),
                len(full_date),
                int(merged.isna().all(axis=1).sum()),
            ],
        }
    )


def merge_data(
    base_df: pd.DataFrame,
    data_map: Dict[str, Optional[pd.DataFrame]],
    date_col: str = "date",
    how: str = "left",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = set_date(base_df, date_col=date_col)
    log_rows = []

    for name, df in data_map.items():
        if df is None or df.empty:
            log_rows.append(
                {
                    "source": name,
                    "status": "skipped",
                    "rows": 0,
                    "cols_added": 0,
                }
            )
            continue

        part = set_date(df, date_col=date_col)

        cols_before = merged.shape[1]
        rows_before = merged.shape[0]

        merged = merged.merge(part, on=date_col, how=how, validate="one_to_one")

        cols_after = merged.shape[1]
        rows_after = merged.shape[0]

        log_rows.append(
            {
                "source": name,
                "status": "ok",
                "rows": rows_after,
                "rows_before": rows_before,
                "cols_added": cols_after - cols_before,
            }
        )

    merged = merged.sort_values(date_col).reset_index(drop=True)
    log_df = pd.DataFrame(log_rows)
    return merged, log_df


def missing_data(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "col": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (100 * df.isna().mean()).round(2).values,
            "non_null_count": df.notna().sum().values,
            "data_type": df.dtypes.astype(str).values,
        }
    )
    return out.sort_values(["missing_pct", "col"], ascending=[False, True]).reset_index(drop=True)


def data_dict(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        series = df[col]
        is_num = pd.api.types.is_numeric_dtype(series)

        rows.append(
            {
                "col": col,
                "data_type": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "min_value": series.min() if is_num else None,
                "max_value": series.max() if is_num else None,
            }
        )

    return pd.DataFrame(rows).sort_values("col").reset_index(drop=True)


def fill_data(
    df: pd.DataFrame,
    method_map: Optional[Dict[str, str]] = None,
    limit_map: Optional[Dict[str, int]] = None,
    default_method: str = "none",
    default_limit: Optional[int] = None,
) -> pd.DataFrame:
    work = df.copy()
    method_map = method_map or {}
    limit_map = limit_map or {}

    for col in work.columns:
        if col == "date":
            continue

        method = method_map.get(col, default_method)
        limit = limit_map.get(col, default_limit)

        if method == "none":
            continue
        if method == "ffill":
            work[col] = work[col].ffill(limit=limit)
        elif method == "bfill":
            work[col] = work[col].bfill(limit=limit)
        elif method == "both":
            work[col] = work[col].ffill(limit=limit).bfill(limit=limit)
        else:
            raise ValueError(f"Fill method is not supported for {col}.")

    return work


def drop_sparse(
    df: pd.DataFrame,
    max_pct: float = 50.0,
    keep_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    keep_cols = keep_cols or []
    miss_pct = 100 * df.isna().mean()
    drop_cols = [col for col in df.columns if miss_pct[col] > max_pct and col not in keep_cols]
    out = df.drop(columns=drop_cols)
    return out, drop_cols


def date_slice(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    work = set_date(df=df, date_col=date_col)

    if start_date is not None:
        work = work[work[date_col] >= pd.Timestamp(start_date)]
    if end_date is not None:
        work = work[work[date_col] <= pd.Timestamp(end_date)]

    return work.reset_index(drop=True)