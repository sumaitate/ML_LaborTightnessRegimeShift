import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fredapi import Fred


logger = logging.getLogger(__name__)


class FredLoader:
    def __init__(self, api_key: str):
        key = api_key.strip()
        if not key:
            raise ValueError("FRED API key is missing.")
        self.fred = Fred(api_key=key)

    def pull_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        max_try: int = 5,
        backoff: float = 1.5,
    ) -> Optional[pd.Series]:
        last_err = None

        for step in range(1, max_try + 1):
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                )
                if data is None or len(data) == 0:
                    return None

                out = pd.Series(data).copy()
                out.index = pd.to_datetime(out.index, errors="coerce")
                out = out[~out.index.isna()].sort_index()
                out = pd.to_numeric(out, errors="coerce")
                out = out.dropna()

                return out if not out.empty else None

            except Exception as err:
                last_err = err
                if step < max_try:
                    time.sleep(backoff * step)

        logger.error("Series pull failed for %s: %s", series_id, last_err)
        return None

    def pull_meta(self, series_id: str, max_try: int = 3) -> pd.Series:
        last_err = None

        for step in range(1, max_try + 1):
            try:
                info = self.fred.get_series_info(series_id)
                if isinstance(info, pd.Series):
                    return info
                return pd.Series(dtype="object")
            except Exception as err:
                last_err = err
                if step < max_try:
                    time.sleep(1.5 * step)

        logger.warning("Metadata pull failed for %s: %s", series_id, last_err)
        return pd.Series(dtype="object")

    def pull_many(
        self,
        series_map: Dict[str, str],
        start_date: str,
        end_date: str,
        month_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        month_map = month_map or {}
        data_list = []
        meta_list = []
        fail_list = []

        total = len(series_map)

        for idx, (col, series_id) in enumerate(series_map.items(), start=1):
            try:
                logger.info("[%s/%s] Pulling %s", idx, total, col)
                raw = self.pull_series(series_id, start_date, end_date)

                if raw is None or raw.empty:
                    fail_list.append({"col": col, "series_id": series_id, "error": "empty_series"})
                    continue

                info = self.pull_meta(series_id)
                freq = str(info.get("frequency_short", "")).strip().upper()
                units = str(info.get("units", "")).strip().lower()
                month_rule = month_map.get(col, self.pick_rule(freq=freq, units=units, col=col))

                month = self.to_month(raw=raw, freq=freq, rule=month_rule)
                month.name = col
                data_list.append(month)

                meta_list.append(
                    {
                        "col": col,
                        "series_id": series_id,
                        "title": info.get("title", ""),
                        "freq": freq,
                        "units": info.get("units", ""),
                        "seasonal_adj": info.get("seasonal_adjustment", ""),
                        "last_updated": info.get("last_updated", ""),
                        "obs_start": info.get("observation_start", ""),
                        "obs_end": info.get("observation_end", ""),
                        "month_rule": month_rule,
                        "raw_count": int(raw.notna().sum()),
                        "month_count": int(month.notna().sum()),
                    }
                )

                time.sleep(0.25)

            except Exception as err:
                fail_list.append({"col": col, "series_id": series_id, "error": str(err)})
                logger.error("Pull failed for %s: %s", col, err)

        if not data_list:
            raise RuntimeError("No series could be downloaded.")

        panel = pd.concat(data_list, axis=1, sort=False)
        panel = panel.sort_index().loc[start_date:end_date].copy()
        panel.index = pd.to_datetime(panel.index)
        panel.index.name = "date"

        meta_df = pd.DataFrame(meta_list).sort_values("col").reset_index(drop=True)
        fail_df = pd.DataFrame(fail_list).reset_index(drop=True)

        return panel, meta_df, fail_df

    @staticmethod
    def pick_rule(freq: str, units: str, col: str) -> str:
        rate_tags = ["rate", "yield", "oas", "spread", "funds", "sentiment", "breakeven", "forward"]
        stock_tags = ["level", "payroll", "housing", "retail", "m2", "ip_total", "capacity_util"]

        col_low = col.lower()

        if freq in {"Q", "QE", "QS"}:
            return "quarter_fill"
        if freq in {"A", "AS", "AE", "SA"}:
            return "year_fill"
        if any(tag in col_low for tag in rate_tags):
            return "month_mean"
        if any(tag in col_low for tag in stock_tags):
            return "month_last"
        if "percent" in units or "rate" in units:
            return "month_mean"

        return "month_mean"

    @staticmethod
    def to_month(raw: pd.Series, freq: str, rule: str) -> pd.Series:
        data = pd.Series(raw).dropna().copy()
        data.index = pd.to_datetime(data.index, errors="coerce")
        data = data[~data.index.isna()].sort_index()

        if data.empty:
            return pd.Series(dtype="float64")

        if freq in {"M", "ME", "MS"}:
            out = data.resample("MS").last()
        elif rule == "month_last":
            out = data.resample("MS").last()
        elif rule == "quarter_fill":
            out = data.resample("MS").ffill()
        elif rule == "year_fill":
            out = data.resample("MS").ffill()
        else:
            out = data.resample("MS").mean()

        out.index = pd.to_datetime(out.index)
        out = pd.to_numeric(out, errors="coerce")

        return out.sort_index()


class FileLoader:
    @staticmethod
    def fetch_excel(
        url: str,
        session: Optional[requests.Session] = None,
        max_try: int = 5,
        timeout: int = 120,
    ) -> Optional[bytes]:
        session = session or requests.Session()
        headers = {"User-Agent": "Mozilla/5.0"}
        last_err = None

        for step in range(1, max_try + 1):
            try:
                resp = session.get(url, headers=headers, timeout=timeout)

                if resp.status_code != 200:
                    last_err = f"Status {resp.status_code}"
                    if step < max_try:
                        time.sleep(2 * step)
                    continue

                content_type = resp.headers.get("Content-Type", "").lower()
                is_excel = (
                    "excel" in content_type
                    or "spreadsheet" in content_type
                    or url.lower().endswith(".xlsx")
                    or url.lower().endswith(".xls")
                )

                if is_excel:
                    return resp.content

                last_err = f"Wrong file type: {content_type}"
                if step < max_try:
                    time.sleep(2 * step)

            except Exception as err:
                last_err = err
                if step < max_try:
                    time.sleep(2 * step)

        logger.error("Excel download failed for %s: %s", url, last_err)
        return None

    @staticmethod
    def fetch_many(
        url_map: Dict[str, str],
        out_root: Path,
    ) -> Tuple[Dict[str, Path], pd.DataFrame]:
        out_root.mkdir(parents=True, exist_ok=True)
        file_map: Dict[str, Path] = {}
        log_rows = []

        with requests.Session() as session:
            for name, url in url_map.items():
                path = out_root / f"{name}.xlsx"

                try:
                    data = FileLoader.fetch_excel(url=url, session=session)
                    if data is None:
                        log_rows.append(
                            {
                                "file": name,
                                "url": url,
                                "path": str(path),
                                "status": "failed",
                                "size_bytes": np.nan,
                            }
                        )
                        continue

                    path.write_bytes(data)
                    file_map[name] = path

                    log_rows.append(
                        {
                            "file": name,
                            "url": url,
                            "path": str(path),
                            "status": "ok",
                            "size_bytes": path.stat().st_size,
                        }
                    )

                except Exception as err:
                    log_rows.append(
                        {
                            "file": name,
                            "url": url,
                            "path": str(path),
                            "status": "failed",
                            "size_bytes": np.nan,
                            "error": str(err),
                        }
                    )

        log_df = pd.DataFrame(log_rows)
        return file_map, log_df