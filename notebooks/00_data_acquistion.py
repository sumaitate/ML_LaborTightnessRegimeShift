#!/usr/bin/env python
# coding: utf-8

# # Post Pandemic Regime Shifts in Labor Market: Pulling Data from FRED and other sources
# 
# Federal Reserve Economic Data — St. Louis Fed
# Atlanta Fed Wage Growth Tracker
# Philadelphia Fed Real-Time Data Research Set
# hiladelphia Fed Survey of Professional Forecasters
# Dallas Fed Trimmed Mean PCE 
# Kansas City Fed Labor Market Conditions Indicators
# 
# #### External Data Sources
# * https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ruc
# * https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/cpi
# * https://www.philadelphiafed.org/surveys-and-data/data-files
# * https://www.atlantafed.org/research-and-data/data/wage-growth-tracker

# ## Purpose

# ## Research Context

# ## Inputs and Outputs

# ## Imports and Configuration

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')

import re
import time
from pathlib import Path
import requests

import numpy as np
import pandas as pd

from fredapi import Fred

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)


# In[20]:


repo_root = Path.home() / "Documents" / "Coding" / "ML_LaborTightnessRegimeShift"
# repo_root = Path(r"C:\Users\sumai\Documents\ML_LaborTightnessRegimeShift")

data_root = repo_root / "data"
raw_root  = data_root / "raw"
fred_root    = raw_root  / "fred"
external_root = raw_root / "external_sources"

fred_root.mkdir(parents=True, exist_ok=True)
external_root.mkdir(parents=True, exist_ok=True)


# In[21]:


API_KEY = """
 insert the key here
"""

api_key = re.sub(r"\s+", "", API_KEY)

fred = Fred(api_key=api_key)

start_date = "2000-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")


# In[22]:


series_map = {
    # job market
    "job_openings_level":         "JTSJOL",
    "hires_rate":                 "JTSHIR",
    "hires_level":                "JTSHIL",
    "quits_rate":                 "JTSQUR",
    "quits_level":                "JTSQUL",
    "total_separations_rate":     "JTSTSR",
    "total_separations_level":    "JTSTSL",
    "layoffs_discharges_level":   "JTSLDL",
    "layoffs_discharges_rate":    "JTSLDR",

    # labor force and unemployment 
    "unemployment_rate":          "UNRATE",
    "unemployment_level":         "UNEMPLOY",
    "u6_rate":                    "U6RATE",
    "prime_age_lfpr":             "LNS11300060",
    "lfpr":                       "CIVPART",
    "epop_ratio":                 "EMRATIO",
    "prime_age_urate":            "LNS13025703",
    "avg_weeks_unemployed":       "UEMPMEAN",

    # payrolls
    "payrolls_nonfarm":           "PAYEMS",
    "payrolls_private":           "USPRIV",
    "payrolls_manufacturing":     "MANEMP",
    "payrolls_services":          "SRVPRD",
    "payrolls_construction":      "USCONS",

    # krarnings, hours, and compensation
    "ahe_private":                "CES0500000003",
    "awe_private":                "CES0500000011",
    "awh_private":                "AWHAETP",
    "awe_manufacturing":          "CES3000000008",
    "ahe_manufacturing":          "CES3000000003",
    "eci_total":                  "ECIALLCIV",
    "eci_wages":                  "ECIWAG",

    # cpi
    "cpi_all":                    "CPIAUCSL",
    "cpi_core":                   "CPILFESL",
    "cpi_less_shelter":           "CUSR0000SA0L2",
    "cpi_services_less_rent":     "CUUR0000SASL2RS",
    "cpi_services_less_energy":   "CUSR0000SASLE",
    "cpi_shelter":                "CUSR0000SEHA",
    "cpi_medical":                "CPIMEDSL",
    "cpi_food":                   "CPIUFDSL",

    # PCE price
    "pce_price":                  "PCEPI",
    "pce_core":                   "PCEPILFE",
    "pce_trimmed_12m":            "PCETRIM12M159SFRBDAL",
    "pce_trimmed_1m":             "PCETRIM1M158SFRBDAL",
    "pce_services":               "DSERRG3M086SBEA",

    # income and spending 
    "saving_rate":                "PMSAVE",
    "real_dpi":                   "DSPIC96",
    "real_pi":                    "RPI",
    "real_pi_less_transfers":     "W875RX1",
    "retail_advance":             "RSAFS",
    "real_retail":                "RRSFS",

    # production
    "ip_total":                   "INDPRO",
    "ip_manufacturing":           "IPMAN",
    "capacity_util":              "CUMFNS",

    # housing
    "housing_starts":             "HOUST",
    "building_permits":           "PERMIT",
    "new_home_sales":             "HSN1F",

    # orders/consumer
    "consumer_sentiment":         "UMCSENT",
    "durable_orders":             "DGORDER",
    "capex_orders":               "NEWORDER",

    # credit
    "ci_loans":                   "BUSLOANS",
    "bank_credit":                "TOTBKCR",
    "m2":                         "M2SL",

    # rates and stuff
    "treasury_1m":                "DGS1MO",
    "treasury_3m":                "DGS3MO",
    "treasury_2y":                "DGS2",
    "treasury_10y":               "GS10",
    "fed_funds":                  "FEDFUNDS",
    "aaa_yield":                  "AAA",
    "baa_yield":                  "BAA",
    "hy_oas":                     "BAMLH0A0HYM2",
    "bbb_oas":                    "BAMLC0A4CBBB",

    # inflation expectations
    "breakeven_5y":               "T5YIE",
    "breakeven_10y":              "T10YIE",
    "forward_5y5y":               "T5YIFR",
    "michigan_1y":                "MICH",
    "michigan_5y":                "MICH5Y",

    # Kansas fed labor market conditions
    "kcfed_lmci_activity":        "FRBKCLMCILA",
    "kcfed_lmci_momentum":        "FRBKCLMCIM",

    # actual recession indicator
    "recession":                  "USREC",

    # other
    "output_per_hour":            "OPHNFB",
    "unit_labor_costs":           "ULCNFB",
}

print(f"Series Map: {len(series_map)}")


# In[23]:


def pull_fred_series(sid, obs_start, obs_end, max_tries=5, backoff=1.5):
    last_err = None

    for attempt in range(1, max_tries + 1):
        try:
            return fred.get_series(sid, observation_start=obs_start, observation_end=obs_end)
        except Exception as exc:
            last_err = exc
            if attempt < max_tries:
                wait = backoff * attempt
                print(f"  Retry {attempt}/{max_tries} for {sid} — waiting {wait:.1f}s")
                time.sleep(wait)

    raise RuntimeError(f"FRED pulling failed for {sid}: {last_err}")


def pull_fred_meta(sid, max_tries=5, backoff=1.5):

    for attempt in range(1, max_tries + 1):
        try:
            return fred.get_series_info(sid)
        except Exception as exc:
            if attempt < max_tries:
                wait = backoff * attempt
                print(f"  Retry {attempt}/{max_tries} for metadata {sid} — waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                return pd.Series(dtype="object")

    return pd.Series(dtype="object")


# Frequency Harmonization

# In[24]:


def to_monthly(series, freq_code):
    freq_code = str(freq_code).strip().upper()

    if freq_code == "M":
        out = series.copy()
    elif freq_code in {"D", "W", "BW"}:
        out = series.resample("MS").mean()
    elif freq_code == "Q":
        out = series.resample("MS").ffill()
    else:
        out = series.resample("MS").mean()

    out.index = pd.to_datetime(out.index)
    return out.sort_index()


# In[25]:


def classify_missing(series, obs_start):

    if pd.isna(obs_start):
        obs_start = pd.Timestamp.min

    obs_start = pd.to_datetime(obs_start, errors="coerce")
    if pd.isna(obs_start):
        obs_start = pd.Timestamp.min

    pre = series.loc[series.index < obs_start]
    post = series.loc[series.index >= obs_start]

    return int(pre.isna().sum()), int(post.isna().sum())


# In[26]:


series_store = []
meta_store = []
fail_store = []
total = len(series_map)

for idx, (col, sid) in enumerate(series_map.items(), start=1):
    try:
        print(f"[{idx:>2}/{total}]  {col}")

        raw = pull_fred_series(sid, start_date, end_date)

        if raw.empty:
            fail_store.append({"col": col, "series_id": sid, "error": "empty_series"})
            print(f"  Warning: {col} returned empty")
            continue

        info = pull_fred_meta(sid)
        freq_code = info.get("frequency_short", "")

        monthly = to_monthly(pd.Series(raw, name=col), freq_code)
        monthly.index = pd.to_datetime(monthly.index)
        monthly.name = col

        series_store.append(monthly)

        obs_start = info.get("observation_start", "")
        struct_miss, active_miss = classify_missing(monthly, obs_start)

        meta_store.append({
            "col": col,
            "series_id": sid,
            "title": info.get("title", ""),
            "freq": str(freq_code).strip().upper(),
            "units": info.get("units", ""),
            "seasonal_adj": info.get("seasonal_adjustment", ""),
            "last_updated": info.get("last_updated", ""),
            "obs_start": obs_start,
            "obs_end": info.get("observation_end", ""),
            "raw_count": int(raw.notna().sum()),
            "monthly_count": int(monthly.notna().sum()),
            "struct_miss": struct_miss,
            "active_miss": active_miss,
        })

        time.sleep(0.35)

    except Exception as exc:
        fail_store.append({"col": col, "series_id": sid, "error": str(exc)})
        print(f"  Error: {col} — {exc}")

if not series_store:
    raise RuntimeError("No FRED downloads worked, recheck API key.")

print(f"\nFRED download complete: {len(series_store)} has succeeded, {len(fail_store)} failed.\n")


# In[35]:


fred_panel = (
    pd.concat(series_store, axis=1, sort=False)
    .sort_index()
    .loc[start_date:end_date]
    .copy()
)

fred_panel.index.name = "date"

fred_meta = pd.DataFrame(meta_store).sort_values("col").reset_index(drop=True)
fred_failed = pd.DataFrame(fail_store)

print(f"FRED Panel: {fred_panel.shape[0]} rows, {fred_panel.shape[1]} columns.")
print(f"Date Range: {fred_panel.index.min().date()} through {fred_panel.index.max().date()}.")


# In[36]:


fred_missing = (
    fred_panel.isna()
    .sum()
    .sort_values(ascending=False)
    .rename("miss_count")
    .to_frame()
)

fred_missing["miss_pct"] = fred_missing["miss_count"] / len(fred_panel)

if not fred_meta.empty:
    fred_missing = (
        fred_missing
        .merge(
            fred_meta[["col", "obs_start", "obs_end", "struct_miss", "active_miss"]],
            left_index=True,
            right_on="col",
            how="left",
        )
        .set_index("col")
    )


# In[37]:


fred_panel.to_csv(fred_root / "01_fred_panel.csv")
fred_meta.to_csv(fred_root / "01_fred_metadata.csv", index=False)
fred_failed.to_csv(fred_root / "01_fred_failed.csv", index=False)
fred_missing.to_csv(fred_root / "01_fred_missing.csv")

print(f"FRED outputs saved to: {fred_root}")
print(f"01_fred_panel.csv — {fred_panel.shape[0]} rows, {fred_panel.shape[1]} columns")
print(f"01_fred_metadata.csv — {len(fred_meta)} series")
print(f"01_fred_failed.csv — {len(fred_failed)} failed")
print(f"01_fred_missing.csv — missingness summary\n")


# In[38]:


external_urls = {
    "atlanta_wage_tracker": "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/datafiles/chcs/wage-growth-tracker/wage-growth-data.xlsx",
    "philly_ruc_vintages": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/rucQvMd.xlsx",
    "philly_cpi_vintages": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/cpiQvMd.xlsx",
    "philly_spf_inflation": "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/Inflation.xlsx",
}


# In[39]:


def fetch_excel(url, session=None, max_tries=5, timeout=120):

    if session is None:
        session = requests.Session()

    headers = {"User-Agent": "Firefox/5.0"}

    for attempt in range(max_tries):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)

            if resp.status_code != 200:
                if attempt < max_tries - 1:
                    time.sleep(2 * (attempt + 1))
                continue

            content_type = resp.headers.get("Content-Type", "").lower()

            if "excel" in content_type or "spreadsheet" in content_type or url.endswith(".xlsx"):
                return resp.content

            raise RuntimeError(f"Invalid content type: {content_type}")

        except Exception as exc:

            if attempt == max_tries - 1:
                raise RuntimeError(f"Failed to download {url}: {exc}")
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Max retries exceeded for {url}")


# In[40]:


download_log = []
downloaded = {}
session = requests.Session()

print(f"Downloading {len(external_urls)} external files.\n")

for name, url in external_urls.items():
    dest = external_root / f"{name}.xlsx"

    try:
        print(f"  {name}...", end=" ")
        content = fetch_excel(url, session=session)
        dest.write_bytes(content)
        downloaded[name] = dest

        size_kb = dest.stat().st_size / 1024
        download_log.append({
            "file": name,
            "url": url,
            "path": str(dest),
            "status": "ok",
            "bytes": dest.stat().st_size,
        })
        print(f"OK ({size_kb:.1f} KB)")

    except Exception as exc:
        download_log.append({
            "file": name,
            "url": url,
            "path": str(dest),
            "status": "failed",
            "bytes": np.nan,
            "error": str(exc),
        })
        print(f"Failed: {exc}")

download_log_df = pd.DataFrame(download_log)
download_log_path = external_root / "download_log.csv"
download_log_df.to_csv(download_log_path, index=False)

failed_ext = download_log_df.loc[download_log_df["status"] != "ok"]

if not failed_ext.empty:
    print(f"\nWarning: {len(failed_ext)} external downloads failed.")
    print(f"See {download_log_path} for details.\n")
else:
    ok_count = int((download_log_df["status"] == "ok").sum())
    print(f"\nAll {ok_count} external files downloaded properly.\n")


# In[41]:


ext_tables = {}

try:
    ext_tables["atl_overall"] = pd.read_excel(
        downloaded["atlanta_wage_tracker"],
        sheet_name="data_overall",
        engine="openpyxl"
    )

except Exception as e:
    print(f"Couldn't parse atlanta_wage_tracker 'data_overall': {e}")
    ext_tables["atl_overall"] = pd.DataFrame()

try:
    ext_tables["atl_12ma"] = pd.read_excel(
        downloaded["atlanta_wage_tracker"],
        sheet_name="Overall 12ma",
        engine="openpyxl"
    )

except Exception as e:
    print(f"Couldn't parse atlanta_wage_tracker 'Overall 12ma': {e}")
    ext_tables["atl_12ma"] = pd.DataFrame()

try:
    ext_tables["phl_ruc"] = pd.read_excel(
        downloaded["philly_ruc_vintages"],
        sheet_name="ruc",
        engine="openpyxl"
    )

except Exception as e:
    print(f"Couldn't parse philly_ruc_vintages: {e}")
    ext_tables["phl_ruc"] = pd.DataFrame()

try:
    ext_tables["phl_cpi"] = pd.read_excel(
        downloaded["philly_cpi_vintages"],
        sheet_name="cpi",
        engine="openpyxl"
    )

except Exception as e:
    print(f"Couldn't parse philly_cpi_vintages: {e}")
    ext_tables["phl_cpi"] = pd.DataFrame()

try:
    ext_tables["spf_inflation"] = pd.read_excel(
        downloaded["philly_spf_inflation"],
        sheet_name="INFLATION",
        engine="openpyxl"
    )

except Exception as e:
    print(f"Couldn't parse philly_spf_inflation: {e}")
    ext_tables["spf_inflation"] = pd.DataFrame()


# In[42]:


key_raw_dir = external_root / "key_raw_tables"
key_raw_dir.mkdir(parents=True, exist_ok=True)

ext_manifest = []

for name, df in ext_tables.items():
    if df.empty:
        continue

    dest = key_raw_dir / f"{name}.csv"
    df.to_csv(dest, index=False)

    ext_manifest.append({
        "table": name,
        "path": str(dest),
        "rows": len(df),
        "cols": df.shape[1],
    })

ext_manifest_df = pd.DataFrame(ext_manifest)
ext_manifest_path = external_root / "external_manifest.csv"
ext_manifest_df.to_csv(ext_manifest_path, index=False)

print(f"External Table Path: {key_raw_dir}")
print(f"External Manifest: {ext_manifest_path}\n")


# In[43]:


required_cols = [

    # V/U ratio comps

    "job_openings_level",
    "unemployment_level",
    "unemployment_rate",

    # job related
    "hires_rate",
    "quits_rate",

    # wage growth
    "ahe_private",
    "eci_total",

    # cpi
    "cpi_all",
    "cpi_core",
    "pce_price",
    "pce_core",
    "pce_trimmed_12m",
    "pce_services",

    # forecasts/predicitions
    "breakeven_5y",
    "forward_5y5y",
    "michigan_1y",

    # tightness controls
    "kcfed_lmci_activity",

    # labor
    "output_per_hour",
    "unit_labor_costs",

    # money policy
    "fed_funds",

    # recess indicator
    "recession",
]

missing_required = [c for c in required_cols if c not in fred_panel.columns]


# In[44]:


diagnostics = pd.DataFrame([
    {"metric": "fred_rows", "value": len(fred_panel)},
    {"metric": "fred_cols", "value": fred_panel.shape[1]},
    {"metric": "fred_failed", "value": len(fred_failed)},
    {"metric": "external_ok", "value": int((download_log_df["status"] == "ok").sum())},
    {"metric": "external_failed", "value": int((download_log_df["status"] != "ok").sum())},
    {"metric": "required_cols_total", "value": len(required_cols)},
    {"metric": "required_cols_missing", "value": len(missing_required)},
    {"metric": "fred_start", "value": str(fred_panel.index.min().date())},
    {"metric": "fred_end", "value": str(fred_panel.index.max().date())},
])

diagnostics_path = raw_root / "acquisition_diagnostics.csv"
diagnostics.to_csv(diagnostics_path, index=False)

print(f"\nFRED Panel: {len(fred_panel)} monthly observations, {fred_panel.shape[1]} series")
print(f"Date Range: {fred_panel.index.min().date()} to {fred_panel.index.max().date()}")

print(f"FRED Failed: {len(fred_failed)} series")
print(f"External Downloaded: {int((download_log_df['status'] == 'ok').sum())} files")

print(f"Required Columns: {len(required_cols)} total")

if missing_required:
    print(f"\nThe following are missing: {missing_required}")
    print("Correct this before EDA/FE.")
else:
    print(f"\nAll {len(required_cols)} necessary columns are present.")

print(f"\nDiagnostics Save Path: {diagnostics_path}")

display(diagnostics)


# In[ ]:




