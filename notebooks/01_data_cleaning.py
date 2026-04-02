#!/usr/bin/env python
# coding: utf-8

# # Post Pandemic Regime Shifts in Labor Market: Data Cleaning and Merge

# ## Purpose

# ## Research Context

# ## Imports and Configuration

# In[16]:


import re
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import openpyxl
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 50)


# In[17]:


repo_root = Path.home() / "Documents" / "Coding" / "ML_LaborTightnessRegimeShift"

data_root = repo_root / "data"
raw_root = data_root / "raw"
fred_root = raw_root / "fred"
external_root = data_root / "external"
processed_root = data_root / "processed"

processed_root.mkdir(parents=True, exist_ok=True)


# In[18]:


fred_panel_path = fred_root / "01_fred_panel.csv"

if not fred_panel_path.exists():
    raise FileNotFoundError(f"FRED panel not found at {fred_panel_path}.")

fred_panel = pd.read_csv(fred_panel_path, index_col="date", parse_dates=True)

print(f"FRED Panel: {fred_panel.shape[0]} monthly observations, {fred_panel.shape[1]} series.")
print(f"Date Range: {fred_panel.index.min().date()} through {fred_panel.index.max().date()}.")

fred_panel_clean = fred_panel.reset_index()
fred_output_path = processed_root / "01_fred_data.csv"
fred_panel_clean.to_csv(fred_output_path, index=False)

print(f"Saved: {fred_output_path}")


# ## Helper Functions

# In[19]:


def load_file_with_check(file_path, file_type="csv", **kwargs):
    """Safely load a file and check existence."""
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return None

    try:
        if file_type == "csv":
            return pd.read_csv(file_path, **kwargs)
        elif file_type == "excel":
            return pd.read_excel(file_path, **kwargs)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None


def clean_vintage_years(col_name, pattern):
    """Extract and convert vintage year from column name."""
    match = re.search(pattern, col_name)
    if not match:
        return None

    year_code = int(match.group(1))
    return 2000 + year_code if year_code < 20 else 1900 + year_code


def process_vintage_data(df, col_pattern, prefix, keep_recent_n=5):
    """Generic function to process vintage data (RUC, CPI, etc.)."""
    # Rename first column to date_str
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date_str"})

    # Convert date string to datetime
    df["date"] = pd.to_datetime(
        df["date_str"].str.replace(":", "-") + "-01",
        format="%Y-%m-%d",
        errors="coerce"
    )
    df = df.drop(columns=["date_str"])
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Extract vintage years
    non_date_cols = [c for c in df.columns if c != "date"]
    vintage_years = {}

    for col in non_date_cols:
        year = clean_vintage_years(col, col_pattern)
        if year:
            if year not in vintage_years:
                vintage_years[year] = []
            vintage_years[year].append(col)

    if not vintage_years:
        print(f"❌ No vintage year columns found using pattern: {col_pattern}")
        return None

    # Keep only recent vintage years
    recent_years = sorted(vintage_years.keys())[-keep_recent_n:]
    cols_to_keep = ["date"] + [col for year in recent_years for col in vintage_years[year]]
    df = df[cols_to_keep].copy()

    # Rename columns with prefix
    cols_to_rename = {c: f"{prefix}_{c}" for c in df.columns if c != "date"}
    df = df.rename(columns=cols_to_rename)

    return df, recent_years



# In[20]:


fred_path = fred_root / "01_fred_panel.csv"
fred_data = load_file_with_check(fred_path, file_type="csv", index_col="date", parse_dates=True)

if fred_data is not None:
    print(f"✓ Loaded: {fred_data.shape[0]} monthly observations, {fred_data.shape[1]} series")
    print(f"  Date Range: {fred_data.index.min().date()} through {fred_data.index.max().date()}")

    fred_merge = fred_data.reset_index()
    fred_output = processed_root / "01_fred_data.csv"
    fred_merge.to_csv(fred_output, index=False)
    print(f"✓ Saved: {fred_output}")
else:
    fred_merge = None
    print("❌ Failed to load FRED data")


# In[21]:


atlanta_path = external_root / "wage-growth-data.xlsx"
atlanta_merge = None

if atlanta_path.exists():
    try:
        atlanta_data = pd.read_excel(atlanta_path, sheet_name="data_overall", engine="openpyxl", skiprows=2)
        print(f"✓ Loaded: {atlanta_data.shape[0]} rows, {atlanta_data.shape[1]} columns")

        # Find date column
        date_candidates = []
        for col in atlanta_data.columns:
            try:
                test_dates = pd.to_datetime(atlanta_data[col], errors="coerce")
                non_null_pct = test_dates.notna().sum() / len(test_dates)
                if non_null_pct > 0.8:
                    date_candidates.append((col, non_null_pct))
            except:
                pass

        if date_candidates:
            date_col = max(date_candidates, key=lambda x: x[1])[0]
            atlanta_data["date"] = pd.to_datetime(atlanta_data[date_col], errors="coerce")

            if date_col != "date":
                atlanta_data = atlanta_data.drop(columns=[date_col])

            atlanta_data = atlanta_data.dropna(subset=["date"])
            atlanta_data = atlanta_data.sort_values("date").reset_index(drop=True)

            # Convert numeric columns
            for col in atlanta_data.columns:
                if col != "date":
                    atlanta_data[col] = pd.to_numeric(atlanta_data[col].replace(".", np.nan), errors="coerce")

            print(f"✓ Cleaned: {atlanta_data.shape[0]} rows with valid dates")
            print(f"  Date Range: {atlanta_data['date'].min().date()} to {atlanta_data['date'].max().date()}")

            # Prepare for merge
            atlanta_merge = atlanta_data.copy()
            cols_to_rename = {c: f"atl_{c}" for c in atlanta_merge.columns if c != "date"}
            atlanta_merge = atlanta_merge.rename(columns=cols_to_rename)
            atlanta_merge = atlanta_merge.sort_values("date").reset_index(drop=True)

            atlanta_output = processed_root / "02_atlanta_data.csv"
            atlanta_merge.to_csv(atlanta_output, index=False)
            print(f"✓ Saved: {atlanta_output}")
        else:
            print("⚠️  Could not identify date column")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print(f"⚠️  File not found: {atlanta_path}")


# In[22]:


philly_ruc_path = external_root / "philly_ruc_vintages.xlsx"
philly_ruc_merge = None

if philly_ruc_path.exists():
    try:
        ruc_df = pd.read_excel(philly_ruc_path, sheet_name="ruc", engine="openpyxl")
        print(f"✓ Loaded: {ruc_df.shape[0]} rows, {ruc_df.shape[1]} columns")

        ruc_df, recent_years = process_vintage_data(ruc_df, r'RUC(\d{2})Q\d', "phl_ruc")

        if ruc_df is not None:
            print(f"✓ Cleaned: {ruc_df.shape[0]} rows with valid dates")
            print(f"  Selected {ruc_df.shape[1] - 1} columns from vintages: {recent_years}")
            print(f"  Date Range: {ruc_df['date'].min().date()} to {ruc_df['date'].max().date()}")

            philly_ruc_merge = ruc_df
            ruc_output = processed_root / "03_philly_ruc_data.csv"
            philly_ruc_merge.to_csv(ruc_output, index=False)
            print(f"✓ Saved: {ruc_output}")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print(f"⚠️  File not found: {philly_ruc_path}")


# In[23]:


philly_cpi_path = external_root / "philly_cpi_vintages.xlsx"
philly_cpi_merge = None

if philly_cpi_path.exists():
    try:
        cpi_df = pd.read_excel(philly_cpi_path, sheet_name="cpi", engine="openpyxl")
        print(f"✓ Loaded: {cpi_df.shape[0]} rows, {cpi_df.shape[1]} columns")

        cpi_df, recent_years = process_vintage_data(cpi_df, r'CPI(\d{2})Q\d', "phl_cpi")

        if cpi_df is not None:
            print(f"✓ Cleaned: {cpi_df.shape[0]} rows with valid dates")
            print(f"  Selected {cpi_df.shape[1] - 1} columns from vintages: {recent_years}")
            print(f"  Date Range: {cpi_df['date'].min().date()} to {cpi_df['date'].max().date()}")

            philly_cpi_merge = cpi_df
            cpi_output = processed_root / "04_philly_cpi_data.csv"
            philly_cpi_merge.to_csv(cpi_output, index=False)
            print(f"✓ Saved: {cpi_output}")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print(f"⚠️  File not found: {philly_cpi_path}")


# In[24]:


cpi_5yr_path = external_root / "Mean_CPI5YR_Level.xlsx"
cpi_5yr_merge = None

if cpi_5yr_path.exists():
    try:
        cpi_5yr_df = pd.read_excel(cpi_5yr_path, sheet_name="Mean_Level", engine="openpyxl")
        print(f"✓ Loaded: {cpi_5yr_df.shape[0]} rows, {cpi_5yr_df.shape[1]} columns")

        # Convert quarter to month
        month_from_quarter = pd.to_numeric(cpi_5yr_df["QUARTER"] * 3 - 2, errors="coerce").fillna(1).astype(int)

        cpi_5yr_df["date"] = pd.to_datetime(
            cpi_5yr_df["YEAR"].astype(str) + "-" + month_from_quarter.astype(str) + "-01",
            format="%Y-%m-%d",
            errors="coerce"
        )

        cpi_5yr_df = cpi_5yr_df[["date", "CPI5YR"]].copy()
        cpi_5yr_df = cpi_5yr_df.rename(columns={"CPI5YR": "exp_5yr_cpi"})
        cpi_5yr_df = cpi_5yr_df.dropna(subset=["date"])
        cpi_5yr_df = cpi_5yr_df.drop_duplicates(subset=["date"]).reset_index(drop=True)
        cpi_5yr_df = cpi_5yr_df.sort_values("date").reset_index(drop=True)

        print(f"✓ Cleaned: {cpi_5yr_df.shape[0]} unique quarters")
        print(f"  Date Range: {cpi_5yr_df['date'].min().date()} to {cpi_5yr_df['date'].max().date()}")
        print(f"  Note: Quarterly data (will be sparse in monthly dataset)")

        cpi_5yr_merge = cpi_5yr_df
        cpi_5yr_output = processed_root / "05_cpi_5yr_expectations_data.csv"
        cpi_5yr_merge.to_csv(cpi_5yr_output, index=False)
        print(f"✓ Saved: {cpi_5yr_output}")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print(f"⚠️  File not found: {cpi_5yr_path}")


# In[25]:


if fred_merge is None:
    print("❌ FRED data is required to proceed. Exiting.")
    exit()

merged_data = fred_merge.copy()
print(f"Starting with FRED: {merged_data.shape[0]} rows, {merged_data.shape[1]} columns")


# In[26]:


merge_sequence = []

# Merge Atlanta
if atlanta_merge is not None:
    before = merged_data.shape[1]
    merged_data = merged_data.merge(atlanta_merge, on="date", how="left")
    cols_added = merged_data.shape[1] - before
    print(f"✓ Merged Atlanta Fed: +{cols_added} columns → {merged_data.shape[1]} total")
    merge_sequence.append({"source": "atlanta", "cols_added": cols_added, "total_cols": merged_data.shape[1]})

# Merge Philadelphia RUC
if philly_ruc_merge is not None:
    before = merged_data.shape[1]
    merged_data = merged_data.merge(philly_ruc_merge, on="date", how="left")
    cols_added = merged_data.shape[1] - before
    print(f"✓ Merged Philadelphia RUC: +{cols_added} columns → {merged_data.shape[1]} total")
    merge_sequence.append({"source": "philly_ruc", "cols_added": cols_added, "total_cols": merged_data.shape[1]})

# Merge Philadelphia CPI
if philly_cpi_merge is not None:
    before = merged_data.shape[1]
    merged_data = merged_data.merge(philly_cpi_merge, on="date", how="left")
    cols_added = merged_data.shape[1] - before
    print(f"✓ Merged Philadelphia CPI: +{cols_added} columns → {merged_data.shape[1]} total")
    merge_sequence.append({"source": "philly_cpi", "cols_added": cols_added, "total_cols": merged_data.shape[1]})

# Merge CPI 5-Year Expectations
if cpi_5yr_merge is not None:
    before = merged_data.shape[1]
    merged_data = merged_data.merge(cpi_5yr_merge, on="date", how="left")
    cols_added = merged_data.shape[1] - before
    print(f"✓ Merged CPI 5-Year Expectations: +{cols_added} columns → {merged_data.shape[1]} total")
    merge_sequence.append({"source": "cpi_5yr_expectations", "cols_added": cols_added, "total_cols": merged_data.shape[1]})


# In[27]:


merged_data = merged_data.sort_values("date").reset_index(drop=True)
merged_data["date"] = pd.to_datetime(merged_data["date"])

print(f"\n✓ Final dataset: {merged_data.shape[0]} rows, {merged_data.shape[1]} columns")
print(f"Date Range: {merged_data['date'].min().date()} to {merged_data['date'].max().date()}")
print(f"Memory: {merged_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")


# In[28]:


missing_analysis = pd.DataFrame({
    "column": merged_data.columns,
    "missing_count": merged_data.isna().sum(),
    "missing_pct": (merged_data.isna().sum() / len(merged_data) * 100).round(2),
    "non_null_count": merged_data.notna().sum(),
}).sort_values("missing_count", ascending=False)

print("\nTop Columns with Missing Data:")
print(missing_analysis.head(20).to_string(index=False))

complete_cases = merged_data.dropna(how="any").shape[0]
complete_pct = (complete_cases / len(merged_data)) * 100

print(f"\nComplete Cases: {complete_cases} ({complete_pct:.2f}%)")


# In[29]:


periods = [
    ("Pre-2010", merged_data[merged_data['date'] < '2010-01-01']),
    ("2010-2019", merged_data[(merged_data['date'] >= '2010-01-01') & (merged_data['date'] < '2020-01-01')]),
    ("2020+", merged_data[merged_data['date'] >= '2020-01-01'])
]

for period_name, period_data in periods:
    if len(period_data) > 0:
        complete = period_data.dropna(how='any').shape[0]
        pct = (complete / len(period_data) * 100)
        print(f"  {period_name}: {complete} of {len(period_data)} ({pct:.1f}%)")


# In[30]:


merge_log_df = pd.DataFrame(merge_sequence)
merge_log_path = processed_root / "06_merge_log.csv"
merge_log_df.to_csv(merge_log_path, index=False)

print(f"Saved Merge Log Path: {merge_log_path}")


# In[31]:


missing_analysis_path = processed_root / "06_missingness_analysis.csv"
missing_analysis.to_csv(missing_analysis_path, index=False)

print(f"Saved Missing Analysis Path: {missing_analysis_path}")


# In[32]:


merged_csv_path = processed_root / "07_merged_data.csv"
merged_data.to_csv(merged_csv_path, index=False)

print(f"Saved Merge CSV Path: {merged_csv_path}")

merged_parquet_path = processed_root / "07_merged_data.parquet"
merged_data.to_parquet(merged_parquet_path, index=False, compression="snappy")

print(f"Saved Merge Parquet Path: {merged_parquet_path}")


# In[33]:


dict_rows = []

for col in merged_data.columns:
    col_data = merged_data[col]
    dict_rows.append({
        "column_name": col,
        "data_type": str(col_data.dtype),
        "non_null_count": col_data.notna().sum(),
        "null_count": col_data.isna().sum(),
        "unique_values": col_data.nunique(),
    })

data_dictionary = pd.DataFrame(dict_rows).sort_values("column_name").reset_index(drop=True)
data_dict_path = processed_root / "07_data_dictionary.csv"
data_dictionary.to_csv(data_dict_path, index=False)

print(f"Saved Dictionary Path: {data_dict_path}")


# In[34]:


summary_metrics = pd.DataFrame([
    {"metric": "fred_rows", "value": fred_merge.shape[0]},
    {"metric": "fred_cols", "value": fred_merge.shape[1]},
    {"metric": "atlanta_rows", "value": atlanta_merge.shape[0] if atlanta_merge is not None else 0},
    {"metric": "atlanta_cols", "value": atlanta_merge.shape[1] if atlanta_merge is not None else 0},
    {"metric": "philly_ruc_rows", "value": philly_ruc_merge.shape[0] if philly_ruc_merge is not None else 0},
    {"metric": "philly_ruc_cols", "value": philly_ruc_merge.shape[1] if philly_ruc_merge is not None else 0},
    {"metric": "philly_cpi_rows", "value": philly_cpi_merge.shape[0] if philly_cpi_merge is not None else 0},
    {"metric": "philly_cpi_cols", "value": philly_cpi_merge.shape[1] if philly_cpi_merge is not None else 0},
    {"metric": "cpi_5yr_rows", "value": cpi_5yr_merge.shape[0] if cpi_5yr_merge is not None else 0},
    {"metric": "cpi_5yr_cols", "value": cpi_5yr_merge.shape[1] if cpi_5yr_merge is not None else 0},
    {"metric": "final_rows", "value": merged_data.shape[0]},
    {"metric": "final_cols", "value": merged_data.shape[1]},
    {"metric": "date_range_start", "value": str(merged_data['date'].min().date())},
    {"metric": "date_range_end", "value": str(merged_data['date'].max().date())},
    {"metric": "complete_cases", "value": complete_cases},
    {"metric": "complete_cases_pct", "value": round(complete_pct, 2)},
])

summary_path = processed_root / "07_merge_summary.csv"
summary_metrics.to_csv(summary_path, index=False)
print(f"Saved Summary Path: {summary_path}")


# In[ ]:




