from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

MERGED_DATA = DATA_PROCESSED / "07_merged_data.csv"
MERGED_DATA_PARQUET = DATA_PROCESSED / "07_merged_data.parquet"

