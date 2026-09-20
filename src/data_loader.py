import configparser
import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "src" / "config.ini"


def _load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    # read() expects file paths, not file objects
    read_files = config.read(CONFIG_PATH, encoding="utf-8-sig")
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    return config


_config = _load_config()

RAW_DATA_PATH = (BASE_DIR / _config["paths"]["raw_data_path"]).resolve()
TARGET_COL = _config["pipeline"]["target_col"]


def load_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        _download()
    df = pd.read_csv(RAW_DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    logger.info(
        f"Loaded {df.shape[0]} rows x {df.shape[1]} cols | "
        f"churn rate: {df[TARGET_COL].eq('Yes').mean():.2%}"
    )
    return df


def _download():
    import kagglehub

    path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))
    csv_files = list(path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {path}")

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_files[0], RAW_DATA_PATH)
    logger.info(f"Downloaded dataset to {RAW_DATA_PATH}")