"""Data access: config, paths, and one pull function per source.

Raw files are written to DATA_DIR/raw/<source>/ exactly as pulled.
Everything built from them goes to DATA_DIR/processed/.
"""

import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def data_dir() -> Path:
    """Shared data folder from .env, or ./data if DATA_DIR is not set."""
    path = Path(os.getenv("DATA_DIR") or ROOT / "data")
    (path / "raw").mkdir(parents=True, exist_ok=True)
    (path / "processed").mkdir(parents=True, exist_ok=True)
    return path


def raw_path(source: str, name: str) -> Path:
    folder = data_dir() / "raw" / source
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def processed_path(name: str) -> Path:
    return data_dir() / "processed" / name


# --- WRDS -------------------------------------------------------------------

def wrds_connection():
    """Connect with the username in .env. Each person uses their own login."""
    import wrds

    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise RuntimeError("Set WRDS_USERNAME in .env")
    return wrds.Connection(wrds_username=username)


def pull_options(cfg: dict) -> pd.DataFrame:
    """Pull end-of-day index option quotes from OptionMetrics.

    Only contracts expiring within a few days of an event are needed, so the
    query should be filtered by the event calendar to keep it small.
    Output: raw/optionmetrics/options.parquet
    """
    # TODO: query optionm.opprcd<year> for cfg["options"]["optionmetrics_secid"]
    raise NotImplementedError


def pull_underlying(cfg: dict) -> pd.DataFrame:
    """Pull daily index closes from OptionMetrics (optionm.secprd).

    Output: raw/optionmetrics/underlying.parquet
    """
    raise NotImplementedError


# --- FRED / ALFRED ----------------------------------------------------------

def pull_fred(cfg: dict) -> pd.DataFrame:
    """Pull every series in cfg["series"], with vintages where available.

    Output: raw/fred/series.parquet (long format: series, date, vintage, value)
    """
    raise NotImplementedError


# --- Bloomberg --------------------------------------------------------------

def read_bloomberg_calendar(cfg: dict) -> pd.DataFrame:
    """Read the calendar exports placed in raw/bloomberg/ by whoever is at the
    terminal.

    Expected columns per event: release date, actual (first release), survey
    median, survey standard deviation.
    """
    raise NotImplementedError
