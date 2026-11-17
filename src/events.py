"""Event calendar and surprises."""

import pandas as pd


def build_calendar(cfg: dict, bloomberg: pd.DataFrame) -> pd.DataFrame:
    """One row per release.

    Columns: event, family, date, time_et, actual, consensus, dispersion.
    Uses first-release values only.
    Output: processed/events.parquet
    """
    raise NotImplementedError


def add_surprises(events: pd.DataFrame) -> pd.DataFrame:
    """Add the standardized surprise: (actual - consensus) / dispersion."""
    raise NotImplementedError


def flag_overlaps(events: pd.DataFrame) -> pd.DataFrame:
    """Flag dates with more than one release so they can be handled explicitly."""
    raise NotImplementedError
