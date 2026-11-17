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

FRED_API = "https://api.stlouisfed.org/fred/"


def fred_get(endpoint: str, **params) -> dict:
    """GET one FRED API endpoint as JSON. The key is read from .env and never
    printed: errors report FRED's message, not the request URL."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("Set FRED_API_KEY in .env")
    query = urllib.parse.urlencode({**params, "api_key": key, "file_type": "json"})
    try:
        with urllib.request.urlopen(f"{FRED_API}{endpoint}?{query}", timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            message = json.load(e).get("error_message", "")
        except ValueError:
            message = ""
        raise RuntimeError(f"FRED {endpoint} {params}: HTTP {e.code} {message}") from None


def fred_release(series_id: str) -> dict:
    """The FRED release that publishes a series: {'id': ..., 'name': ...}."""
    releases = fred_get("series/release", series_id=series_id)["releases"]
    if len(releases) != 1:
        raise RuntimeError(f"{series_id}: expected one release, got {releases}")
    return releases[0]


def fred_release_dates(release_id: int) -> list[str]:
    """Every historical date of a release, all vintages, dates with data only."""
    dates, offset = [], 0
    while True:
        page = fred_get(
            "release/dates", release_id=release_id, realtime_start="1776-07-04",
            realtime_end="9999-12-31", sort_order="asc", limit=10000, offset=offset,
            include_release_dates_with_no_data="false",
        )
        dates += [d["date"] for d in page["release_dates"]]
        offset += len(page["release_dates"])
        if offset >= page["count"] or not page["release_dates"]:
            return dates


def fred_first_release_dates(series_id: str) -> set[pd.Timestamp]:
    """Dates on which a series first published a new observation (ALFRED,
    output_type 4 = initial release only)."""
    obs = fred_get("series/observations", series_id=series_id, output_type=4,
                   realtime_start="1776-07-04", realtime_end="9999-12-31")["observations"]
    if not obs:
        raise RuntimeError(f"{series_id}: no initial-release observations")
    return set(pd.to_datetime([o["realtime_start"] for o in obs]))


def pull_release_dates(cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Historical release dates for each event with an anchor_series.

    FRED's release calendar also lists revision-only dates (CPI seasonal
    factors, annual benchmarks, corrections). new_observation marks dates
    on which the anchor series published a new observation; build_calendar
    keeps only those.

    Output: raw/fred/release_dates.parquet (event, release_date, new_observation)
    """
    path = raw_path("fred", "release_dates.parquet")
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    frames = []
    for event, spec in cfg["events"].items():
        if "anchor_series" not in spec:
            continue
        release = fred_release(spec["anchor_series"])
        dates = fred_release_dates(release["id"])
        if not dates:
            raise RuntimeError(f"{event}: release {release['id']} returned no dates")
        print(f"  {event}: {spec['anchor_series']} -> release {release['id']} "
              f"'{release['name']}', {len(dates)} dates {dates[0]} to {dates[-1]}")
        first = fred_first_release_dates(spec["anchor_series"])
        df = pd.DataFrame({"event": event, "release_date": pd.to_datetime(dates)})
        df["new_observation"] = df["release_date"].isin(first)
        missing = first - set(df["release_date"])
        if missing:
            print(f"    note: {len(missing)} first-release dates of {spec['anchor_series']} "
                  f"not on the release calendar, e.g. {sorted(missing)[-3:]}")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(["event", "release_date"]).any():
        raise RuntimeError("duplicate event, release_date in FRED release dates")
    out.to_parquet(path, index=False)
    return out


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

    Returns None when no export is present, so the pipeline runs without
    Bloomberg. Reading the exports is step 7.
    """
    folder = data_dir() / "raw" / "bloomberg"
    if not folder.exists() or not any(folder.glob("*.csv")):
        return None
    raise NotImplementedError("Bloomberg exports found; reader is built in step 7")
