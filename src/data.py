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


# What the OptionMetrics library holds (inspected on WRDS, 2026-11-17):
# - optionm has per-year tables opprcd<YYYY> (option quotes) and secprd<YYYY>
#   (security prices), 1996-2025. Data end on 2025-08-29.
# - opprcd: secid, date, symbol, exdate, cp_flag, strike_price, best_bid,
#   best_offer, volume, open_interest, impl_volatility, delta, optionid,
#   am_settlement, expiry_indicator, root (null), ... strike_price is the
#   strike x 1000 (4200000 = 4200).
# - am_settlement = 1: morning-settled standard monthly SPX. 0: close-settled
#   SPXW (expiry_indicator 'w' weekly, 'm' month-end, null on standard dates).
# - secnmd / securd: secid 108105 is ticker SPX, "CBOE S&P 500 INDEX",
#   index_flag 1. Matches config options.optionmetrics_secid.
# - secprd: date, close (index level); 2023-06-01 close 4221.02.

def _wrds_years(db, cfg: dict) -> list[int]:
    """Years in the sample that have both an opprcd and a secprd table."""
    tables = set(db.list_tables("optionm"))
    start = pd.Timestamp(cfg["sample"]["start"]).year
    end = pd.Timestamp(cfg["sample"]["end"]).year if cfg["sample"]["end"] else 9999
    years = [y for y in range(start, min(end, pd.Timestamp.today().year) + 1)
             if f"opprcd{y}" in tables and f"secprd{y}" in tables]
    if not years:
        raise RuntimeError("no OptionMetrics tables for the sample period")
    return years


def _in_sample(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keep = df["date"] >= pd.Timestamp(cfg["sample"]["start"])
    if cfg["sample"]["end"]:
        keep &= df["date"] <= pd.Timestamp(cfg["sample"]["end"])
    return df[keep]


def pull_underlying(cfg: dict, refresh: bool = False, db=None) -> pd.DataFrame:
    """Daily index close from optionm.secprd<year>.

    Output: raw/optionmetrics/underlying.parquet (date, close). Its dates are
    the project's trading calendar.
    """
    path = raw_path("optionmetrics", "underlying.parquet")
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    db = db or wrds_connection()
    secid = cfg["options"]["optionmetrics_secid"]
    frames = []
    for year in _wrds_years(db, cfg):
        df = db.raw_sql(f"select date, close from optionm.secprd{year} where secid = %(secid)s",
                        params={"secid": secid}, date_cols=["date"])
        if df.empty:
            raise RuntimeError(f"secprd{year}: no rows for secid {secid}")
        frames.append(df)
    out = _in_sample(pd.concat(frames, ignore_index=True), cfg)
    out = out.sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].astype("datetime64[ns]")
    if out["date"].duplicated().any():
        raise RuntimeError("underlying: duplicate dates")
    if out["close"].isna().any() or (out["close"] <= 0).any():
        raise RuntimeError("underlying: missing or non-positive closes")
    out.to_parquet(path, index=False)
    return out


OPTION_COLUMNS = ["date", "exdate", "cp_flag", "strike", "best_bid", "best_offer",
                  "volume", "open_interest", "impl_volatility", "delta", "optionid",
                  "am_settlement", "symbol"]

OPTIONS_SQL = """
    select o.date, o.exdate, o.cp_flag, o.strike_price, o.best_bid, o.best_offer,
           o.volume, o.open_interest, o.impl_volatility, o.delta, o.optionid,
           o.am_settlement, o.symbol
    from optionm.opprcd{y} o
    join optionm.secprd{y} p on p.secid = o.secid and p.date = o.date
    where o.secid = %(secid)s
      and o.exdate - o.date between 0 and %(max_days)s
      and abs(o.strike_price / %(scale)s / p.close - 1) <= %(band)s
"""


def _strike_scale(db, secid: int, year: int) -> int:
    """Strike multiplier, checked against the close on one day rather than
    assuming the documented x1000."""
    df = db.raw_sql(f"""
        select o.strike_price / p.close as ratio
        from optionm.opprcd{year} o join optionm.secprd{year} p
          on p.secid = o.secid and p.date = o.date
        where o.secid = %(secid)s and o.date = (select max(date) from optionm.secprd{year}
                                                where secid = %(secid)s)
    """, params={"secid": secid})
    if df.empty:
        raise RuntimeError("strike scale check: no rows")
    ratio = df["ratio"].median()
    scale = 1000 if 500 < ratio < 2000 else 1 if 0.5 < ratio < 2 else None
    if scale is None:
        raise RuntimeError(f"strike scale check: median strike/close {ratio:.3f}")
    print(f"  strike scale: median strike_price/close = {ratio:.1f} -> divide by {scale}")
    return scale


def pull_options(cfg: dict, refresh: bool = False, db=None) -> None:
    """Short-dated S&P 500 index option quotes, one query per year.

    Filtered in SQL to the index secid, 0 to options.max_days_to_expiry days
    to expiry, and strikes within options.strike_band_pct of that day's close.
    Output: raw/optionmetrics/options_<year>.parquet (read with load_options)
    """
    o = cfg["options"]
    db = db or wrds_connection()
    years = _wrds_years(db, cfg)
    scale = None
    for year in years:
        path = raw_path("optionmetrics", f"options_{year}.parquet")
        if path.exists() and not refresh:
            continue
        scale = scale or _strike_scale(db, o["optionmetrics_secid"], years[-1])
        df = db.raw_sql(OPTIONS_SQL.format(y=year), date_cols=["date", "exdate"], params={
            "secid": o["optionmetrics_secid"], "max_days": o["max_days_to_expiry"],
            "band": o["strike_band_pct"], "scale": scale})
        if df.empty:
            raise RuntimeError(f"opprcd{year}: query returned no rows")
        df["strike"] = df.pop("strike_price") / scale
        before = len(df)
        df = _in_sample(df, cfg)
        if len(df) != before:
            print(f"  {year}: sample window removed {before - len(df)} rows")
        for c in ["date", "exdate"]:
            df[c] = df[c].astype("datetime64[ns]")
        df = df[OPTION_COLUMNS].reset_index(drop=True)
        df.to_parquet(path, index=False)
        print(f"  options {year}: {len(df):,} rows")


def load_options(cfg: dict) -> pd.DataFrame:
    """All cached option years in one frame."""
    files = sorted((data_dir() / "raw" / "optionmetrics").glob("options_*.parquet"))
    if not files:
        raise RuntimeError("no options_<year>.parquet files; run the data step")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def validate_options(cfg: dict) -> None:
    """Step 2 acceptance checks."""
    import numpy as np

    und = pd.read_parquet(raw_path("optionmetrics", "underlying.parquet"))
    opt = load_options(cfg)
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  {'PASS' if passed else 'FAIL'}  {name}{': ' + detail if detail else ''}")

    last = und["date"].max()
    print(f"  underlying: {len(und):,} trading dates {und['date'].min().date()} to {last.date()}")

    counts = opt.groupby(opt["date"].dt.year).size()
    years = range(und["date"].min().year, last.year + 1)
    empty = [y for y in years if counts.get(y, 0) == 0]
    print("  option rows per year:")
    for y in years:
        print(f"    {y}: {counts.get(y, 0):>10,}")
    check("no empty year", not empty, f"empty: {empty}" if empty else "")

    dup = opt.duplicated(["date", "optionid"]).sum()
    check("no duplicate date, optionid", dup == 0, f"{dup} duplicates" if dup else "")

    for col in ["date", "exdate", "cp_flag", "strike", "optionid", "am_settlement"]:
        n = opt[col].isna().sum()
        check(f"no nulls in {col}", n == 0, f"{n} nulls" if n else "")
    check("cp_flag in C/P", opt["cp_flag"].isin(["C", "P"]).all())
    check("am_settlement in 0/1", opt["am_settlement"].isin([0, 1]).all())
    orphan = ~opt["date"].isin(und["date"])
    check("every option date has an index close", not orphan.any(),
          f"{orphan.sum()} rows" if orphan.any() else "")

    good = (opt["best_bid"] > 0) & (opt["best_offer"] > opt["best_bid"])
    print(f"  share with best_bid > 0 and best_offer > best_bid: {good.mean():.1%} "
          f"({good.sum():,} of {len(opt):,})")

    print("  ATM quotes on three dates (nearest close-settled expiry after the date, as "
          "traded at entry; strike nearest close):")
    tradable = opt[(opt["am_settlement"] == 0) & good & (opt["exdate"] > opt["date"])]
    quoted = und[und["date"].isin(tradable["date"])].reset_index(drop=True)
    for target in und["date"].iloc[[len(und) // 10, len(und) // 2, len(und) * 9 // 10]]:
        d = quoted.loc[quoted["date"] >= target, "date"].iloc[0]   # first date with quotes
        close = quoted.loc[quoted["date"] == d, "close"].iloc[0]
        q = tradable[tradable["date"] == d]
        q = q[q["exdate"] == q["exdate"].min()]
        both = q.groupby("strike")["cp_flag"].nunique()
        both = both[both == 2].index.to_numpy()
        k = both[np.abs(both - close).argmin()]
        row = q[q["strike"] == k].set_index("cp_flag")
        print(f"    {d.date()}  close {close:.2f}  exdate {q['exdate'].iloc[0].date()}  "
              f"K {k:g}  call {row.at['C', 'best_bid']:.2f}/{row.at['C', 'best_offer']:.2f}  "
              f"put {row.at['P', 'best_bid']:.2f}/{row.at['P', 'best_offer']:.2f}")

    pm_today = opt.loc[(opt["am_settlement"] == 0) & (opt["exdate"] == opt["date"]), "date"].unique()
    share = und["date"].isin(pm_today).groupby(und["date"].dt.year).mean()
    print("  share of trading dates with a close-settled expiry that day:")
    for y, v in share.items():
        print(f"    {y}: {v:.0%}")

    print(f"  last date available: {last.date()}")
    ev_path = processed_path("events.parquet")
    if ev_path.exists():
        ev = pd.read_parquet(ev_path)
        print(f"  events after {last.date()} (outside the sample): {(ev['event_date'] > last).sum()}")
    else:
        print("  events table not built yet (needs FRED key); count of events after "
              "the last date pending")

    if not ok:
        raise AssertionError("options: acceptance checks failed (see FAIL lines)")


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
