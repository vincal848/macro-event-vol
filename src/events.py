"""Event calendar and surprises."""

import numpy as np
import pandas as pd
import pandas.tseries.holiday as hol

# Acceptance: releases per full calendar year, as (low, high, tolerance).
# Inside low-high: PASS. Within `tolerance` of the range: WARN, printed but not
# raised (real gaps such as shutdowns, delayed or combined releases). Beyond
# that: FAIL, which points to a structural error (wrong release ID, revision
# dates listed as releases, a broken pull).
MONTHLY_COUNT = (11, 12, 2)
GDP_COUNT = (4, 4, 1)
FOMC_COUNT = (8, 8, 0)     # from config; cancellations are listed there


class NYSEHolidays(hol.AbstractHolidayCalendar):
    """Regular NYSE full-day holidays. Fallback only: misses one-off closures
    (e.g. Hurricane Sandy 2012, national days of mourning), which the trading
    calendar from the underlying table picks up after the WRDS pull."""
    rules = [
        hol.Holiday("New Year's Day", month=1, day=1, observance=hol.sunday_to_monday),
        hol.USMartinLutherKingJr,
        hol.USPresidentsDay,
        hol.GoodFriday,
        hol.USMemorialDay,
        hol.Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                    observance=hol.nearest_workday),
        hol.Holiday("Independence Day", month=7, day=4, observance=hol.nearest_workday),
        hol.USLaborDay,
        hol.USThanksgivingDay,
        hol.Holiday("Christmas", month=12, day=25, observance=hol.nearest_workday),
    ]


def trading_calendar(underlying: pd.DataFrame | None, start, end) -> tuple[pd.DatetimeIndex, str]:
    """Trading dates from the underlying table, or weekdays less regular NYSE
    holidays if it does not exist yet (recompute after the WRDS pull)."""
    if underlying is not None:
        return pd.DatetimeIndex(underlying["date"]).sort_values(), "underlying"
    days = pd.bdate_range(start, end)
    days = days[~days.isin(NYSEHolidays().holidays(start, end))]
    return days, "weekdays less regular NYSE holidays (underlying not pulled yet)"


def _log_drop(label: str, before: int, after: int) -> None:
    print(f"  filter {label}: removed {before - after} rows ({after} left)")


def _advance_estimates(dates: pd.Series) -> pd.Series:
    """First release date after each quarter end."""
    quarter_end = (dates - pd.offsets.QuarterEnd(1)).dt.normalize()
    keep = dates.groupby(quarter_end.dt.to_period("Q")).transform("min") == dates
    return dates[keep]


def build_calendar(cfg: dict, release_dates: pd.DataFrame,
                   underlying: pd.DataFrame | None = None,
                   today: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per release.

    Columns: event, family, event_date (trading date), release_date (as
    published; differs only for non-trading-day releases), entry_date,
    time_et. n_events_on_date is added by flag_overlaps.
    """
    today = (today or pd.Timestamp.today()).normalize()
    frames = []
    for event, spec in cfg["events"].items():
        if event == "fomc":
            dates = pd.Series(pd.to_datetime(cfg["fomc_dates"]))
        else:
            rel = release_dates[release_dates["event"] == event]
            if rel.empty:
                raise RuntimeError(f"{event}: no release dates")
            dates = rel.loc[rel["new_observation"], "release_date"]
            _log_drop(f"{event} revision-only dates (no new observation)", len(rel), len(dates))
            if spec.get("first_release_after_quarter_end"):
                before = len(dates)
                dates = _advance_estimates(dates)
                _log_drop(f"{event} advance estimate only", before, len(dates))
        frames.append(pd.DataFrame({
            "event": event, "family": spec["family"],
            "event_date": pd.to_datetime(dates.to_numpy()), "time_et": spec["time_et"],
        }))
    events = pd.concat(frames, ignore_index=True)

    start = pd.Timestamp(cfg["sample"]["start"])
    end = pd.Timestamp(cfg["sample"]["end"]) if cfg["sample"]["end"] else today
    before = len(events)
    events = events[events["event_date"] >= start]
    _log_drop(f"event_date >= sample start {start.date()}", before, len(events))
    before = len(events)
    events = events[events["event_date"] <= end]
    _log_drop(f"event_date <= {'sample end' if cfg['sample']['end'] else 'today'} {end.date()}",
              before, len(events))

    calendar, source = trading_calendar(underlying, start - pd.Timedelta(days=15),
                                        end + pd.Timedelta(days=15))
    print(f"  trading calendar: {source}")
    if underlying is not None:
        before = len(events)
        events = events[events["event_date"] <= calendar.max()]
        _log_drop(f"event_date <= last date with data {calendar.max().date()} "
                  "(later events are outside the sample)", before, len(events))

    # Releases on non-trading days (e.g. payrolls on Good Friday).
    events = events.rename(columns={"event_date": "release_date"})
    closed = ~events["release_date"].isin(calendar)
    rule = cfg["non_trading_day_releases"]
    if closed.any():
        print(f"  {closed.sum()} releases on non-trading days ({rule}):")
        print("    " + events.loc[closed, ["event", "release_date"]]
              .to_string(index=False).replace("\n", "\n    "))
    if rule == "drop":
        before = len(events)
        events = events[~closed]
        _log_drop("release on non-trading day", before, len(events))
    elif rule != "next_trading_day":
        raise ValueError(f"non_trading_day_releases: unknown rule {rule!r}")
    nxt = calendar.searchsorted(events["release_date"].to_numpy(), side="left")
    if (nxt >= len(calendar)).any():
        raise RuntimeError("release after the last trading date; extend the calendar")
    events = events.assign(event_date=calendar[nxt])

    # Entry: last trading date strictly before the event date.
    pos = calendar.searchsorted(events["event_date"].to_numpy(), side="left") - 1
    if (pos < 0).any():
        raise RuntimeError("event before the first trading date; extend the calendar")
    events = events.assign(entry_date=calendar[pos])

    events = events.sort_values(["event_date", "event"]).reset_index(drop=True)
    if events.duplicated(["event", "event_date"]).any():
        raise RuntimeError("duplicate event, event_date in calendar")
    return events[["event", "family", "event_date", "release_date", "entry_date", "time_et"]]


def add_surprises(events: pd.DataFrame, bloomberg: pd.DataFrame | None) -> pd.DataFrame:
    """Add the standardized surprise: (actual - consensus) / dispersion.

    Skipped without error when there is no Bloomberg export.
    """
    if bloomberg is None:
        print("  no Bloomberg export: surprises skipped")
        return events
    out = events.merge(bloomberg, on=["event", "event_date"], how="left", validate="1:1")
    out["surprise_z"] = (out["actual"] - out["consensus"]) / out["dispersion"].replace(0, np.nan)
    return out


def flag_overlaps(events: pd.DataFrame) -> pd.DataFrame:
    """Number of releases sharing each event date."""
    return events.assign(
        n_events_on_date=events.groupby("event_date")["event"].transform("size"))


# --- acceptance -------------------------------------------------------------

def _full_years(events: pd.DataFrame, cfg: dict, last_date: pd.Timestamp) -> range:
    start = pd.Timestamp(cfg["sample"]["start"])
    first = start.year if start == pd.Timestamp(start.year, 1, 1) else start.year + 1
    last = last_date.year if last_date >= pd.Timestamp(last_date.year, 12, 31) else last_date.year - 1
    return range(first, last + 1)


def _count_check(events, event, years, lo, hi, tol, extra=()) -> bool:
    dates = pd.concat([events.loc[events["event"] == event, "event_date"],
                       pd.Series(pd.to_datetime(list(extra)), dtype="datetime64[ns]")])
    counts = dates.dt.year.value_counts().reindex(years, fill_value=0)
    off = counts[(counts < lo) | (counts > hi)]
    fail = off[(off < lo - tol) | (off > hi + tol)]
    status = "FAIL" if not fail.empty else "WARN" if not off.empty else "PASS"
    want = f"{lo}" if lo == hi else f"{lo}-{hi}"
    note = f" (incl. {len(extra)} cancelled)" if len(extra) else ""
    print(f"  {status}  {event}: {want} per full year {years[0]}-{years[-1]}{note}")
    for year, n in off.items():
        print(f"          {year}: {n}{'  <- outside tolerance' if year in fail.index else ''}")
    return fail.empty


def validate_events(events: pd.DataFrame, cfg: dict, last_date: pd.Timestamp) -> None:
    years = list(_full_years(events, cfg, last_date))
    ok = True
    for event, spec in cfg["events"].items():
        if event == "fomc":
            ok &= _count_check(events, event, years, *FOMC_COUNT,
                               extra=cfg.get("fomc_cancelled", []))
        elif spec.get("first_release_after_quarter_end"):
            ok &= _count_check(events, event, years, *GDP_COUNT)
        else:
            ok &= _count_check(events, event, years, *MONTHLY_COUNT)

    # On release_date: event_date is a trading date by construction, so a
    # weekend in the source would otherwise be silently rolled to Monday.
    weekend = events[events["release_date"].dt.dayofweek >= 5]
    print(f"  {'PASS' if weekend.empty else 'FAIL'}  no release on a weekend "
          f"({len(weekend)} found)")
    if not weekend.empty:
        print(weekend.to_string(index=False))
    ok &= weekend.empty

    print("  Five dates to confirm by eye (first release of each event in the "
          "middle year of the sample):")
    mid = years[len(years) // 2]
    sample = (events[events["event_date"].dt.year == mid]
              .groupby("event").head(1).head(5))
    print(sample.to_string(index=False))

    if not ok:
        raise AssertionError("events: acceptance checks failed (see FAIL lines)")
