"""Run the pipeline end to end, or one step with --step.

    python run.py
    python run.py --step data
    python run.py --step events --refresh   # re-pull instead of reading the cache
"""

import argparse

import pandas as pd

from src import backtest, data, events, options, signal


def step_data(cfg, refresh):
    # FRED series (data.pull_fred) are added here in step 5.
    db = data.wrds_connection()
    try:
        data.pull_underlying(cfg, refresh, db)
        data.pull_options(cfg, refresh, db)
    finally:
        db.close()
    data.validate_options(cfg)


def step_events(cfg, refresh):
    releases = data.pull_release_dates(cfg, refresh)
    underlying_path = data.raw_path("optionmetrics", "underlying.parquet")
    underlying = pd.read_parquet(underlying_path) if underlying_path.exists() else None
    calendar = events.build_calendar(cfg, releases, underlying)
    calendar = events.add_surprises(calendar, data.read_bloomberg_calendar(cfg))
    calendar = events.flag_overlaps(calendar)
    calendar.to_parquet(data.processed_path("events.parquet"), index=False)
    last_date = underlying["date"].max() if underlying is not None else pd.Timestamp.today()
    events.validate_events(calendar, cfg, last_date)


def _read_events():
    path = data.processed_path("events.parquet")
    if not path.exists():
        raise RuntimeError("events.parquet missing: run --step events first")
    return pd.read_parquet(path)


def step_options(cfg, refresh):
    underlying = pd.read_parquet(data.raw_path("optionmetrics", "underlying.parquet"))
    structures, misses = options.build_structures(
        data.load_options(cfg), underlying, _read_events(), cfg)
    structures.to_parquet(data.processed_path("structures.parquet"), index=False)
    options.validate_structures(structures, misses, len(underlying) - 1)


def step_signal(cfg, refresh):
    raise NotImplementedError


def step_backtest(cfg, refresh):
    path = data.processed_path("structures.parquet")
    if not path.exists():
        raise RuntimeError("structures.parquet missing: run --step options first")
    structures = pd.read_parquet(path)
    out = backtest.base_backtest(structures, cfg)
    out["results"].to_parquet(data.processed_path("results.parquet"), index=False)
    out["by_group"].to_parquet(data.processed_path("results_by_group.parquet"), index=False)
    out["event_minus_non_event"].to_parquet(
        data.processed_path("results_event_minus_non_event.parquet"), index=False)
    backtest.plot_cumulative(structures, data.processed_path("figures") / "cumulative_pnl.png")
    backtest.validate_backtest(structures, out)


STEPS = {
    "data": step_data,
    "events": step_events,
    "options": step_options,
    "signal": step_signal,
    "backtest": step_backtest,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=STEPS, help="run a single step")
    parser.add_argument("--refresh", action="store_true",
                        help="pull again even if cached files exist")
    args = parser.parse_args()

    cfg = data.load_config()
    for name in [args.step] if args.step else STEPS:
        print(f"== {name} ==")
        STEPS[name](cfg, args.refresh)


if __name__ == "__main__":
    main()
