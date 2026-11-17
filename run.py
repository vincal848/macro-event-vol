"""Run the pipeline end to end, or one step with --step.

    python run.py
    python run.py --step data
"""

import argparse

from src import backtest, data, events, options, signal


def step_data(cfg):
    data.pull_fred(cfg)
    data.pull_underlying(cfg)
    data.pull_options(cfg)


def step_events(cfg):
    calendar = events.build_calendar(cfg, data.read_bloomberg_calendar(cfg))
    calendar = events.flag_overlaps(events.add_surprises(calendar))
    calendar.to_parquet(data.processed_path("events.parquet"))


def step_options(cfg):
    raise NotImplementedError


def step_signal(cfg):
    raise NotImplementedError


def step_backtest(cfg):
    raise NotImplementedError


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
    args = parser.parse_args()

    cfg = data.load_config()
    for name in [args.step] if args.step else STEPS:
        print(f"== {name} ==")
        STEPS[name](cfg)


if __name__ == "__main__":
    main()
