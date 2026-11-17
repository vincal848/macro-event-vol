"""Base strategy, overlay, and performance metrics."""

import numpy as np
import pandas as pd


def base_positions(events: pd.DataFrame) -> pd.Series:
    """Equal size on every event."""
    return pd.Series(1.0, index=events.index, name="size")


def overlay_positions(events: pd.DataFrame, score: pd.DataFrame, cfg: dict) -> pd.Series:
    """Size each event by the focus score of its family: full size where the
    Fed is least focused, reduced where it is most focused.
    """
    raise NotImplementedError


def run(pnl_per_unit: pd.Series, size: pd.Series) -> pd.Series:
    """Per-event strategy return."""
    return pnl_per_unit * size


def metrics(returns: pd.Series, events_per_year: float) -> dict:
    """Mean return, Sharpe ratio and maximum drawdown."""
    cumulative = returns.cumsum()
    drawdown = cumulative - cumulative.cummax()
    std = returns.std()
    return {
        "mean": returns.mean(),
        "sharpe": np.sqrt(events_per_year) * returns.mean() / std if std > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "n_events": int(returns.count()),
    }


def placebo(pnl_per_unit: pd.Series, size: pd.Series, n: int = 1000, seed: int = 0) -> pd.Series:
    """Mean return of the overlay with sizes shuffled across events.

    If the real overlay does no better than this distribution, the score adds
    no information.
    """
    rng = np.random.default_rng(seed)
    values = size.to_numpy()
    means = [(pnl_per_unit.to_numpy() * rng.permutation(values)).mean() for _ in range(n)]
    return pd.Series(means, name="placebo_mean")
