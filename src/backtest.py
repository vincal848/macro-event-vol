"""Base strategy, overlay, and performance metrics.

Returns are per structure: P&L divided by the structure's maximum loss at
mid (ret_mid / ret_cross from options.py). Sizes scale those returns.
"""

import numpy as np
import pandas as pd


def base_positions(events: pd.DataFrame) -> pd.Series:
    """Equal size on every event."""
    return pd.Series(1.0, index=events.index, name="size")


def overlay_positions(events: pd.DataFrame, score: pd.DataFrame, cfg: dict) -> pd.Series:
    """Size each event by the focus score of its family: full size where the
    Fed is least focused, reduced where it is most focused. Built in step 6.
    """
    raise NotImplementedError


def run(pnl_per_unit: pd.Series, size: pd.Series) -> pd.Series:
    """Per-event strategy return."""
    return pnl_per_unit * size


# --- metrics ----------------------------------------------------------------

def bootstrap_mean_ci(x: np.ndarray, draws: int, seed: int, level: float = 0.95) -> tuple:
    """Percentile interval for the mean, resampling observations (events)."""
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def metrics(returns: pd.Series, events_per_year: float, draws: int = 0, seed: int = 0) -> dict:
    """Performance of one return series, one value per structure.

    sharpe_annual uses the variant's own events per year. max_drawdown is on
    cumulative summed returns. worst_5_mean is the mean of the 5% worst
    returns (at least one). skew is the sample skewness (pandas, bias-adjusted).
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n == 0:
        raise ValueError("metrics: empty return series")
    mean = r.mean()
    std = r.std(ddof=1) if n > 1 else np.nan
    cum = np.cumsum(r)
    drawdown = cum - np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    worst = np.sort(r)[: max(1, int(np.floor(0.05 * n)))]
    out = {
        "n": n,
        "mean_ret": mean,
        "std_ret": std,
        "t_stat": mean / (std / np.sqrt(n)) if std and std > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "sharpe_annual": np.sqrt(events_per_year) * mean / std if std and std > 0 else np.nan,
        "max_drawdown": float(min(drawdown.min(), 0.0)),
        "worst_5_mean": float(worst.mean()),
        "skew": float(pd.Series(r).skew()) if n > 2 else np.nan,
    }
    if draws:
        out["mean_ret_lo"], out["mean_ret_hi"] = bootstrap_mean_ci(r, draws, seed)
    return out


def events_per_year(dates: pd.Series) -> float:
    """Observations per year over the span the series covers."""
    years = (dates.max() - dates.min()).days / 365.25
    return len(dates) / years if years > 0 else np.nan


def placebo(pnl_per_unit: pd.Series, size: pd.Series, n: int = 1000, seed: int = 0) -> pd.Series:
    """Mean return of the overlay with sizes shuffled across events.

    If the real overlay does no better than this distribution, the score adds
    no information.
    """
    rng = np.random.default_rng(seed)
    values = size.to_numpy()
    means = [(pnl_per_unit.to_numpy() * rng.permutation(values)).mean() for _ in range(n)]
    return pd.Series(means, name="placebo_mean")


# --- step 4: base backtest --------------------------------------------------

def base_backtest(st: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Base strategy (event days, size 1) against the benchmark (non-event
    days), on exact-expiry and all structures, with mid and crossed fills."""
    b = cfg["backtest"]
    rows = []
    for sample, sub in [("exact_expiry", st[st["exact_expiry"]]), ("all", st)]:
        for strategy, part in [("base", sub[sub["is_event_day"]]),
                               ("benchmark", sub[~sub["is_event_day"]])]:
            for fill in ["mid", "cross"]:
                ret = run(part[f"ret_{fill}"], base_positions(part))
                epy = events_per_year(part["event_date"])
                m = metrics(ret, epy, b["bootstrap_draws"], b["seed"])
                rows.append({"strategy": strategy, "sample": sample, "fill": fill,
                             "events_per_year": epy, **m})
    results = pd.DataFrame(rows)

    # Mean return by event and by family (event days only; a date with two
    # releases counts once under each).
    ev = st[st["is_event_day"]]
    by = []
    for col, label in [("events", "event"), ("families", "family")]:
        long = ev.assign(key=ev[col].str.split(",")).explode("key")
        for sample, sub in [("exact_expiry", long[long["exact_expiry"]]), ("all", long)]:
            g = sub.groupby("key").agg(n=("ret_mid", "size"), mean_ret_mid=("ret_mid", "mean"),
                                       mean_ret_cross=("ret_cross", "mean"))
            by.append(g.reset_index().rename(columns={"key": "name"})
                      .assign(by=label, sample=sample))
    by_group = pd.concat(by, ignore_index=True)[
        ["by", "name", "sample", "n", "mean_ret_mid", "mean_ret_cross"]]

    # Event minus non-event mean return, bootstrapping each group separately.
    rng = np.random.default_rng(b["seed"])
    diffs = []
    for sample, sub in [("exact_expiry", st[st["exact_expiry"]]), ("all", st)]:
        for fill in ["mid", "cross"]:
            e = sub.loc[sub["is_event_day"], f"ret_{fill}"].to_numpy()
            n = sub.loc[~sub["is_event_day"], f"ret_{fill}"].to_numpy()
            d = (e[rng.integers(0, len(e), (b["bootstrap_draws"], len(e)))].mean(1)
                 - n[rng.integers(0, len(n), (b["bootstrap_draws"], len(n)))].mean(1))
            diffs.append({"sample": sample, "fill": fill, "diff_mean_ret": e.mean() - n.mean(),
                          "diff_lo": np.quantile(d, 0.025), "diff_hi": np.quantile(d, 0.975),
                          "n_event": len(e), "n_non_event": len(n)})
    return {"results": results, "by_group": by_group, "event_minus_non_event": pd.DataFrame(diffs)}


def plot_cumulative(st: pd.DataFrame, path) -> None:
    """Cumulative return, base vs benchmark, exact-expiry sample, both fills."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = st[st["exact_expiry"]].sort_values("event_date")
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, part in [("base (event days)", sub[sub["is_event_day"]]),
                        ("benchmark (non-event days)", sub[~sub["is_event_day"]])]:
        for fill, style in [("mid", "-"), ("cross", ":")]:
            ax.plot(part["event_date"], part[f"ret_{fill}"].cumsum(), style,
                    label=f"{label}, {fill} fills")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Cumulative return per unit of max loss, exact-expiry structures")
    ax.set_ylabel("sum of returns")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- acceptance -------------------------------------------------------------

def _selftest() -> None:
    # Constant series: no dispersion, no drawdown.
    c = metrics(pd.Series([0.1] * 10), events_per_year=50)
    assert c["n"] == 10 and abs(c["mean_ret"] - 0.1) < 1e-12 and c["std_ret"] < 1e-12
    assert c["hit_rate"] == 1.0 and c["max_drawdown"] == 0.0 and abs(c["worst_5_mean"] - 0.1) < 1e-12

    # Known series [1, -2, 3, -1, 4], by hand:
    # mean 1.0; deviations 0,-3,2,-2,3 -> SS 26 -> var 6.5, std 2.5495
    # t = 1 / (2.5495 / sqrt 5) = 0.8771; Sharpe at 4/yr = 2 * 1 / 2.5495 = 0.7845
    # cumulative 1,-1,2,1,5 -> peak 1 then -1 = drawdown -2; hit rate 3/5
    # worst 5% of 5 -> 1 obs -> -2; skew (pandas, adjusted) = 0
    k = metrics(pd.Series([1.0, -2.0, 3.0, -1.0, 4.0]), events_per_year=4)
    expect = {"n": 5, "mean_ret": 1.0, "std_ret": 2.5495, "t_stat": 0.8771, "hit_rate": 0.6,
              "sharpe_annual": 0.7845, "max_drawdown": -2.0, "worst_5_mean": -2.0, "skew": 0.0}
    for key, v in expect.items():
        assert abs(k[key] - v) < 1e-4, (key, k[key], v)
    print("  PASS  self-test: metrics match hand values (constant and 5-point series)")


def validate_backtest(st: pd.DataFrame, out: dict) -> None:
    _selftest()
    ev, non = st[st["is_event_day"]], st[~st["is_event_day"]]
    overlap = set(ev["event_date"]) & set(non["event_date"])
    print(f"  {'PASS' if not overlap else 'FAIL'}  event and non-event samples do not overlap")
    together = len(ev) + len(non) == len(st)
    print(f"  {'PASS' if together else 'FAIL'}  event ({len(ev):,}) + non-event ({len(non):,}) "
          f"= structures ({len(st):,})")
    cols = ["strategy", "sample", "fill", "n", "mean_ret", "mean_ret_lo", "mean_ret_hi",
            "std_ret", "t_stat", "hit_rate", "sharpe_annual", "max_drawdown", "worst_5_mean", "skew"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out["results"][cols].round(4).to_string(index=False))
        print("  event minus non-event mean return:")
        print(out["event_minus_non_event"].round(4).to_string(index=False))
    if overlap or not together:
        raise AssertionError("backtest: acceptance checks failed")
