"""Contract selection and pricing of the event-day option structure.

Structure: short at-the-money straddle on the nearest close-settled expiry on
or after the event date, with out-of-the-money options bought on both sides
to cap the loss (an iron fly). Entry at the close of the prior trading date,
exit at the close of the event date.
"""

import numpy as np
import pandas as pd

OUT_COLUMNS = [
    "event_date", "entry_date", "exdate", "exact_expiry", "is_event_day", "events",
    "families", "s0", "s1", "k", "k_put_wing", "k_call_wing", "credit_mid",
    "credit_cross", "close_cost_mid", "close_cost_cross", "pnl_mid", "pnl_cross",
    "max_loss_mid", "ret_mid", "ret_cross", "implied_move_pct", "realized_move_pct",
]


def eligible(quotes: pd.DataFrame) -> pd.DataFrame:
    """Two-sided quotes on close-settled contracts. am_settlement = 1 is the
    morning-settled standard monthly SPX (confirmed in the data, see data.py)."""
    return quotes[(quotes["best_bid"] > 0) & (quotes["best_offer"] > quotes["best_bid"])
                  & (quotes["am_settlement"] == 0)]


def _nearest(strikes: np.ndarray, target: float) -> float | None:
    """Strike nearest the target; ties go to the lower strike."""
    if len(strikes) == 0:
        return None
    strikes = np.sort(strikes)
    return float(strikes[np.abs(strikes - target).argmin()])   # argmin takes the first (lower)


def select(entry_quotes: pd.DataFrame, t: pd.Timestamp, s0: float, cfg: dict):
    """Pick the four contracts for event date t from eligible entry quotes.

    Returns (legs, reason): legs is a dict of quote rows keyed call_k, put_k,
    call_wing, put_wing, or None with the reason no structure was built.
    """
    o = cfg["options"]
    q = entry_quotes[entry_quotes["exdate"] >= t]
    if q.empty:
        return None, "no eligible expiry on or after event date"
    exdate = q["exdate"].min()
    if (exdate - t).days > o["max_days_to_expiry"]:
        return None, f"earliest expiry more than {o['max_days_to_expiry']} days out"
    q = q[q["exdate"] == exdate]
    calls, puts = q[q["cp_flag"] == "C"], q[q["cp_flag"] == "P"]

    both = np.intersect1d(calls["strike"].to_numpy(), puts["strike"].to_numpy())
    k = _nearest(both, s0)
    if k is None:
        return None, "no strike with both call and put eligible"
    w = o["wing_width_pct"]
    kp = _nearest(puts.loc[puts["strike"] < k, "strike"].to_numpy(), k * (1 - w))
    kc = _nearest(calls.loc[calls["strike"] > k, "strike"].to_numpy(), k * (1 + w))
    if kp is None or kc is None:
        return None, "wing missing"

    def one(df, strike):
        rows = df[df["strike"] == strike]
        if len(rows) != 1:
            raise RuntimeError(f"{t.date()}: {len(rows)} quotes for strike {strike} {exdate.date()}")
        return rows.iloc[0]

    legs = {"call_k": one(calls, k), "put_k": one(puts, k),
            "call_wing": one(calls, kc), "put_wing": one(puts, kp)}
    return legs, None


def price(legs: dict, exit_quotes: pd.DataFrame | None, t, s0: float, s1: float):
    """P&L of one structure. exit_quotes are eligible quotes on t (unused when
    the structure expires on t). Returns (row, reason)."""
    mid = {n: (r["best_bid"] + r["best_offer"]) / 2 for n, r in legs.items()}
    k, kp, kc = legs["call_k"]["strike"], legs["put_wing"]["strike"], legs["call_wing"]["strike"]
    exdate = legs["call_k"]["exdate"]
    exact = exdate == t

    credit_mid = mid["call_k"] + mid["put_k"] - mid["call_wing"] - mid["put_wing"]
    credit_cross = (legs["call_k"]["best_bid"] + legs["put_k"]["best_bid"]
                    - legs["call_wing"]["best_offer"] - legs["put_wing"]["best_offer"])

    if exact:
        settle = min(s1 - k, kc - k) if s1 > k else min(k - s1, k - kp)
        cost_mid = cost_cross = settle
    else:
        ex = exit_quotes.set_index("optionid") if exit_quotes is not None else pd.DataFrame()
        ids = {n: r["optionid"] for n, r in legs.items()}
        if not all(i in ex.index for i in ids.values()):
            return None, "exit quote missing on event date"
        e = {n: ex.loc[i] for n, i in ids.items()}
        emid = {n: (r["best_bid"] + r["best_offer"]) / 2 for n, r in e.items()}
        cost_mid = emid["call_k"] + emid["put_k"] - emid["call_wing"] - emid["put_wing"]
        cost_cross = (e["call_k"]["best_offer"] + e["put_k"]["best_offer"]
                      - e["call_wing"]["best_bid"] - e["put_wing"]["best_bid"])

    max_loss = max(kc - k, k - kp) - credit_mid
    row = {
        "exdate": exdate, "exact_expiry": bool(exact), "s0": s0, "s1": s1,
        "k": k, "k_put_wing": kp, "k_call_wing": kc,
        "credit_mid": credit_mid, "credit_cross": credit_cross,
        "close_cost_mid": cost_mid, "close_cost_cross": cost_cross,
        "pnl_mid": credit_mid - cost_mid, "pnl_cross": credit_cross - cost_cross,
        "max_loss_mid": max_loss,
        "ret_mid": (credit_mid - cost_mid) / max_loss,
        "ret_cross": (credit_cross - cost_cross) / max_loss,
        "implied_move_pct": (mid["call_k"] + mid["put_k"]) / s0,
        "realized_move_pct": abs(s1 / s0 - 1),
    }
    return row, None


def build_structures(options: pd.DataFrame, underlying: pd.DataFrame,
                     events: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """One structure per trading date T, entry at the close of the prior
    trading date. Returns the table and the count of each reason a structure
    was not built."""
    und = underlying.sort_values("date").reset_index(drop=True)
    quotes = eligible(options)
    print(f"  eligibility filter (two-sided, close-settled): removed "
          f"{len(options) - len(quotes):,} of {len(options):,} quote rows")
    by_date = dict(tuple(quotes.groupby("date")))
    empty = quotes.iloc[:0]

    rows, misses = [], []
    for i in range(1, len(und)):
        entry, t = und.at[i - 1, "date"], und.at[i, "date"]
        s0, s1 = und.at[i - 1, "close"], und.at[i, "close"]
        legs, reason = select(by_date.get(entry, empty), t, s0, cfg)
        if legs is None:
            misses.append(reason)
            continue
        row, reason = price(legs, by_date.get(t), t, s0, s1)
        if row is None:
            misses.append(reason)
            continue
        rows.append({"event_date": t, "entry_date": entry, **row})

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no structures built")
    ev = (events.sort_values("event").groupby("event_date")
          .agg(events=("event", ",".join),
               families=("family", lambda f: ",".join(sorted(set(f)))))
          .reset_index())
    out = out.merge(ev, on="event_date", how="left", validate="1:1")
    out["is_event_day"] = out["events"].notna()
    out[["events", "families"]] = out[["events", "families"]].fillna("")
    if out["event_date"].duplicated().any():
        raise RuntimeError("duplicate event_date in structures")
    misses = pd.Series(misses, dtype=str).value_counts()
    return out[OUT_COLUMNS], misses


# --- acceptance -------------------------------------------------------------

def _selftest() -> None:
    """Two hand-built iron flies, k = 100, wings 98 / 102, expiring on t."""
    t, entry = pd.Timestamp("2024-03-08"), pd.Timestamp("2024-03-07")
    q = pd.DataFrame({
        "date": entry, "exdate": t, "am_settlement": 0, "optionid": [1, 2, 3, 4],
        "cp_flag": ["C", "P", "C", "P"], "strike": [100.0, 100.0, 102.0, 98.0],
        "best_bid": [1.00, 0.90, 0.20, 0.15], "best_offer": [1.20, 1.10, 0.30, 0.25],
    })
    cfg = {"options": {"max_days_to_expiry": 7, "wing_width_pct": 0.02}}
    legs, reason = select(eligible(q), t, 100.2, cfg)
    assert reason is None, reason
    # credit_mid = 1.10 + 1.00 - 0.25 - 0.20 = 1.65; credit_cross = 1.00 + 0.90 - 0.30 - 0.25 = 1.35
    # max_loss_mid = 2 - 1.65 = 0.35
    inside, _ = price(legs, None, t, 100.2, 101.0)     # settles at 1.00
    assert round(inside["credit_mid"], 2) == 1.65 and round(inside["credit_cross"], 2) == 1.35
    assert round(inside["pnl_mid"], 2) == 0.65 and round(inside["pnl_cross"], 2) == 0.35
    outside, _ = price(legs, None, t, 100.2, 95.0)     # settles at the cap, 2.00
    assert round(outside["pnl_mid"], 2) == -0.35 and round(outside["pnl_cross"], 2) == -0.65
    assert round(outside["max_loss_mid"], 2) == 0.35 and round(outside["ret_mid"], 2) == -1.00
    print("  PASS  self-test: inside wings pnl_mid 0.65, outside wings pnl_mid -0.35")


def validate_structures(st: pd.DataFrame, misses: pd.Series, n_dates: int) -> None:
    _selftest()
    ok = True
    eps = 1e-9

    def check(name, bad):
        nonlocal ok
        ok &= not bad.any()
        print(f"  {'PASS' if not bad.any() else 'FAIL'}  {name}"
              + (f": {int(bad.sum())} rows" if bad.any() else ""))

    ex = st[st["exact_expiry"]]
    check("pnl_mid <= credit_mid on exact-expiry rows", ex["pnl_mid"] > ex["credit_mid"] + eps)
    check("pnl_mid >= -max_loss_mid on exact-expiry rows", ex["pnl_mid"] < -ex["max_loss_mid"] - eps)
    check("credit_cross <= credit_mid", st["credit_cross"] > st["credit_mid"] + eps)
    for col in ["s0", "s1", "k", "credit_mid", "pnl_mid", "ret_mid"]:
        check(f"no nulls in {col}", st[col].isna())

    print(f"  trading dates: {n_dates:,}  structures built: {len(st):,}  "
          f"exact-expiry: {int(st['exact_expiry'].sum()):,}")
    print("  not built, by reason:")
    for reason, n in misses.items():
        print(f"    {reason}: {n:,}")

    long = st.assign(event=st["events"].str.split(",")).explode("event")
    long["event"] = long["event"].replace("", "(non-event day)")
    moves = long.groupby("event")[["implied_move_pct", "realized_move_pct"]].mean()
    moves["n"] = long.groupby("event").size()
    print("  mean implied and realized move (%), all structures:")
    print((moves.assign(implied_move_pct=moves["implied_move_pct"] * 100,
                        realized_move_pct=moves["realized_move_pct"] * 100)
           .round(3).to_string()))

    if not ok:
        raise AssertionError("structures: acceptance checks failed (see FAIL lines)")
