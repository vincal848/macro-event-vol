"""Contract selection and pricing of the event-day option structure.

Structure: short at-the-money straddle expiring on the event day, with
out-of-the-money options bought on both sides to cap the loss.
"""

import pandas as pd


def select_contracts(options: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """For each event, pick the contracts quoted at the prior close:
    the at-the-money call and put expiring on the event day, and the wings
    at cfg["options"]["wing_width_pct"] from the strike.

    Where no contract expires on the event day, use the nearest later expiry
    and record the days to expiry.
    """
    raise NotImplementedError


def implied_move(structures: pd.DataFrame) -> pd.Series:
    """Straddle price as a fraction of the index level at entry."""
    raise NotImplementedError


def realized_move(structures: pd.DataFrame, underlying: pd.DataFrame) -> pd.Series:
    """Absolute index return from the prior close to the event-day close."""
    raise NotImplementedError


def structure_pnl(structures: pd.DataFrame, underlying: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Profit and loss per event for one unit of the structure, after costs
    as set in cfg["costs"].
    """
    raise NotImplementedError
