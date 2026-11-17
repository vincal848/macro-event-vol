"""Fed focus score: how much each release family matters for policy.

Three layers, combined with the fixed weights in config.yaml. Every input
must be known before the event date (no look-ahead).

Import as `from src import signal` from the repo root, so it is not confused
with Python's built-in `signal` module.
"""

import pandas as pd


def fundamentals(fred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Composite score per family from standardized macro series, using the
    data vintage available on each date.
    """
    raise NotImplementedError


def fed_assessment(cfg: dict) -> pd.DataFrame:
    """Score per family from the Fed's own stated risk assessments
    (Summary of Economic Projections), carried forward between releases.
    """
    raise NotImplementedError


def sensitivity(events: pd.DataFrame, fred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rolling response of the 2-year Treasury yield to surprises in each
    family, estimated on past events only.
    """
    raise NotImplementedError


def focus_score(layers: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Combine the layers into one score per family per date.

    Output: processed/focus_score.parquet
    """
    raise NotImplementedError
