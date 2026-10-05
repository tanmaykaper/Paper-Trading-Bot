"""Small, causal, vectorised indicators.

Every function returns a Series aligned to its input where bar t depends only
on bars <= t, so the same code drives the live decision (last row) and the
research walk-forward (every row) without a lookahead leak. No TA-Lib: one
less C dependency to break a GitHub Actions run.
"""

import numpy as np
import pandas as pd


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def true_range(df):
    prev_close = df['close'].shift(1)
    return pd.concat([df['high'] - df['low'],
                      (df['high'] - prev_close).abs(),
                      (df['low'] - prev_close).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    """Wilder's ATR (RMA of true range)."""
    return true_range(df).ewm(alpha=1.0 / n, adjust=False).mean()


def prior_high(series, n):
    """Highest value over the n bars BEFORE this one — today cannot set its own bar."""
    return series.shift(1).rolling(n, min_periods=n).max()


def prior_low(series, n):
    return series.shift(1).rolling(n, min_periods=n).min()


def relative_volume(volume, n=20):
    """Today's volume over the average of the previous n sessions."""
    base = volume.shift(1).rolling(n, min_periods=max(5, n // 2)).mean()
    return volume / base.replace(0, np.nan)


def close_location(df):
    """Close location value in [-1, 1]: +1 closed on the high, -1 on the low.
    A bar-level proxy for who won the auction — buyers or sellers."""
    rng = (df['high'] - df['low']).replace(0, np.nan)
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / rng
    return clv.fillna(0.0)


def rate_of_change(series, n):
    return series / series.shift(n) - 1.0


def median_turnover(df, n=20):
    """Median daily traded value in rupees — robust to one block deal."""
    return (df['close'] * df['volume']).rolling(n, min_periods=max(5, n // 2)).median()


def session_vwap(df):
    """VWAP that resets every session. Expects intraday bars with a tz-aware
    or naive 'datetime' column; sessions are split on calendar date."""
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    session = pd.to_datetime(df['datetime']).dt.date
    pv = (tp * df['volume']).groupby(session).cumsum()
    vol = df['volume'].groupby(session).cumsum()
    return (pv / vol.replace(0, np.nan)).fillna(tp)


def cross_sectional_percentile(panel):
    """Rank each row across columns, 0-100. NaN stays NaN and is excluded."""
    return panel.rank(axis=1, pct=True) * 100.0
