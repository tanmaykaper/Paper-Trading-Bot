import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def daily_frame(closes, volume=200_000.0, start='2025-01-01', spread=0.01):
    closes = np.asarray(closes, dtype=float)
    vol = np.broadcast_to(np.asarray(volume, dtype=float), closes.shape).copy()
    return pd.DataFrame({
        'datetime': pd.bdate_range(start, periods=len(closes)),
        'open': closes * (1 - spread / 4),
        'high': closes * (1 + spread / 2),
        'low': closes * (1 - spread / 2),
        'close': closes,
        'volume': vol,
    })


def uptrend_then_breakout(n=200, seed=1, base=500.0):
    """Steady uptrend, a 25-bar sideways pause under resistance, then a
    high-volume close above the pause's high on the final bar."""
    rng = np.random.default_rng(seed)
    trend = base * np.exp(np.cumsum(rng.normal(0.0015, 0.008, n - 26)))
    pause = trend[-1] * (1 + rng.uniform(-0.015, 0.01, 25))
    last = pause.max() * 1.03
    closes = np.concatenate([trend, pause, [last]])
    vol = np.full(n, 200_000.0)
    vol[-1] = 600_000.0
    df = daily_frame(closes, vol)
    # A breakout bar that closes near its high.
    df.loc[df.index[-1], 'high'] = last * 1.002
    df.loc[df.index[-1], 'low'] = pause.max() * 0.995
    return df


def intraday_sessions(n_sessions=5, breakout_time=None, side='LONG', base=1000.0,
                      start='2026-09-28', or_volume=30_000.0, bar_volume=10_000.0):
    """5-minute NSE sessions, 09:15-15:25. Prior sessions drift flat; on the
    last session the opening range is 995-1005 on heavy volume and, if
    breakout_time is given, that bar breaks the range on a volume surge."""
    days = pd.bdate_range(start, periods=n_sessions)
    rows = []
    for k, day in enumerate(days):
        today = k == n_sessions - 1
        broken = False
        t = pd.Timestamp(day).tz_localize('Asia/Kolkata') + pd.Timedelta(hours=9, minutes=15)
        for i in range(75):
            ts = t + pd.Timedelta(minutes=5 * i)
            hhmm = ts.strftime('%H:%M')
            # After the breakout the stock holds its new level (+/- 8).
            lvl = base + ((8.0 if side == 'LONG' else -8.0) if broken else 0.0)
            o, h, l, c, v = lvl, lvl + 1.5, lvl - 1.5, lvl + 0.2, bar_volume
            if today and i < 3:
                o, h, l, c, v = base, base + 5.0, base - 5.0, base + (1 if i % 2 else -1), or_volume
            elif today and breakout_time and hhmm == breakout_time:
                broken = True
                if side == 'LONG':
                    o, h, l, c, v = base + 2.0, base + 8.0, base + 1.0, base + 7.9, 3 * bar_volume
                else:
                    o, h, l, c, v = base - 2.0, base - 1.0, base - 8.0, base - 7.9, 3 * bar_volume
            rows.append({'datetime': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    return pd.DataFrame(rows)


@pytest.fixture
def breakout_df():
    return uptrend_then_breakout()
