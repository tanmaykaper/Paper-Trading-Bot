import numpy as np
import pandas as pd

from nsebot import indicators as ind
from nsebot.data import completed_bars, sanitize
from conftest import daily_frame, intraday_sessions


def test_prior_high_excludes_the_current_bar():
    s = pd.Series([1, 2, 3, 10, 4], dtype=float)
    ph = ind.prior_high(s, 3)
    assert np.isnan(ph.iloc[2])
    assert ph.iloc[3] == 3.0          # bar 3 (value 10) cannot see itself
    assert ph.iloc[4] == 10.0


def test_atr_constant_range():
    df = daily_frame(np.full(60, 100.0), spread=0.02)       # high-low = 2.0 every bar
    assert abs(ind.atr(df, 14).iloc[-1] - 2.0) < 1e-6


def test_relative_volume_uses_prior_sessions_only():
    v = pd.Series([100.0] * 20 + [300.0])
    assert abs(ind.relative_volume(v, 20).iloc[-1] - 3.0) < 1e-9


def test_close_location_bounds():
    df = pd.DataFrame({'high': [10.0, 10.0, 10.0], 'low': [0.0, 0.0, 10.0],
                       'close': [10.0, 0.0, 10.0]})
    clv = ind.close_location(df)
    assert list(clv) == [1.0, -1.0, 0.0]          # flat bar -> neutral, not NaN


def test_session_vwap_resets_each_session():
    bars = intraday_sessions(n_sessions=2)
    vwap = ind.session_vwap(bars)
    first_of_day2 = bars.index[75]
    tp = (bars.loc[first_of_day2, ['high', 'low', 'close']].sum()) / 3
    assert abs(vwap.loc[first_of_day2] - tp) < 1e-9


def test_completed_bars_drops_forming_bar():
    bars = intraday_sessions(n_sessions=1)
    now = pd.Timestamp(bars['datetime'].iloc[10]) + pd.Timedelta(minutes=3)
    done = completed_bars(bars, now, 5)
    assert len(done) == 10                      # bar 10 started 3 min ago: still forming


def test_sanitize_normalises_and_drops_bad_rows():
    raw = pd.DataFrame({'Open': [1, 2, np.nan], 'High': [2, 3, 4], 'Low': [0.5, 1, 1],
                        'Close': [1.5, 2.5, 3], 'Volume': [10, 0, 5]},
                       index=pd.date_range('2026-01-01', periods=3))
    out = sanitize(raw)
    assert list(out.columns) == ['datetime', 'open', 'high', 'low', 'close', 'volume']
    assert len(out) == 1                       # zero-volume and NaN-open rows dropped
