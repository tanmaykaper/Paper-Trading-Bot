import numpy as np
import pandas as pd

from nsebot.config import SwingSignalConfig
from nsebot.signals import SwingSignalEngine
from conftest import daily_frame, uptrend_then_breakout


def test_breakout_fires_on_volume_confirmed_new_high(breakout_df):
    sigs = SwingSignalEngine().scan({'AAA': breakout_df})
    assert len(sigs) == 1
    s = sigs[0]
    assert s.trigger == 'breakout' and s.side == 'LONG'
    assert s.stop < s.ref_price
    cfg = SwingSignalConfig()
    assert cfg.min_stop_pct - 1e-9 <= s.stop_pct <= cfg.max_stop_pct + 1e-9


def test_downtrend_never_signals():
    rng = np.random.default_rng(3)
    closes = 800 * np.exp(np.cumsum(rng.normal(-0.002, 0.01, 200)))
    vol = np.full(200, 200_000.0)
    vol[-1] = 900_000.0
    assert SwingSignalEngine().scan({'DDD': daily_frame(closes, vol)}) == []


def test_illiquid_name_is_skipped(breakout_df):
    thin = breakout_df.copy()
    thin['volume'] = thin['volume'] / 1000.0           # ~₹1 lakh/day
    assert SwingSignalEngine().scan({'THIN': thin}) == []


def test_blow_off_day_is_skipped(breakout_df):
    blow = breakout_df.copy()
    i = blow.index[-1]
    blow.loc[i, 'close'] = blow['close'].iloc[-2] * 1.12            # +12% day
    blow.loc[i, 'high'] = blow.loc[i, 'close'] * 1.001
    assert SwingSignalEngine().scan({'BLOW': blow}) == []


def test_no_lookahead_features_match_truncated_history(breakout_df):
    eng = SwingSignalEngine()
    full = eng.features(breakout_df)
    for k in (150, 175, len(breakout_df)):
        part = eng.features(breakout_df.iloc[:k])
        for col in ('trigger', 'stop', 'trend', 'liquid'):
            a, b = full[col].iloc[k - 1], part[col].iloc[-1]
            assert (a == b) or (pd.isna(a) and pd.isna(b)) or abs(float(a) - float(b)) < 1e-9, col


def test_ranking_prefers_stronger_relative_strength():
    strong = uptrend_then_breakout(seed=11)
    weak = uptrend_then_breakout(seed=12)
    # Flatten the weak name's last 3 months so its RS is lower.
    weak.loc[weak.index[-90:-1], ['open', 'high', 'low', 'close']] = (
        weak.loc[weak.index[-90:-1], ['open', 'high', 'low', 'close']].iloc[0].to_numpy())
    weak.loc[weak.index[-1], ['close', 'high', 'low']] = weak['close'].iloc[-2] * np.array([1.03, 1.032, 0.99])
    weak['datetime'] = strong['datetime']
    sigs = SwingSignalEngine().scan({'STRONG': strong, 'WEAK': weak})
    names = [s.symbol for s in sigs]
    if len(names) == 2:
        assert names[0] == 'STRONG'
    else:
        assert names == ['STRONG']


def test_stale_symbol_cannot_signal(breakout_df):
    stale = breakout_df.iloc[:-5].copy()          # stopped printing 5 sessions ago
    live = uptrend_then_breakout(seed=5)
    sigs = SwingSignalEngine().scan({'STALE': stale, 'LIVE': live})
    assert all(s.symbol != 'STALE' for s in sigs)
