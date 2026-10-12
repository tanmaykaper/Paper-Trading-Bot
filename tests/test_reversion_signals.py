import numpy as np
import pandas as pd

from nsebot.research.experiments import Panel
from nsebot.signals import ReversionSignalEngine
from conftest import daily_frame


def _dip(n=260, drop=0.10, drift=0.003, seed=2, volume=200_000.0):
    rng = np.random.default_rng(seed)
    closes = 500 * np.exp(np.cumsum(rng.normal(drift, 0.008, n - 3)))
    last = closes[-1]
    closes = np.r_[closes, last * (1 - drop / 3), last * (1 - 2 * drop / 3), last * (1 - drop)]
    return daily_frame(closes, volume)


def test_sharp_dip_in_uptrend_signals_with_3atr_stop():
    sigs = ReversionSignalEngine().scan({'DIP': _dip()})
    assert len(sigs) == 1
    s = sigs[0]
    assert s.trigger == 'dip_reversion' and s.side == 'LONG'
    assert abs((s.ref_price - s.stop) - 3.0 * s.atr) < 1e-6
    assert s.features['drop_3d'] <= -0.08


def test_shallow_dip_does_not_signal():
    assert ReversionSignalEngine().scan({'SHALLOW': _dip(drop=0.05)}) == []


def test_falling_knife_below_ema200_does_not_signal():
    assert ReversionSignalEngine().scan({'KNIFE': _dip(drift=-0.003)}) == []


def test_illiquid_and_short_history_do_not_signal():
    assert ReversionSignalEngine().scan({'THIN': _dip(volume=500.0)}) == []
    assert ReversionSignalEngine().scan({'NEW': _dip(n=150)}) == []


def test_live_engine_matches_the_researched_variant():
    """The rule the bot trades must be the rule that was tested (S4b)."""
    universe = {f'S{i}': _dip(seed=i, drop=0.06 + 0.01 * i) for i in range(6)}
    eng = ReversionSignalEngine()
    P = Panel(universe)
    c = P.a['close']
    ret3 = c / np.roll(c, 3, axis=0) - 1.0
    ret3[:3] = np.nan
    research = P.liquid & (c > P.a['ema200']) & (ret3 <= -0.08)
    for j, sym in enumerate(P.symbols):
        f = eng.features(universe[sym]).set_index('datetime').reindex(P.dates)
        live = f['signal'].fillna(False).to_numpy(bool)
        # The engine additionally requires 200 bars of history; on bars past
        # that warm-up the two must agree exactly.
        warm = np.arange(len(P.dates)) >= 199
        assert (live[warm] == research[warm, j]).all(), sym
