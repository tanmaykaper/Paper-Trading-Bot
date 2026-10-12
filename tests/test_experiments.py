import numpy as np
import pandas as pd

from nsebot.research.experiments import (Panel, intraday_experiments, simulate_events,
                                         simulate_rotation, swing_experiments, top_k_per_day,
                                         trade_stats)
from conftest import daily_frame, intraday_sessions, uptrend_then_breakout


def _universe(n=12, bars=320):
    rng = np.random.default_rng(0)
    u = {}
    for i in range(n):
        drift = 0.002 if i < n // 2 else -0.0005
        u[f'S{i:02d}'] = daily_frame(500 * np.exp(np.cumsum(rng.normal(drift, 0.012, bars))))
    return u


def test_event_simulator_stop_and_time_exit_are_gap_aware():
    df = daily_frame(np.r_[np.full(60, 100.0), [100.0], np.full(5, 100.0), [80.0], np.full(20, 80.0)])
    P = Panel({'A': df})
    sig = np.zeros((len(P.dates), 1), dtype=bool)
    sig[60, 0] = True
    stop = np.full(sig.shape, 95.0)
    t = simulate_events(P, sig, stop, 'time', hold=15)
    assert len(t) == 1
    # Gapped from ~100 to 80 through a 95 stop: filled at the 80-ish open, not at 95.
    assert t['exit'].iloc[0] < 95.0 and t['r'].iloc[0] < -1.0


def test_top_k_per_day_keeps_best_scores():
    sig = np.array([[True, True, True]])
    score = np.array([[1.0, 3.0, 2.0]])
    assert top_k_per_day(sig, score, 2).tolist() == [[False, True, True]]


def test_rotation_holds_strongest_and_charges_costs():
    P = Panel(_universe())
    mom = (P.df['close'] / P.df['close'].shift(126) - 1).to_numpy()
    eq, costs = simulate_rotation(P, mom, P.liquid, n=3)
    assert costs > 0 and np.isfinite(eq.iloc[-1])
    assert eq.iloc[-1] > 50_000          # strong half of the universe trends up


def test_swing_experiments_render_both_halves():
    u = _universe()
    u['BRK'] = uptrend_then_breakout(n=320, seed=4)
    for k in u:
        u[k]['datetime'] = u['S00']['datetime']
    idx = daily_frame(20000 * np.exp(np.cumsum(np.full(320, 0.0008))))
    md, events, rot = swing_experiments(u, idx)
    assert 'OOS' in md and len(events) == 7 and len(rot) == 5


def test_intraday_experiments_run():
    bars = {'A': intraday_sessions(n_sessions=6, breakout_time='10:00'),
            'B': intraday_sessions(n_sessions=6, breakout_time='11:00', side='SHORT')}
    md, rows = intraday_experiments(bars, {}, None)
    assert '| I1 ORB baseline (Phase 2)' in md and len(rows) == 3


def test_trade_stats_kelly_sign():
    t = pd.DataFrame({'r': [2.0, -1.0, 2.0, -1.0], 'net_pct': [0.02, -0.01, 0.02, -0.01]})
    s = trade_stats(t)
    assert s['win'] == 0.5 and abs(s['kelly'] - 0.25) < 1e-9
