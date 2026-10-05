# diagnostics/test_v2_paralysis.py  ── OFFLINE REPRODUCTION OF THE V2 LOCKS
# ═════════════════════════════════════════════════════════════════════════════
# Every claim in docs/AUTOPSY_V2.md that can be proven without network access
# is proven here, against the V2 code exactly as it stands in the repo. Run:
#
#     python -m pytest diagnostics/test_v2_paralysis.py -v
#
# V2's modules now live in legacy/ (retired in Phase 4); this file imports
# them from there so the autopsy stays reproducible.
#     python diagnostics/test_v2_paralysis.py          # no pytest needed
#
# Each test PASSES when the lock it describes is PRESENT. They are evidence,
# not regression guards: once the rebuild replaces these modules, the
# rebuild's own suite takes over.
# ═════════════════════════════════════════════════════════════════════════════

import os
import re
import sys
import tempfile
import types

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = os.path.join(ROOT, 'legacy')          # V2 was retired here in Phase 4
sys.path.insert(0, LEGACY)

import market_state                                     # noqa: E402
from market_state import MarketState, BreadthPanel     # noqa: E402
import orchestrator                                     # noqa: E402
from entry_execution import attempt_fill, get_profile as exec_profile  # noqa: E402


# ── Synthetic data ───────────────────────────────────────────────────────────
def _frame(closes, start='2026-01-01'):
    closes = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        'datetime': dates,
        'open': closes * 0.999,
        'high': closes * 1.006,
        'low': closes * 0.994,
        'close': closes,
        'volume': np.full(len(closes), 1_000_000.0),
    })


def _trend_then_selloff(n=260, seed=0, selloff_bars=25, selloff_drift=-0.006):
    """A normal uptrend that rolls over into a broad correction — the shape of
    the NSE tape from mid-September 2026 onward."""
    rng = np.random.default_rng(seed)
    up = rng.normal(0.0008, 0.010, n - selloff_bars)
    down = rng.normal(selloff_drift, 0.010, selloff_bars)
    return 100.0 * np.exp(np.cumsum(np.concatenate([up, down])))


def _correction_universe(n_symbols=60):
    return {f'S{i:02d}': _frame(_trend_then_selloff(seed=i)) for i in range(n_symbols)}


# ═════════════════════════════════════════════════════════════════════════════
# LOCK A — DEFENSIVE clamp zeroes exposure and slots in any correction
# ═════════════════════════════════════════════════════════════════════════════
def test_lock_a_defensive_clamp_zeroes_slots_in_a_correction():
    universe = _correction_universe()
    index_df = _frame(_trend_then_selloff(seed=999))

    ms = MarketState('growth').assess(index_df, breadth_panel=BreadthPanel(universe),
                                      base_slots=3)

    assert ms['state'] == 'DEFENSIVE', ms
    assert ms['triggers'], 'a correction should fire at least one defensive trigger'
    assert ms['exposure'] == 0.0
    assert ms['max_slots'] == 0
    assert ms['new_entries_allowed'] is False


def test_lock_a_risk_on_is_unreachable_while_index_trend_is_zero():
    """With Nifty below its EMA-20/50 (trend component = 0.00, as in every V2
    live log since 15 Sep), the best possible score from the other three
    components is 0.74 x their weights' share — compare to the 0.54 bar."""
    w = market_state.COMPONENT_WEIGHTS
    P = market_state.get_profile('growth')
    best_without_trend = (w['breadth'] * 1.0 + w['volatility'] * 1.0 + w['drawdown'] * 1.0) / sum(w.values())
    # Every other component must be near-perfect simultaneously to clear the bar.
    needed_avg = P['risk_on_threshold'] / best_without_trend
    assert best_without_trend < 0.75
    assert needed_avg > 0.72, (
        f'with trend=0 the other components must average {needed_avg:.0%} '
        f'of their maximum just to reach RISK_ON')


# ═════════════════════════════════════════════════════════════════════════════
# LOCK B — NEUTRAL with open slots still suspends entries (commit 027aa1f)
# ═════════════════════════════════════════════════════════════════════════════
def test_lock_b_neutral_with_open_slots_still_blocks_entries():
    st = MarketState('growth')
    # Exactly the live reading of 21 Sep 2026: NEUTRAL, exposure 0.13x, 1 slot.
    r = st._result('NEUTRAL', 0.36, 0.13, 3, {}, [], 'live 2026-09-21 reading')
    assert r['max_slots'] == 1
    assert r['exposure'] > 0.05
    assert r['new_entries_allowed'] is False, 'entry_requires_risk_on vetoes a slot it just granted'
    assert market_state.get_profile('growth').get('entry_requires_risk_on') is True


# ═════════════════════════════════════════════════════════════════════════════
# LOCK C — the T+1 pending-order book is never persisted on GitHub Actions
# ═════════════════════════════════════════════════════════════════════════════
class _DummyManager:
    def get_open_trades(self):
        return []


def _orch(state_path):
    sig = types.SimpleNamespace(calibrator=None)
    return orchestrator.TradingOrchestrator(_DummyManager(), sig, {}, profile='growth',
                                            trades_csv=os.devnull, state_path=state_path)


def test_lock_c_pending_plans_die_with_the_runner():
    plan = {'mode': 'MARKET_OPEN', 'limit_price': None, 'valid_until_bars': 1, 'rationale': 't'}
    details = {'entry_price': 500.0, 'position_size': 20, 'stop_loss': 480.0,
               'target_price': 540.0, 'indicators': {'sigma_abs': 8.0}}

    with tempfile.TemporaryDirectory() as run_1:
        state = os.path.join(run_1, orchestrator.STATE_JSON)
        o = _orch(state)
        o.pending.place('INFY', details, plan)
        o._save_state()
        assert os.path.exists(state)
        assert 'INFY' in _orch(state).pending.book      # survives on the SAME disk

    # GitHub Actions: every run is a fresh checkout. orchestrator_state.json is
    # not committed, so the next run sees no file and an empty book.
    with tempfile.TemporaryDirectory() as run_2:
        fresh = _orch(os.path.join(run_2, orchestrator.STATE_JSON))
        assert fresh.pending.book == {}, 'yesterday\'s plan is gone — it can never fill'


def test_lock_c_workflow_does_not_commit_the_state_file():
    wf = open(os.path.join(LEGACY, 'main_v2.yml')).read()      # the V2-era workflow
    add_lines = [l.strip() for l in wf.splitlines() if l.strip().startswith('git add')]
    assert add_lines, 'workflow has no git add step'
    assert all(orchestrator.STATE_JSON not in l for l in add_lines), add_lines
    assert not os.path.exists(os.path.join(ROOT, orchestrator.STATE_JSON))


# ═════════════════════════════════════════════════════════════════════════════
# LOCK D — condition conjunction: 27 sequential HOLD exits before a BUY
# ═════════════════════════════════════════════════════════════════════════════
def test_lock_d_signal_generator_has_27_sequential_veto_points():
    src = open(os.path.join(LEGACY, 'signal_generator.py')).read()
    body = src[src.index('    def _evaluate('):src.index('    def _build_frame(')]
    assert body.count("return 'HOLD'") == 27


def test_lock_d_momentum_gate_discards_70pct_before_any_signal():
    assert orchestrator.MOMENTUM_GATE_PERCENTILE == 70.0


# ═════════════════════════════════════════════════════════════════════════════
# LOCK E — even a persisted plan rarely fills: 0.5% gap cap on the next open
# ═════════════════════════════════════════════════════════════════════════════
def test_lock_e_market_open_abandoned_on_ordinary_gap():
    P = exec_profile('growth')
    assert P['max_slippage_pct'] == 0.005
    details = {'entry_price': 1000.0, 'indicators': {'sigma_abs': 25.0}}
    plan = {'mode': 'MARKET_OPEN', 'limit_price': None, 'valid_until_bars': 1}
    # A 0.6% gap up — an ordinary open for a momentum name that just signalled.
    res = attempt_fill(plan, {'open': 1006.0, 'high': 1020.0, 'low': 1001.0}, details, 'growth')
    assert res['status'] == 'ABANDONED', res


# ═════════════════════════════════════════════════════════════════════════════
# LANDMINE — hard-coded ₹0.05 tick (NSE ticks are price-banded)
# ═════════════════════════════════════════════════════════════════════════════
def test_landmine_tick_is_hardcoded():
    from signal_generator import NSEMicrostructure
    assert NSEMicrostructure.TICK == 0.05
    # ₹1,443.05 is a 0.05 multiple but not a 0.10 multiple — a live order at
    # this price on a stock whose instrument tick_size is 0.10 is rejected.
    p = NSEMicrostructure.round_to_tick(1443.07)
    assert p == 1443.05
    assert abs(round(p / 0.10) * 0.10 - p) > 1e-6


def test_landmine_runner_target_is_1000R_away():
    from tranche_manager import build_tranches
    import inspect
    assert '1000 * risk_per_share' in inspect.getsource(build_tranches)


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  ✔ {name}')
        except AssertionError as e:
            failed += 1
            print(f'  ✘ {name}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} locks reproduced')
    sys.exit(1 if failed else 0)
