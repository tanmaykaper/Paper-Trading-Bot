import os
from datetime import time

import pandas as pd
import pytest

from nsebot.config import (BreakerConfig, ExitConfig, intraday_breakers, intraday_exits,
                           intraday_sizing, momentum_exits, swing_breakers, swing_exits,
                           swing_sizing)
from nsebot.risk import (CircuitBreakers, Position, advance, allocate, estimate_edge, evaluate,
                         open_risk, risk_fraction, size_position)
from nsebot.signals import Signal
import dataclasses


def _risk_cfg():
    """The Kelly ('risk') sizer with swing-like caps — swing itself now sizes by notional."""
    return dataclasses.replace(swing_sizing(), sizing_mode='risk', min_notional_inr=0.0,
                               max_portfolio_heat_pct=0.12)


# ═════════════════════════════════════════════════════════════════════════════
# Sizing
# ═════════════════════════════════════════════════════════════════════════════
def test_edge_is_the_prior_until_trades_arrive():
    cfg = swing_sizing()
    e = estimate_edge(cfg)
    assert e.win_rate == cfg.prior_win_rate and e.payoff == cfg.prior_payoff
    assert abs(e.kelly_full - (cfg.prior_win_rate - (1 - cfg.prior_win_rate) / cfg.prior_payoff)) < 1e-4
    assert e.prior_weight == 1.0


def test_realised_losers_drag_kelly_negative_and_size_to_floor():
    cfg = _risk_cfg()
    e = estimate_edge(cfg, [-1.0] * 150 + [0.8] * 50)
    assert e.kelly_full < 0 and e.prior_weight < 0.25
    assert risk_fraction(cfg, e) == cfg.risk_floor_pct
    d = size_position(cfg, equity=100_000, cash=100_000, entry=500, stop=480, edge=e)
    assert 'negative' in d.notes


def test_half_kelly_capped():
    cfg = swing_sizing()
    e = estimate_edge(cfg, [3.0] * 60 + [-1.0] * 40)          # a strong measured edge
    assert risk_fraction(cfg, e) == cfg.risk_cap_pct


def _edge(cfg, kelly_full=0.10):
    from nsebot.risk.sizing import EdgeEstimate
    return EdgeEstimate(0.45, 2.0, kelly_full, 0, 1.0)


def test_kelly_risk_binds_with_ample_cash():
    cfg = _risk_cfg()
    d = size_position(cfg, equity=100_000, cash=100_000, entry=1000, stop=950, edge=_edge(cfg))
    # half of 0.10 = 5% capped at 3% -> ₹3,000 at ₹50/share = 60 shares, but 40%
    # concentration allows only 40 shares of a ₹1,000 stock.
    assert d.binding == 'concentration' and d.qty == 40


def test_buying_power_binds_and_mis_leverage_lifts_it():
    cash_only = size_position(_risk_cfg(), equity=50_000, cash=5_000, entry=1000, stop=980,
                              edge=_edge(None))
    assert cash_only.binding == 'buying power' and cash_only.qty == 5
    mis = size_position(intraday_sizing(), equity=50_000, cash=5_000, entry=1000, stop=995,
                        edge=_edge(None))
    assert mis.qty == 25                         # 5x intraday margin on the same cash


def test_liquidity_and_heat_caps():
    cfg = _risk_cfg()
    d = size_position(cfg, equity=1_000_000, cash=1_000_000, entry=100, stop=95, edge=_edge(cfg),
                      median_turnover=1_00_000)                  # ₹1 lakh/day traded
    assert d.binding == 'liquidity' and d.qty == 10
    h = size_position(cfg, equity=100_000, cash=100_000, entry=100, stop=95, edge=_edge(cfg),
                      heat_room=500)
    assert h.binding == 'portfolio heat' and h.qty == 100


def test_lot_size_rounds_down():
    d = size_position(_risk_cfg(), equity=100_000, cash=100_000, entry=100, stop=90,
                      edge=_edge(None), lot_size=75)
    assert d.qty % 75 == 0


def test_regime_and_breaker_scale_risk():
    cfg = _risk_cfg()
    full = size_position(cfg, equity=100_000, cash=100_000, entry=100, stop=90, edge=_edge(cfg))
    half = size_position(cfg, equity=100_000, cash=100_000, entry=100, stop=90, edge=_edge(cfg),
                         regime_mult=0.5)
    assert half.risk_rupees == pytest.approx(full.risk_rupees / 2, rel=0.05)


# ═════════════════════════════════════════════════════════════════════════════
# Exits
# ═════════════════════════════════════════════════════════════════════════════
def _pos(side='LONG', entry=100.0, stop=95.0, mode='swing', atr=2.0):
    return Position('X', mode, side, 10, entry, stop, stop, pd.Timestamp('2026-10-01 10:00'), atr)


def _bar(o, h, l, c, ts='2026-10-02 11:00'):
    return {'open': o, 'high': h, 'low': l, 'close': c, 'datetime': pd.Timestamp(ts)}


def test_stop_fills_at_stop_or_worse_on_a_gap():
    cfg = momentum_exits()
    d = evaluate(_pos(), _bar(99, 100, 94, 96), cfg)
    assert d.action == 'EXIT' and d.price == 95.0 and d.reason == 'stop'
    g = evaluate(_pos(), _bar(90, 91, 88, 89), cfg)
    assert g.action == 'EXIT' and g.price == 90.0 and 'gap' in g.reason


def test_trail_goes_to_breakeven_then_chandelier_and_never_loosens():
    cfg = momentum_exits()
    p = _pos()
    d1 = evaluate(p, _bar(101, 105.5, 101, 105), cfg)             # +1.1R best
    assert d1.action == 'TRAIL' and d1.new_stop >= 100.0
    advance(p, _bar(101, 105.5, 101, 105), d1)
    d2 = evaluate(p, _bar(110, 125, 110, 124), cfg)               # +5R: tight 2xATR trail
    assert d2.action == 'TRAIL' and d2.new_stop == pytest.approx(125 - 2 * 2.0)
    advance(p, _bar(110, 125, 110, 124), d2)
    d3 = evaluate(p, _bar(122, 123, 121.5, 122), cfg)             # pullback: stop must not drop
    assert d3.action in ('HOLD', 'TRAIL') and (d3.new_stop or p.stop) >= p.stop


def test_short_side_mirror():
    cfg = intraday_exits()
    p = _pos('SHORT', entry=100.0, stop=101.0, mode='intraday', atr=0.5)
    d = evaluate(p, _bar(100.5, 101.2, 100.3, 101.1, '2026-10-02 11:00'), cfg)
    assert d.action == 'EXIT' and d.price == 101.0
    t = evaluate(p, _bar(99.0, 99.1, 98.0, 98.2, '2026-10-02 11:00'), cfg)    # +1.8R in favour
    assert t.action == 'TRAIL' and t.new_stop < 101.0


def test_mis_square_off_before_315_no_exceptions():
    cfg = intraday_exits()
    assert cfg.square_off == time(15, 10) and cfg.square_off < time(15, 15)
    p = _pos(mode='intraday', entry=100, stop=99)
    d = evaluate(p, _bar(103, 104, 102.5, 103.5, '2026-10-02 15:10'), cfg)
    assert d.action == 'EXIT' and d.price == 103.5 and 'square-off' in d.reason


def test_stagnation_and_max_hold():
    cfg = momentum_exits()
    p = _pos()
    p.bars_held = cfg.stagnation_bars - 1
    d = evaluate(p, _bar(100.5, 101, 100, 100.8), cfg)
    assert d.action == 'EXIT' and 'stagnant' in d.reason
    q = _pos()
    q.bars_held = cfg.max_hold_bars - 1
    q.best_price = 130
    q.stop = 120
    e = evaluate(q, _bar(125, 126, 124, 125), cfg)
    assert e.action == 'EXIT' and 'max hold' in e.reason


def test_reversion_exit_matches_the_researched_rule():
    cfg = swing_exits()                                         # S4b: EMA-5 target, 7 bars, no trail
    p = _pos(entry=100.0, stop=91.0)
    hold = evaluate(p, _bar(100, 103, 99, 102), cfg, exit_level=102.5)
    assert hold.action == 'HOLD'                                 # no trail, no breakeven
    tgt = evaluate(p, _bar(100, 104, 99, 103), cfg, exit_level=102.5)
    assert tgt.action == 'EXIT' and tgt.price == 103 and 'reversion target' in tgt.reason
    p.bars_held = 6
    t = evaluate(p, _bar(100, 101, 99, 100), cfg, exit_level=101.0)
    assert t.action == 'EXIT' and 'max hold 7' in t.reason
    stop_first = evaluate(_pos(entry=100.0, stop=91.0), _bar(92, 93, 90, 95), cfg, exit_level=94.0)
    assert stop_first.reason == 'stop' and stop_first.price == 91.0   # stop beats target, as researched


# ═════════════════════════════════════════════════════════════════════════════
# Breakers
# ═════════════════════════════════════════════════════════════════════════════
def test_intraday_loss_streak_halts_session_then_resumes_at_half_size(tmp_path):
    b = CircuitBreakers(intraday_breakers(), 'intraday', workdir=str(tmp_path))
    b.start_session('2026-10-05')
    for _ in range(3):
        b.on_exit(-100)
    v = b.check(equity=100_000)
    assert not v.entries_allowed and any('halted' in r for r in v.reasons)
    b.start_session('2026-10-06')
    v2 = b.check(equity=100_000)
    assert v2.entries_allowed and v2.size_mult == 0.5
    b.on_exit(+50)
    assert b.check(equity=100_000).size_mult == 1.0


def test_swing_preset_has_no_loss_streak_breaker(tmp_path):
    b = CircuitBreakers(swing_breakers(), 'swing', workdir=str(tmp_path))
    b.start_session('d1')
    for _ in range(20):
        b.on_exit(-1)
    v = b.check(100_000)
    assert v.entries_allowed and v.size_mult == 1.0


def test_streak_cooldown_counts_sessions(tmp_path):
    cfg = BreakerConfig(max_consecutive_losses=4, loss_cooldown_sessions=3)
    b = CircuitBreakers(cfg, 'swing', workdir=str(tmp_path))
    b.start_session('d1')
    for _ in range(4):
        b.on_exit(-1)
    assert not b.check(100_000).entries_allowed
    for d in ('d2', 'd3'):
        b.start_session(d)
        assert not b.check(100_000).entries_allowed
    b.start_session('d4')
    assert b.check(100_000).entries_allowed


def test_daily_loss_limit_flattens_intraday(tmp_path):
    b = CircuitBreakers(intraday_breakers(), 'intraday', workdir=str(tmp_path))
    b.start_session('2026-10-05')
    b.on_exit(-2_000)
    v = b.check(equity=100_000, open_pnl=-1_500)                 # -3.5% today
    assert not v.entries_allowed and v.flatten


def test_drawdown_latch_requires_a_human(tmp_path):
    b = CircuitBreakers(swing_breakers(), 'swing', workdir=str(tmp_path))
    b.check(equity=100_000)
    v = b.check(equity=74_000)
    assert not v.entries_allowed
    latch = tmp_path / 'BREAKER_TRIPPED_swing'
    assert latch.exists()
    assert not b.check(equity=99_000).entries_allowed            # recovery alone does not release it
    latch.unlink()
    b.s['peak_equity'] = 99_000
    assert b.check(equity=99_000).entries_allowed


def test_drawdown_latch_is_stamped_with_the_session_date(tmp_path):
    b = CircuitBreakers(swing_breakers(), 'swing', workdir=str(tmp_path))
    b.start_session('2020-03-06')
    b.check(52_000)
    b.start_session('2020-03-09')
    b.check(38_000)
    assert 'on 2020-03-09' in (tmp_path / 'BREAKER_TRIPPED_swing').read_text()


def test_kill_switch_and_trade_cap(tmp_path):
    b = CircuitBreakers(intraday_breakers(), 'intraday', workdir=str(tmp_path))
    b.start_session('2026-10-05')
    for _ in range(6):
        b.on_entry()
    assert any('trades/day' in r for r in b.check(100_000).reasons)
    (tmp_path / 'STOP_TRADING').write_text('')
    assert any('kill switch' in r for r in b.check(100_000).reasons)


def test_breaker_state_round_trips(tmp_path):
    b = CircuitBreakers(swing_breakers(), 'swing', workdir=str(tmp_path))
    b.start_session('d1')
    b.on_exit(-1)
    b2 = CircuitBreakers(swing_breakers(), 'swing', state=b.state(), workdir=str(tmp_path))
    assert b2.state() == b.state()


# ═════════════════════════════════════════════════════════════════════════════
# Allocator
# ═════════════════════════════════════════════════════════════════════════════
def _sig(sym, price=100.0, stop=95.0):
    return Signal(sym, 'swing', 'LONG', 'x', pd.Timestamp('2026-10-05'), price, stop, 2.0, 1.0)


def test_allocator_respects_slots_sectors_held_and_budget():
    cfg = swing_sizing()
    sectors = {'A': 'IT', 'B': 'IT', 'C': 'IT', 'D': 'BANK', 'E': 'BANK', 'F': 'AUTO'}
    held = [Position('E', 'swing', 'LONG', 10, 100, 95, 95, pd.Timestamp('2026-10-01'), 2.0)]
    plan = allocate([_sig(s) for s in 'ABCDEF'], held, cfg, equity=100_000, cash=100_000,
                    edge=_edge(cfg), max_new=3, sector_of=lambda s: sectors[s])
    assert [o.symbol for o in plan.orders] == ['A', 'B', 'D']      # C blocked by IT cap
    reasons = dict(plan.declined)
    assert 'sector IT' in reasons['C'] and 'already held' in reasons['E']
    assert 'budget' in reasons['F']


def test_open_risk_is_zero_once_stop_locks_profit():
    p = Position('X', 'swing', 'LONG', 10, 100, 95, 102, pd.Timestamp('2026-10-01'), 2.0)
    assert open_risk(p) == 0.0



# ═════════════════════════════════════════════════════════════════════════════
# Phase 4b: notional sizing and the breaker deadlock
# ═════════════════════════════════════════════════════════════════════════════
def test_notional_sizing_targets_a_fifth_of_equity():
    d = size_position(swing_sizing(), equity=50_000, cash=50_000, entry=500, stop=470,
                      edge=_edge(None))
    assert d.binding == 'target notional' and d.qty == 20 and d.notional == 10_000


def test_notional_sizing_caps_risk_for_wide_stops():
    # ₹20k (20% of ₹100k) at a 20% stop would risk ₹4,000 = 4% > the 3% cap
    # -> ₹3,000 / ₹100 per share = 30 shares = ₹15k, still above the ₹8k floor.
    d = size_position(swing_sizing(), equity=100_000, cash=100_000, entry=500, stop=400,
                      edge=_edge(None))
    assert d.binding == 'risk cap' and d.qty == 30 and d.notional == 15_000


def test_below_the_floor_is_skipped_not_shrunk():
    d = size_position(swing_sizing(), equity=50_000, cash=6_000, entry=500, stop=470,
                      edge=_edge(None))
    assert not d.ok and d.binding == 'below minimum notional'


def test_floor_lift_keeps_a_shrunken_book_trading():
    """Round 5: below ₹40k equity 20% is under the ₹8k floor, and every signal
    was skipped for good. The position is now lifted to the floor instead."""
    d = size_position(swing_sizing(), equity=38_000, cash=38_000, entry=500, stop=455,
                      edge=_edge(None))
    assert d.ok and d.qty == 16 and d.notional == 8_000 and d.binding == 'minimum notional (lifted)'
    assert d.risk_rupees <= 0.03 * 38_000
    off = size_position(dataclasses.replace(swing_sizing(), lift_to_floor=False), equity=38_000,
                        cash=38_000, entry=500, stop=455, edge=_edge(None))
    assert not off.ok and off.binding == 'below minimum notional'      # the old freeze


def test_floor_lift_never_breaks_another_cap():
    # risk cap: 16 shares x ₹200 stop distance = ₹3,200 > 3% of ₹38k
    assert not size_position(swing_sizing(), equity=38_000, cash=38_000, entry=500, stop=300,
                             edge=_edge(None)).ok
    # buying power: ₹6k of cash cannot buy ₹8k
    assert not size_position(swing_sizing(), equity=38_000, cash=6_000, entry=500, stop=455,
                             edge=_edge(None)).ok
    # concentration: ₹8k is more than 40% of ₹19k
    assert not size_position(swing_sizing(), equity=19_000, cash=19_000, entry=500, stop=490,
                             edge=_edge(None)).ok


def test_floor_lift_leaves_normal_equity_sizing_alone():
    # ₹41k: the ₹8,200 target clears the floor; 2 shares of ₹3,000 round below it -> skipped, as before
    d = size_position(swing_sizing(), equity=41_000, cash=41_000, entry=3_000, stop=2_800,
                      edge=_edge(None))
    assert not d.ok and d.binding == 'below minimum notional'
    d = size_position(swing_sizing(), equity=50_000, cash=50_000, entry=500, stop=470, edge=_edge(None))
    assert d.binding == 'target notional' and d.notional == 10_000


def test_notional_mode_ignores_a_negative_kelly_for_size():
    cfg = swing_sizing()
    bad = estimate_edge(cfg, [-1.0] * 200)
    d = size_position(cfg, equity=50_000, cash=50_000, entry=500, stop=470, edge=bad)
    assert d.notional == 10_000                         # no fee death-spiral from Kelly


def test_reduced_mode_expires_without_a_win(tmp_path):
    """The deadlock the backtest found: reduced size below the economic floor
    means no trades, so no winner can ever clear it. It must expire on the clock."""
    cfg = BreakerConfig(max_consecutive_losses=2, loss_cooldown_sessions=1, reduced_max_sessions=3)
    b = CircuitBreakers(cfg, 'swing', workdir=str(tmp_path))
    b.start_session('d0')
    b.on_exit(-1)
    b.on_exit(-1)
    mults = []
    for d in ('d1', 'd2', 'd3', 'd4', 'd5'):
        b.start_session(d)
        mults.append(b.check(100_000).size_mult)
    assert mults[-1] == 1.0 and 0.5 in mults
