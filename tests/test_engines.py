import dataclasses

import numpy as np
import pandas as pd
import pytest

from nsebot.broker.paper import PaperBroker
from nsebot.config import BotConfig
from nsebot.engine import DailyMarket, IntradayEngine, SwingEngine
from nsebot.ledger import Ledger
from nsebot.market import band_tick, round_to_tick
from nsebot.risk.exits import Position
from conftest import daily_frame, intraday_sessions


# ═════════════════════════════════════════════════════════════════════════════
# Swing
# ═════════════════════════════════════════════════════════════════════════════
def _dip_then_bounce(n_trend=260, seed=3):
    rng = np.random.default_rng(seed)
    trend = 500 * np.exp(np.cumsum(rng.normal(0.003, 0.006, n_trend)))
    last = trend[-1]
    dip = [last * 0.965, last * 0.93, last * 0.90]                  # -10% in 3 sessions
    bounce = [last * 0.905, last * 0.93, last * 0.96, last * 0.985, last * 1.0, last * 1.01]
    closes = np.r_[trend, dip, bounce]
    df = daily_frame(closes, 300_000.0)
    df['open'] = df['close'].shift(1).fillna(df['close'])           # opens at prior close
    df['high'] = df[['open', 'close']].max(axis=1) * 1.004
    df['low'] = df[['open', 'close']].min(axis=1) * 0.996
    return df, n_trend + 2                                          # index of the signal bar


@pytest.fixture
def swing_world():
    dip, sig_i = _dip_then_bounce()
    flat = daily_frame(400 * np.exp(np.cumsum(np.full(len(dip), 0.0005))), 300_000.0)
    index = daily_frame(20000 * np.exp(np.cumsum(np.full(len(dip), 0.0008))))
    market = DailyMarket({'DIP': dip, 'FLAT': flat}, index)
    return market, market.dates[sig_i]


def _engine(root):
    L = Ledger(str(root), 'swing', 50_000)
    return SwingEngine(BotConfig(), PaperBroker(slippage_bps=5), L, workdir=str(root), log=lambda m: None), L


def test_plan_survives_a_fresh_checkout_and_fills_next_open(swing_world, tmp_path):
    """The exact failure that froze V2: a plan placed today must exist tomorrow."""
    market, t = swing_world
    eng, L = _engine(tmp_path)
    rep = eng.run(market, t)
    assert [p[0] for p in rep['placed']] == ['DIP']
    # New process, state read back from disk — what a GitHub Actions run sees.
    eng2, L2 = _engine(tmp_path)
    assert L2.state['pending'][0]['spec']['symbol'] == 'DIP'
    nxt = market.dates[market.dates.get_loc(t) + 1]
    rep2 = eng2.run(market, nxt)
    assert rep2['filled'] and rep2['filled'][0][0] == 'DIP'
    raw = market.bar('DIP', nxt)['open'] * 1.0005              # next open + 5 bps slippage
    expected = round_to_tick(raw, band_tick(raw), 'up')        # snapped UP to a valid NSE tick
    assert rep2['filled'][0][2] == pytest.approx(expected, abs=1e-9)


def test_full_round_trip_exits_on_reversion_target_and_books_net(swing_world, tmp_path):
    market, t = swing_world
    eng, L = _engine(tmp_path)
    i = market.dates.get_loc(t)
    closed = []
    for d in market.dates[i:i + 8]:
        closed += eng.run(market, d)['closed']
    assert closed and closed[0][0] == 'DIP' and 'reversion target' in closed[0][1]
    trades = L.closed_trades()
    assert len(trades) == 1
    assert L.cash == pytest.approx(50_000 + trades['net_pnl'].iloc[0], abs=0.05)
    assert trades['costs'].iloc[0] > 0 and np.isfinite(trades['net_r'].iloc[0])


def test_rerun_of_the_same_session_is_a_no_op(swing_world, tmp_path):
    market, t = swing_world
    eng, L = _engine(tmp_path)
    eng.run(market, t)
    again = eng.run(market, t)
    assert 'already processed' in again['status']
    assert len(L.state['pending']) == 1


def test_missed_runs_catch_up_bar_by_bar(swing_world, tmp_path):
    market, t = swing_world
    eng, L = _engine(tmp_path)
    eng.run(market, t)
    i = market.dates.get_loc(t)
    rep = eng.run(market, market.dates[i + 6])                     # five sessions skipped
    assert rep['filled'] and rep['closed']
    trade = L.closed_trades().iloc[0]
    assert pd.Timestamp(trade['entry_time']) == market.dates[i + 1]  # still the next-open fill
    assert pd.Timestamp(trade['exit_time']) < market.dates[i + 6]    # exited on the right bar


def _small_book(root, cash=38_000, lift=True):
    cfg = BotConfig()
    if not lift:
        cfg = dataclasses.replace(cfg, swing_sizing=dataclasses.replace(cfg.swing_sizing,
                                                                        lift_to_floor=False))
    L = Ledger(str(root), 'swing', cash)
    return SwingEngine(cfg, PaperBroker(slippage_bps=5), L, workdir=str(root), log=lambda m: None)


def test_a_book_below_40k_keeps_trading(swing_world, tmp_path):
    market, t = swing_world
    rep = _small_book(tmp_path).run(market, t)
    assert [p[0] for p in rep['placed']] == ['DIP']
    assert rep['placed'][0][4] == 'minimum notional (lifted)'


def test_a_starved_book_says_it_is_not_trading(tmp_path):
    """Signals, free slots, nothing opened, three sessions running: warn — the
    shape of V2's paralysis and of the round-5 floor freeze."""
    frames = {f'DIP{k}': _dip_then_bounce(260 + k, seed=3 + k)[0] for k in range(3)}
    index = daily_frame(20000 * np.exp(np.cumsum(np.full(270, 0.0008))))
    market = DailyMarket(frames, index)
    days = market.dates[262:265]                      # DIP0, DIP1, DIP2 signal on consecutive days

    frozen = _small_book(tmp_path / 'off', lift=False)
    reps = [frozen.run(market, d) for d in days]
    assert [r['signals'] for r in reps] == [1, 1, 1] and not any(r['placed'] for r in reps)
    assert not reps[1]['warnings']
    assert any(w.startswith('NOT TRADING: 3 sessions') for w in reps[2]['warnings'])

    fixed = _small_book(tmp_path / 'on')
    reps = [fixed.run(market, d) for d in days]
    assert reps[0]['placed'] and reps[1]['placed']
    assert reps[2]['declined'] == [('DIP2', 'sector OTHER at its cap of 2')]   # a real cap, not starvation
    assert not any(r['warnings'] for r in reps)


def test_kill_switch_blocks_entries_but_not_exits(swing_world, tmp_path):
    market, t = swing_world
    eng, L = _engine(tmp_path)
    (tmp_path / 'STOP_TRADING').write_text('')
    rep = eng.run(market, t)
    assert rep['placed'] == [] and any('kill switch' in r for r in rep['breakers'])


# ═════════════════════════════════════════════════════════════════════════════
# Intraday
# ═════════════════════════════════════════════════════════════════════════════
def _at(bars, hhmm, secs=20):
    day = pd.Timestamp(bars['datetime'].iloc[-1]).normalize()
    h, m = map(int, hhmm.split(':'))
    return day + pd.Timedelta(hours=h, minutes=m, seconds=secs)


def _intraday(root):
    L = Ledger(str(root), 'intraday', 50_000)
    return IntradayEngine(BotConfig(), PaperBroker(slippage_bps=5), L, workdir=str(root),
                          log=lambda m: None, universe=['AAA']), L


def test_intraday_signal_fill_and_square_off_before_315(tmp_path):
    bars = {'AAA': intraday_sessions(breakout_time='10:00', side='LONG')}
    eng, L = _intraday(tmp_path)
    r1 = eng.step(bars, _at(bars['AAA'], '10:05'))
    assert r1['placed'] and r1['placed'][0][:2] == ('AAA', 'LONG')
    r2 = eng.step(bars, _at(bars['AAA'], '10:10'))
    assert r2['filled'] and r2['filled'][0][0] == 'AAA'
    r3 = eng.step(bars, _at(bars['AAA'], '15:11'))
    assert r3['closed'] and 'square-off' in r3['closed'][0][1]
    t = L.closed_trades().iloc[0]
    assert pd.Timestamp(t['exit_time']).time() <= pd.Timestamp('15:15').time()
    assert L.positions == []


def test_intraday_one_trade_per_symbol_per_session(tmp_path):
    bars = {'AAA': intraday_sessions(breakout_time='10:00', side='LONG')}
    eng, L = _intraday(tmp_path)
    eng.step(bars, _at(bars['AAA'], '10:05'))
    eng.step(bars, _at(bars['AAA'], '10:10'))
    assert eng.step(bars, _at(bars['AAA'], '10:15'))['placed'] == []
    assert L.state['traded_today'] == ['AAA']


def test_intraday_carry_over_is_closed_on_a_new_session(tmp_path):
    bars = {'AAA': intraday_sessions(breakout_time=None)}
    eng, L = _intraday(tmp_path)
    L.state['session'] = '2026-09-01'
    L.positions.append(Position('AAA', 'intraday', 'LONG', 10, 1000.0, 995.0, 995.0,
                                pd.Timestamp('2026-09-01 10:00', tz='Asia/Kolkata'), 1.0,
                                meta={'margin': 2000.0, 'entry_costs': 1.0, 'trade_id': 'I1'}))
    rep = eng.step(bars, _at(bars['AAA'], '09:40'))
    assert rep['closed'] and 'carry-over' in rep['closed'][0][1]
    assert L.positions == []


def test_intraday_daily_loss_limit_flattens(tmp_path):
    bars = {'AAA': intraday_sessions(breakout_time='10:00', side='LONG')}
    eng, L = _intraday(tmp_path)
    eng.step(bars, _at(bars['AAA'], '10:05'))
    eng.step(bars, _at(bars['AAA'], '10:10'))
    assert L.positions
    eng.breakers.on_exit(-2_000)                # an earlier loser today
    rep = eng.step(bars, _at(bars['AAA'], '10:20'))
    assert rep['flattened'] and L.positions == []
    assert any('daily loss' in r for r in rep['breakers'])


def test_partial_daily_bar_is_never_treated_as_final():
    from nsebot.engine.market_view import completed_session
    idx = daily_frame(np.full(5, 100.0), start='2026-09-29')            # last bar = Fri 2026-10-03
    last = pd.Timestamp(idx['datetime'].iloc[-1])
    during = last + pd.Timedelta(hours=11)                              # 11:00 IST that day
    after = last + pd.Timedelta(hours=17, minutes=15)
    assert completed_session(idx, now=during) == pd.Timestamp(idx['datetime'].iloc[-2])
    assert completed_session(idx, now=after) == last


def test_intraday_holiday_writes_no_equity_row(tmp_path):
    class NoData:
        def daily(self, *a, **k):
            return {}

        def intraday(self, *a, **k):
            return {}
    eng, L = _intraday(tmp_path)
    t = iter([pd.Timestamp('2026-10-02 09:40', tz='Asia/Kolkata')] * 5)
    eng.run_session(NoData(), clock=lambda: next(t), sleep=lambda s: None)
    assert L.equity_history().empty
