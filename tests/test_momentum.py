import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

import nsebot.data
import nsebot.listing
from nsebot.__main__ import main
from nsebot.broker.base import Fill
from nsebot.broker.paper import PaperBroker
from nsebot.config import BotConfig, momentum_breakers
from nsebot.engine import MomentumEngine, MomentumMarket
from nsebot.ledger import Ledger
from nsebot.listing import ListUnavailable, nse_equities, parse_equity_list
from nsebot.notify import fy_tax_estimate, momentum_report
from nsebot.research.experiments import Panel, locked_bars, simulate_rotation
from nsebot.risk import CircuitBreakers
from nsebot.signals.momentum import MomentumRanker
from conftest import daily_frame


# ═════════════════════════════════════════════════════════════════════════════
# Worlds
# ═════════════════════════════════════════════════════════════════════════════
def _frame(dates, close, volume=1_000_000.0):
    close = np.round(np.asarray(close, dtype=float), 1)         # on every NSE tick grid below ₹5,000
    open_ = np.r_[close[0], close[:-1]]                         # gapless: opens at the prior close
    return pd.DataFrame({'datetime': dates, 'open': open_,
                         'high': np.round(np.maximum(open_, close) * 1.01, 1),
                         'low': np.round(np.minimum(open_, close) * 0.99, 1),
                         'close': close, 'volume': np.broadcast_to(volume, close.shape).astype(float)})


def _index(dates):
    return pd.DataFrame({'datetime': dates, 'open': 1e4, 'high': 1e4, 'low': 1e4, 'close': 1e4,
                         'volume': 0.0})


def research_world(n_sym=100, T=520, illiquid=(0, 1, 2, 3), drying=(4, 5, 6), seed=6, switch=380):
    """100 names: 4 never liquid; 3 early leaders whose trading dries up (sold
    as no longer eligible); the rest change trend at `switch`, so the 12-1
    ranking turns over and holdings drop out of the top 30."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2022-01-03', periods=T)
    u = {}
    for i in range(n_sym):
        d1, d2 = rng.normal(0.0005, 0.0015, 2)
        if i in drying:
            d1 = 0.003
        drift = np.r_[np.full(switch, d1), np.full(T - switch, d2)]
        close = rng.uniform(150, 450) * np.exp(np.cumsum(rng.normal(drift, 0.022)))
        vol = np.full(T, 1_000.0 if i in illiquid else 1_000_000.0)
        if i in drying:
            vol[400 + 15 * (i - 4):] = 1_000.0
        u[f'M{i:02d}'] = _frame(dates, close, vol)
    return u, _index(dates)


def small_cfg(**kw):
    """Same mechanics, short windows: top 2, sold once out of the top 2."""
    cfg = BotConfig()
    m = dataclasses.replace(cfg.momentum, **{'top_n': 2, 'band_mult': 1, 'lookback': 40, 'skip': 5, **kw})
    return dataclasses.replace(cfg, momentum=m)


def small_world(n=6, T=120, price=300.0):
    """S0 strongest ... S5 weakest, noise-free so the ranking is certain."""
    dates = pd.bdate_range('2024-01-01', periods=T)
    u = {f'S{i}': _frame(dates, price * np.exp(np.cumsum(np.full(T, 0.004 - 0.0015 * i)))) for i in range(n)}
    return u, _index(dates), dates


def engine(root, cfg=None, cash=50_000.0, broker=None):
    L = Ledger(str(root), 'momentum', cash)
    return MomentumEngine(cfg or BotConfig(), broker or PaperBroker(slippage_bps=0), L,
                          workdir=str(root), log=lambda m: None), L


def seed_position(eng, sym, qty, price, when):
    eng.book.open(Fill(sym, 'BUY', qty, price, 0.0, pd.Timestamp(when)),
                  {'symbol': sym, 'side': 'LONG', 'stop': 0.0, 'signal_time': str(when), 'meta': {}},
                  pd.Timestamp(when), 0.0)


def lock(df, date, up=True):
    """Freeze one bar at a single price, above (up) or below the prior close."""
    i = int(np.flatnonzero(df['datetime'] == date)[0])
    px = round(float(df['close'].iloc[i - 1]) * (1.05 if up else 0.95), 1)
    df.loc[i, ['open', 'high', 'low', 'close']] = px


# ═════════════════════════════════════════════════════════════════════════════
# Faithfulness: the live engine is the researched design
# ═════════════════════════════════════════════════════════════════════════════
def test_ranker_matches_the_research_panel():
    u, _ = research_world()
    P = Panel(u)
    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    for t in (260, 410, 519):
        table = MomentumRanker().rank(u, P.dates[t])
        for j, s in enumerate(P.symbols):
            assert bool(table.loc[s, 'eligible']) == bool(P.liquid[t, j]), (t, s)
            a, b = table.loc[s, 'score'], mom[t, j]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-12), (t, s)


def test_ranker_matches_the_research_panel_with_gaps_listings_and_delistings():
    """Suspensions punch holes in the calendar, late listings start it later,
    a delisting ends it early: the ranker's calendar shortcut must still agree
    with the research panel exactly."""
    u, _ = research_world()
    rng = np.random.default_rng(9)
    for i, s in enumerate(sorted(u)):
        df = u[s]
        if i % 5 == 0:                                              # random suspended days
            df = df.drop(rng.choice(df.index[260:], 15, replace=False))
        if i % 7 == 0:                                              # listed later
            df = df.iloc[120:]
        if i % 11 == 0:                                             # delisted early
            df = df.iloc[:450]
        u[s] = df.reset_index(drop=True)
    P = Panel(u)
    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    for t in (300, 449, 460, 519):
        table = MomentumRanker().rank(u, P.dates[t])
        for j, s in enumerate(P.symbols):
            a, b = table.loc[s, 'score'], mom[t, j]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-12), (t, s)
            assert bool(table.loc[s, 'eligible']) == bool(P.liquid[t, j]), (t, s)


def test_live_engine_trades_exactly_what_the_research_simulator_measured(tmp_path):
    """Run the live engine session by session and the research simulator over
    the same history: same equity every day, same total costs. (Gapless,
    on-tick prices and zero slippage remove the only two places the live path
    differs by design: sizing on the close, and tick snapping.)"""
    u, idx = research_world()
    P = Panel(u)
    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    assert float(np.nanmax(c.to_numpy())) < 5_000                       # one tick grid throughout
    buy_ok, sell_ok = locked_bars(P)
    research, research_costs = simulate_rotation(P, mom, P.liquid, n=10, buy_ok=buy_ok, sell_ok=sell_ok)
    start = int(np.argmax(np.isfinite(mom).sum(axis=1) >= 20))
    valid = (P.liquid & np.isfinite(mom)).sum(axis=1)[start:]
    assert valid.min() >= 20                                             # the data-fault guard never fires

    eng, L = engine(tmp_path)
    market = MomentumMarket(u, idx)
    reps = [eng.run(market, d) for d in P.dates[start:]]

    eq = L.equity_history().set_index('session')['equity']
    expected = research.loc[P.dates[start:]].round(2)
    assert len(eq) == len(expected)
    assert np.abs(eq.to_numpy() - expected.to_numpy()).max() <= 0.011
    rebalanced = [d for d, r in zip(P.dates[start:], reps) if r['rebalanced']]
    assert rebalanced[:-1] == list(P.dates[start:-1:5])[:len(rebalanced) - 1]
    trades = L.closed_trades()
    engine_costs = float(trades['costs'].sum()) + sum(p.meta['entry_costs'] for p in L.positions)
    assert engine_costs == pytest.approx(research_costs, abs=0.05)
    assert len(trades) >= 8 and len(L.positions) == 10
    reasons = set(trades['exit_reason'].str.split(' \\(').str[0])
    assert 'rebalance: no longer eligible' in reasons and 'rebalance: fell out of the top 30' in reasons


# ═════════════════════════════════════════════════════════════════════════════
# Schedule, persistence, catch-up
# ═════════════════════════════════════════════════════════════════════════════
def test_rebalances_on_the_first_run_then_every_five_sessions(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    market = MomentumMarket(u, idx, cfg.momentum)
    eng, L = engine(tmp_path, cfg)
    reps = [eng.run(market, d) for d in dates[60:71]]
    assert [r['rebalanced'] for r in reps] == [True, False, False, False, False,
                                                True, False, False, False, False, True]
    assert [p[0] for p in reps[0]['placed']] == ['S0', 'S1']
    assert {p.symbol for p in L.positions} == {'S0', 'S1'}
    assert eng.run(market, dates[70])['status'].startswith('session')       # idempotent


def test_orders_survive_a_fresh_checkout_and_a_missed_run_fills_at_the_right_open(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    market = MomentumMarket(u, idx, cfg.momentum)
    eng, _ = engine(tmp_path, cfg)
    eng.run(market, dates[60])
    eng2, L2 = engine(tmp_path, cfg)                         # new process, state read from disk
    assert {p['spec']['symbol'] for p in L2.state['pending']} == {'S0', 'S1'}
    rep = eng2.run(market, dates[63])                        # runs for 61 and 62 were missed
    assert {f[0] for f in rep['filled']} == {'S0', 'S1'}
    s0 = next(p for p in L2.positions if p.symbol == 'S0')
    assert s0.entry_time == dates[61] and s0.entry_price == float(u['S0']['open'].iloc[61])
    assert not rep['rebalanced'] and L2.state['sessions_since_rebalance'] == 3
    assert eng2.run(market, dates[65])['rebalanced']          # 3 + 2 sessions: due


# ═════════════════════════════════════════════════════════════════════════════
# Fills the exchange could not have given
# ═════════════════════════════════════════════════════════════════════════════
def test_a_buy_frozen_at_the_upper_circuit_is_cancelled(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    lock(u['S0'], dates[61], up=True)
    market = MomentumMarket(u, idx, cfg.momentum)
    eng, L = engine(tmp_path, cfg)
    eng.run(market, dates[60])
    rep = eng.run(market, dates[61])
    assert ('S0', f'frozen at its upper circuit on {dates[61].date()} (no sellers)') in rep['cancelled']
    assert [p.symbol for p in L.positions] == ['S1']


def test_a_sale_frozen_at_the_lower_circuit_waits_and_buys_use_only_real_cash(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    lock(u['S5'], dates[61], up=False)
    market = MomentumMarket(u, idx, cfg.momentum)
    eng, L = engine(tmp_path, cfg)
    seed_position(eng, 'S5', 100, float(u['S5']['close'].iloc[59]), dates[59])
    rep0 = eng.run(market, dates[60])
    assert rep0['selling'][0][:2] == ('S5', 100) and len(rep0['placed']) == 2
    planned = {p[0]: p[1] for p in rep0['placed']}
    rep1 = eng.run(market, dates[61])
    assert rep1['kept'][0][0] == 'S5' and 'lower circuit' in rep1['kept'][0][1]
    held = {p.symbol: p.qty for p in L.positions}
    assert held['S5'] == 100                                 # still held, re-checked next rebalance
    assert held['S1'] < planned['S1']                        # its sale cash never arrived
    assert L.cash >= -50                                     # at most a fee below zero, as researched


# ═════════════════════════════════════════════════════════════════════════════
# Live-data guards: splits, late bars, stale prices
# ═════════════════════════════════════════════════════════════════════════════
def test_a_split_rescales_the_holding_instead_of_booking_a_fake_loss(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    m1 = MomentumMarket(u, idx, cfg.momentum)
    eng, L = engine(tmp_path / 'split', cfg)
    ref, Lr = engine(tmp_path / 'ref', cfg)
    for d in dates[60:62]:
        eng.run(m1, d)
        ref.run(m1, d)
    qty = next(p.qty for p in L.positions if p.symbol == 'S0')
    # A 2:1 split before the next session: Yahoo now shows every S0 price halved,
    # history included.
    u2 = dict(u, S0=u['S0'].assign(**{c: u['S0'][c] / 2 for c in ('open', 'high', 'low', 'close')}))
    rep = eng.run(MomentumMarket(u2, idx, cfg.momentum), dates[62])
    ref.run(m1, dates[62])
    assert rep['adjusted'] == [('S0', '2:1', 2 * qty)]
    s0 = next(p for p in L.positions if p.symbol == 'S0')
    assert s0.qty == 2 * qty
    assert rep['equity'] == pytest.approx(Lr.equity_history()['equity'].iloc[-1], abs=0.05)


def test_a_data_revision_is_flagged_not_mistaken_for_a_split(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    eng, L = engine(tmp_path, cfg)
    m1 = MomentumMarket(u, idx, cfg.momentum)
    for d in dates[60:62]:
        eng.run(m1, d)
    qty = next(p.qty for p in L.positions if p.symbol == 'S0')
    bad = u['S0'].copy()
    bad.loc[bad['datetime'] == dates[61], 'close'] /= 1.37          # one field revised, not a clean ratio
    rep = eng.run(MomentumMarket(dict(u, S0=bad), idx, cfg.momentum), dates[62])
    assert not rep['adjusted'] and next(p.qty for p in L.positions if p.symbol == 'S0') == qty
    assert any('not a clean split' in w for w in rep['warnings'])


def test_clean_split_ratios():
    from nsebot.engine.momentum import clean_split_ratio
    assert [str(clean_split_ratio(r)) for r in (2.0, 1.5, 4 / 3, 10.02, 0.1, 2.5, 6 / 5)] == \
        ['2', '3/2', '4/3', '10', '1/10', '5/2', '6/5']
    assert clean_split_ratio(1.37) is None and clean_split_ratio(2.06) is None


def test_late_yahoo_data_keeps_an_order_queued_then_fills_at_the_right_open(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    eng, L = engine(tmp_path, cfg)
    eng.run(MomentumMarket(u, idx, cfg.momentum), dates[60])
    late = dict(u, S0=u['S0'][u['S0']['datetime'] <= dates[60]])          # S0's bar for 61 not in yet
    rep = eng.run(MomentumMarket(late, idx, cfg.momentum), dates[61])
    assert ('S0', f'no data for {dates[61].date()} yet — order kept queued') in rep['kept']
    assert [p.symbol for p in L.positions] == ['S1']
    rep = eng.run(MomentumMarket(u, idx, cfg.momentum), dates[62])
    s0 = next(p for p in L.positions if p.symbol == 'S0')
    assert s0.entry_time == dates[61] and s0.entry_price == float(u['S0']['open'].iloc[61])


def test_an_order_whose_data_never_arrives_gives_up(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    eng, L = engine(tmp_path, cfg)
    eng.run(MomentumMarket(u, idx, cfg.momentum), dates[60])
    gone = MomentumMarket(dict(u, S0=u['S0'][u['S0']['datetime'] <= dates[60]]), idx, cfg.momentum)
    reps = [eng.run(gone, d) for d in dates[61:65]]
    assert ('S0', f'did not trade on {dates[61].date()}') in reps[3]['cancelled']
    assert not any(p['spec']['symbol'] == 'S0' for p in L.state['pending'] if 'spec' in p)


def test_a_holding_without_todays_bar_is_kept_not_sold_and_stale_prices_are_flagged(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    eng, L = engine(tmp_path, cfg)
    seed_position(eng, 'S5', 100, float(u['S5']['close'].iloc[54]), dates[54])
    stale = dict(u, S5=u['S5'][u['S5']['datetime'] <= dates[54]])        # nothing since day 54
    rep = eng.run(MomentumMarket(stale, idx, cfg.momentum), dates[60])
    assert rep['rebalanced'] and not rep['selling']
    assert ('S5', f'no bar for {dates[60].date()} — held, re-checked next rebalance') in rep['kept']
    assert any(w.startswith('S5: no price since') for w in rep['warnings'])


# ═════════════════════════════════════════════════════════════════════════════
# Breakers, data faults, paralysis
# ═════════════════════════════════════════════════════════════════════════════
def test_the_55pct_latch_stops_buys_but_never_sales(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    market = MomentumMarket(u, idx, cfg.momentum)
    L = Ledger(str(tmp_path), 'momentum', 50_000)
    L.state['breakers'] = {'peak_equity': 200_000.0}          # equity is 75% below this peak
    eng = MomentumEngine(cfg, PaperBroker(slippage_bps=0), L, workdir=str(tmp_path), log=lambda m: None)
    seed_position(eng, 'S5', 100, float(u['S5']['close'].iloc[59]), dates[59])
    rep = eng.run(market, dates[60])
    assert rep['buys_blocked'] and not rep['placed']
    assert [s[0] for s in rep['selling']] == ['S5']
    assert (tmp_path / 'BREAKER_TRIPPED_momentum').exists()
    assert momentum_breakers().max_drawdown_pct == 0.55


def test_too_few_eligible_stocks_skips_the_rebalance_instead_of_selling_the_book(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    thin = {s: (df if s in ('S0', 'S1', 'S2') else df.assign(volume=10.0)) for s, df in u.items()}
    eng, L = engine(tmp_path, cfg)
    rep = eng.run(MomentumMarket(thin, idx, cfg.momentum), dates[60])
    assert not rep['rebalanced'] and rep['warnings'][0].startswith('DATA: only 3 eligible')
    assert L.state.get('sessions_since_rebalance') is None   # still due
    assert eng.run(MomentumMarket(u, idx, cfg.momentum), dates[61])['rebalanced']


def test_a_lower_eligible_count_that_persists_is_accepted_as_the_market(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    eng, L = engine(tmp_path, cfg)
    L.state['last_eligible'] = 20                            # last time: 20; now 6 < half of it
    market = MomentumMarket(u, idx, cfg.momentum)
    reps = [eng.run(market, d) for d in dates[60:63]]
    assert [r['rebalanced'] for r in reps] == [False, False, True]
    assert 'accepted as the market' in reps[2]['warnings'][0]


def test_a_book_that_can_afford_nothing_says_it_is_not_trading(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world(price=40_000.0)               # every share costs more than a slot's ₹25k
    market = MomentumMarket(u, idx, cfg.momentum)
    eng, _ = engine(tmp_path, cfg)
    reps = [eng.run(market, d) for d in dates[60:66]]
    assert not any(r['placed'] for r in reps)
    assert not reps[0]['warnings'] and reps[5]['warnings'][0].startswith('NOT TRADING: 2 rebalances')


def test_daily_loss_limit_zero_means_off(tmp_path):
    b = CircuitBreakers(momentum_breakers(), 'momentum', workdir=str(tmp_path))
    b.start_session('2026-01-05')
    b.on_exit(-40_000)
    assert b.check(60_000).entries_allowed


def test_momentum_is_paper_only(tmp_path):
    class Live(PaperBroker):
        is_live = True
    with pytest.raises(ValueError, match='paper-only'):
        engine(tmp_path, broker=Live())


def test_engine_and_market_must_share_one_config(tmp_path):
    u, idx, dates = small_world()
    eng, _ = engine(tmp_path, small_cfg())
    with pytest.raises(ValueError, match='configs differ'):
        eng.run(MomentumMarket(u, idx), dates[60])


# ═════════════════════════════════════════════════════════════════════════════
# NSE list
# ═════════════════════════════════════════════════════════════════════════════
def _equity_l(n=320):
    rows = ['SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING']
    rows += [f'X{i:04d},Co {i},EQ,01-JAN-2010' for i in range(n)] + ['SMEX,SME,SM,01-JAN-2020']
    return '\n'.join(rows) + '\n'


def test_list_is_downloaded_and_saved(tmp_path):
    path = str(tmp_path / 'list.json')
    syms, note = nse_equities(path, get=lambda url: _equity_l(), today='2026-10-05')
    assert len(syms) == 320 and 'SMEX' not in syms and note.startswith('NSE EQUITY_L')
    assert json.load(open(path))['fetched'] == '2026-10-05'


def test_list_falls_back_to_a_recent_saved_copy_then_refuses_a_stale_one(tmp_path):
    path = str(tmp_path / 'list.json')
    nse_equities(path, get=lambda url: _equity_l(), today='2026-10-05')
    syms, note = nse_equities(path, get=lambda url: None, today='2026-10-20')
    assert len(syms) == 320 and 'saved NSE list from 2026-10-05' in note
    with pytest.raises(ListUnavailable, match='40 days old'):
        nse_equities(path, get=lambda url: None, today='2026-11-14')
    with pytest.raises(ListUnavailable, match='no saved list'):
        nse_equities(str(tmp_path / 'none.json'), get=lambda url: None)


def test_an_error_page_is_never_the_market():
    assert parse_equity_list('<html>Access Denied</html>', ('EQ',)) == []
    assert len(parse_equity_list(_equity_l(), ('EQ', 'BE', 'BZ'))) == 320


# ═════════════════════════════════════════════════════════════════════════════
# Report and tax estimate
# ═════════════════════════════════════════════════════════════════════════════
def test_fy_tax_estimate_offsets_short_term_losses_against_long_term_gains():
    trades = pd.DataFrame({
        'entry_time': ['2026-05-01', '2026-06-01', '2025-01-02', '2025-06-01'],
        'exit_time': ['2026-06-01', '2026-07-01', '2026-08-01', '2026-03-01'],   # last: previous FY
        'net_pnl': [10_000.0, -2_000.0, 200_000.0, 99_999.0]})
    tx = fy_tax_estimate(trades, '2026-10-05')
    assert tx['fy'] == 'FY27' and tx['short'] == 8_000 and tx['long'] == 200_000
    assert tx['tax'] == pytest.approx(1.04 * (0.20 * 8_000 + 0.125 * 75_000))
    loss = fy_tax_estimate(trades.assign(net_pnl=[-50_000.0, 0.0, 200_000.0, 0.0]), '2026-10-05')
    assert loss['tax'] == pytest.approx(1.04 * 0.125 * 25_000)


def test_momentum_report_renders(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    eng, L = engine(tmp_path, cfg)
    market = MomentumMarket(u, idx, cfg.momentum)
    eng.run(market, dates[60])
    rep = eng.run(market, dates[61])
    md = momentum_report(rep, 50_000, L.closed_trades())
    assert '## nsebot momentum' in md and '### Holdings' in md and 'rough tax' in md
    assert 'S0' in md and 'next rebalance in 4 session(s)' in md


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
LISTED = [f'N{i:02d}' for i in range(30)]


class FakeProvider:
    def __init__(self, *a, **k):
        pass

    def daily(self, symbols, lookback_days=400):
        out = {}
        for k, s in enumerate(symbols):
            if s == '^NSEI':
                out[s] = daily_frame(20000 * np.exp(np.cumsum(np.full(320, 0.0006))), start='2025-07-01')
            else:
                rng = np.random.default_rng(k)
                out[s] = daily_frame(300 * np.exp(np.cumsum(rng.normal(0.001, 0.015, 320))), 400_000.0,
                                     start='2025-07-01')
        return out


def test_momentum_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(nsebot.data, 'YahooProvider', FakeProvider)
    monkeypatch.setattr(nsebot.listing, 'nse_equities', lambda *a, **k: (LISTED, 'fake list'))
    monkeypatch.chdir(tmp_path)
    state = str(tmp_path / 'state')
    assert main(['momentum', '--state', state]) == 0
    L = Ledger(state, 'momentum', 50_000)
    assert L.state['last_processed'] and len(L.state['pending']) == 10
    assert main(['momentum', '--state', state]) == 0                     # same session: no-op
    assert len(Ledger(state, 'momentum', 50_000).equity_history()) == 1
    assert main(['status', '--state', state]) == 0


def test_momentum_cli_refuses_without_a_list_or_enough_data(tmp_path, monkeypatch):
    def no_list(*a, **k):
        raise ListUnavailable('NSE unreachable and no saved list exists')
    monkeypatch.setattr(nsebot.data, 'YahooProvider', FakeProvider)
    monkeypatch.setattr(nsebot.listing, 'nse_equities', no_list)
    monkeypatch.chdir(tmp_path)
    assert main(['momentum', '--state', str(tmp_path / 'state')]) == 3
    monkeypatch.setattr(nsebot.listing, 'nse_equities', lambda *a, **k: (LISTED + [f'Z{i}' for i in range(40)], 'x'))

    class Partial(FakeProvider):
        def daily(self, symbols, lookback_days=400):
            return {s: f for s, f in super().daily(symbols, lookback_days).items() if not s.startswith('Z')}
    monkeypatch.setattr(nsebot.data, 'YahooProvider', Partial)
    assert main(['momentum', '--state', str(tmp_path / 'state')]) == 3      # 30 of 70 = 43% resolved
    assert not (tmp_path / 'state' / 'momentum' / 'state.json').exists()


def test_momentum_replay_study_runs_on_fake_sources(tmp_path):
    from nsebot.research import momentum_replay
    u, idx = research_world(n_sym=40, T=330, switch=300)

    def fetch(symbols, start, end):
        return {'^NSEI': idx} if symbols == ['^NSEI'] else {s: u[s] for s in symbols if s in u}

    report = momentum_replay.run(str(tmp_path), fetch=fetch, get_list=lambda: (sorted(u), 'fake list'))
    assert '| research B4 | live engine |' in report and 'rebalances' in report


def test_phase_runs_delay_the_first_rebalance_and_the_study_renders(tmp_path):
    from nsebot.research import momentum_phase
    u, idx = research_world(n_sym=40, T=330, switch=300)
    P = Panel(u)
    curves = momentum_phase.phase_runs(P)
    first_trade = [int(np.argmax(c.to_numpy() != 50_000.0)) for c in curves]
    assert first_trade == sorted(first_trade) and len(set(first_trade)) == momentum_phase.PHASES

    def fetch(symbols, start, end):
        return {'^NSEI': idx} if symbols == ['^NSEI'] else {s: u[s] for s in symbols if s in u}

    report = momentum_phase.run(str(tmp_path), fetch=fetch, get_list=lambda: (sorted(u), 'fake list'))
    assert 'Nifty sessions (live)' in report and '0 stock-trading dates are not Nifty sessions' in report
