"""Nov 2026: fixes 1-4 from docs/DIAGNOSIS.md and the move of the whole book
into the momentum sleeve — retired sleeves, the cash sweep and its
deployment, the corporate-action guard, late split anchors, the V4 filter
(kept off), time-weighted returns, and the run-orchestration guards."""

import dataclasses
import importlib.util
import json
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import nsebot.data
import nsebot.listing
from nsebot.__main__ import main
from nsebot.broker.paper import PaperBroker
from nsebot.config import BotConfig
from nsebot.engine import DailyMarket, IntradayEngine, MomentumMarket, SwingEngine
from nsebot.ledger import Ledger
from nsebot.research import diagnosis
from nsebot.research.experiments import Panel, locked_bars
from conftest import daily_frame, intraday_sessions
from test_engines import _at, _dip_then_bounce
from test_momentum import LISTED, FakeProvider, engine, research_world, small_cfg, small_world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load('should_run', ('scripts', 'should_run.py'))
progress = _load('progress', ('scripts', 'progress.py'))
RETIRED = {'status': 'retired', 'since': '2026-11-03', 'sweep_to': 'momentum'}


def _scale(df, k, f):
    """Every price from bar k on multiplied by f (a gap Yahoo has not adjusted)."""
    df = df.copy()
    for col in ('open', 'high', 'low', 'close'):
        df.loc[k:, col] = (df.loc[k:, col] * f).round(1)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Fix 1: retired sleeves
# ═════════════════════════════════════════════════════════════════════════════
def test_the_gate_never_runs_a_retired_flat_sleeve_even_when_triggered_externally():
    now = datetime.fromisoformat('2026-11-04 09:22').replace(tzinfo=gate.IST)
    assert gate.decide('intraday', now, {}, 'workflow_dispatch', RETIRED) == \
        (False, 'sleeve retired in state/sleeves.json and flat')
    swing = {'last_processed': '2026-11-02', 'positions': [{'symbol': 'CUB'}]}
    evening = datetime.fromisoformat('2026-11-04 18:00').replace(tzinfo=gate.IST)
    assert gate.decide('swing', evening, swing, 'schedule', RETIRED)[0]           # exits still run
    assert gate.decide('momentum', evening, {}, 'schedule', {'status': 'active'})[0]


def test_the_gate_reads_the_plan_file(tmp_path, monkeypatch):
    (tmp_path / 'state' / 'intraday').mkdir(parents=True)
    (tmp_path / 'state' / 'intraday' / 'state.json').write_text(json.dumps({'positions': []}))
    (tmp_path / 'state' / 'sleeves.json').write_text(json.dumps({'intraday': RETIRED}))
    out = tmp_path / 'out'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_OUTPUT', str(out))
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'workflow_dispatch')
    assert gate.main(['intraday']) == 0 and out.read_text().startswith('run=false\n')


def test_the_live_plan_retires_swing_and_intraday_into_momentum():
    with open(os.path.join(ROOT, 'state', 'sleeves.json')) as fh:
        plan = json.load(fh)
    assert plan['momentum']['status'] == 'active'
    for mode in ('swing', 'intraday'):
        assert plan[mode]['status'] == 'retired' and plan[mode]['sweep_to'] == 'momentum'


def test_a_retired_swing_sleeve_opens_nothing():
    dip, sig_i = _dip_then_bounce()
    flat = daily_frame(400 * np.exp(np.cumsum(np.full(len(dip), 0.0005))), 300_000.0)
    index = daily_frame(20000 * np.exp(np.cumsum(np.full(len(dip), 0.0008))))
    market = DailyMarket({'DIP': dip, 'FLAT': flat}, index)
    for retired, placed in ((None, ['DIP']), ('sleeve retired', [])):
        L = Ledger(os.path.join(ROOT, 'tests', '__pycache__', 'unused'), 'swing', 50_000)   # never saved
        eng = SwingEngine(BotConfig(), PaperBroker(), L, workdir=ROOT, log=lambda m: None, retired=retired)
        eng.L.save = lambda: None
        eng.L.append_equity = lambda row: None
        rep = eng.run(market, market.dates[sig_i])
        assert [p[0] for p in rep['placed']] == placed
        if retired:
            assert 'sleeve retired' in rep['breakers']
        else:                                       # retired the next day: the queued plan never fills
            eng.retired = 'sleeve retired'
            rep = eng.run(market, market.dates[sig_i + 1])
            assert rep['cancelled'] == [('DIP', 'sleeve retired')] and not rep['filled'] and not L.positions


# ═════════════════════════════════════════════════════════════════════════════
# Fix 2: corporate actions
# ═════════════════════════════════════════════════════════════════════════════
def test_a_demerger_gap_is_held_alerted_and_credited_once_recorded(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    eng, L = engine(tmp_path, cfg)
    for d in dates[60:62]:
        eng.run(MomentumMarket(u, idx, cfg.momentum), d)
    qty = next(p.qty for p in L.positions if p.symbol == 'S0')
    gapped = dict(u, S0=_scale(u['S0'], 62, 0.1))                  # 90% of the value leaves on the 62nd bar
    m2 = MomentumMarket(gapped, idx, cfg.momentum)
    rep = eng.run(m2, dates[62])
    assert any(w.startswith('CORPORATE ACTION? S0') for w in rep['warnings'])
    for d in dates[63:66]:                                          # 65 rebalances: S0 is now under ₹50
        rep = eng.run(m2, d)
    assert rep['rebalanced'] and any(k[0] == 'S0' and 'corporate action' in k[1] for k in rep['kept'])
    assert 'S0' in {p.symbol for p in L.positions}                 # not sold as "no longer eligible"

    prev, ex_open = float(u['S0']['close'].iloc[61]), float(gapped['S0']['open'].iloc[62])
    (tmp_path / 'momentum' / 'corporate_actions.json').write_text(json.dumps([
        {'id': 'S0-demerger', 'type': 'demerger', 'symbol': 'S0', 'ex_date': str(dates[62].date()),
         'new_symbol': 'S0NEW', 'ratio': 1, 'value_per_share': round(prev - ex_open, 2)}]))
    before = L.equity_history()['equity'].iloc[-1]
    rep = eng.run(m2, dates[66])
    assert rep['actions'] and rep['actions'][0][0] == 'S0-demerger'
    new = next(p for p in L.positions if p.symbol == 'S0NEW')
    assert new.qty == qty and new.meta['awaiting_listing']
    assert rep['equity'] - before == pytest.approx(qty * (prev - ex_open), rel=0.05)
    credit = L.closed_trades().iloc[-1]
    assert credit['trigger'] == 'corporate_action' and credit['net_pnl'] == pytest.approx(qty * (prev - ex_open), abs=1)
    assert not next(p for p in L.positions if p.symbol == 'S0').meta.get('ca_suspect')
    assert any('S0NEW' in w and 'until it lists' in w for w in rep['warnings'])
    rep = eng.run(m2, dates[67])
    assert not rep['actions']                                       # applied once only
    rep = eng.run(m2, dates[70])
    assert ('S0', 1 * qty, 'no longer eligible') in rep['selling']  # released to the normal rules


def test_a_split_yahoo_adjusts_late_is_caught_by_the_entry_anchor(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    eng, L = engine(tmp_path / 'split', cfg)
    ref, Lr = engine(tmp_path / 'ref', cfg)
    m1 = MomentumMarket(u, idx, cfg.momentum)
    for d in dates[60:63]:
        eng.run(m1, d)
        ref.run(m1, d)
    qty = next(p.qty for p in L.positions if p.symbol == 'S0')
    late = dict(u, S0=_scale(u['S0'], 63, 0.5))                     # 2:1 from bar 63, history not yet adjusted
    rep = eng.run(MomentumMarket(late, idx, cfg.momentum), dates[63])
    ref.run(m1, dates[63])
    assert any(w.startswith('CORPORATE ACTION? S0') for w in rep['warnings'])
    adjusted = dict(u, S0=_scale(u['S0'], 0, 0.5))                  # a day later Yahoo halves all of it
    rep = eng.run(MomentumMarket(adjusted, idx, cfg.momentum), dates[64])
    ref.run(m1, dates[64])
    assert rep['adjusted'] == [('S0', '2:1', 2 * qty)]
    s0 = next(p for p in L.positions if p.symbol == 'S0')
    assert s0.qty == 2 * qty and not s0.meta.get('ca_suspect')
    assert rep['equity'] == pytest.approx(Lr.equity_history()['equity'].iloc[-1], rel=0.01)


def test_the_live_corporate_action_file_credits_tieramet():
    with open(os.path.join(ROOT, 'state', 'momentum', 'corporate_actions.json')) as fh:
        (a,) = json.load(fh)
    assert (a['type'], a['symbol'], a['new_symbol'], a['ex_date']) == ('demerger', 'BHAGYANGR', 'TIERAMET',
                                                                         '2026-10-08')
    assert a['value_per_share'] * 12 == pytest.approx(4_987, abs=1)


# ═════════════════════════════════════════════════════════════════════════════
# Money moved in, and put to work
# ═════════════════════════════════════════════════════════════════════════════
def test_cash_moved_in_tops_holdings_up_to_an_equal_share_at_the_next_rebalance(tmp_path):
    cfg = small_cfg()
    u, idx, dates = small_world()
    market = MomentumMarket(u, idx, cfg.momentum)
    eng, L = engine(tmp_path, cfg)
    for d in dates[60:62]:
        eng.run(market, d)
    qty = {p.symbol: p.qty for p in L.positions}
    peak = eng.breakers.s['peak_equity']
    assert eng.receive(100_000, 'swing', dates[62]) == 100_000
    assert eng.breakers.s['peak_equity'] == pytest.approx(peak + 100_000)
    rep = eng.run(market, dates[62])
    assert rep['equity'] > 140_000 and L.state['flows'] == [
        {'session': str(dates[62].date()), 'amount': 100_000.0, 'with': 'swing'}]
    assert eng.breakers.state()['latched'] is False
    for d in dates[63:65]:
        eng.run(market, d)
    rep = eng.run(market, dates[65])                               # the rebalance after the money arrived
    assert {t[0] for t in rep['topups']} == {'S0', 'S1'} and not L.state['deploy_inflow']
    rep = eng.run(market, dates[66])
    assert {f[0] for f in rep['filled']} == {'S0 (top-up)', 'S1 (top-up)'}
    eq = rep['equity']
    for p in L.positions:
        assert p.qty > qty[p.symbol]
        assert p.qty * float(u[p.symbol]['close'].iloc[66]) == pytest.approx(eq / 2, rel=0.05)
    assert L.cash < 0.05 * eq


def test_the_momentum_run_sweeps_a_retired_flat_sleeve(tmp_path, monkeypatch):
    monkeypatch.setattr(nsebot.data, 'YahooProvider', FakeProvider)
    monkeypatch.setattr(nsebot.listing, 'nse_equities', lambda *a, **k: (LISTED, 'fake list'))
    monkeypatch.chdir(tmp_path)
    state = tmp_path / 'state'
    swing = Ledger(str(state), 'swing', 50_000)
    swing.state['cash'] = 51_682.0
    swing.state['last_processed'] = '2026-11-02'
    swing.save()
    swing.append_equity({'session': '2026-11-02', 'equity': 51_682.0, 'cash': 51_682.0})
    (state / 'sleeves.json').write_text(json.dumps({'momentum': {'status': 'active'}, 'swing': RETIRED}))
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    for k in ('EMAIL_SENDER', 'EMAIL_PASSWORD', 'EMAIL_RECIPIENT'):
        monkeypatch.delenv(k, raising=False)
    assert main(['momentum', '--state', str(state)]) == 0
    mom = Ledger(str(state), 'momentum', 50_000)
    assert mom.cash == pytest.approx(101_682.0) and mom.contributed == pytest.approx(101_682.0)
    assert mom.state['deploy_inflow'] is False                    # the first run rebalanced and deployed it
    sw = Ledger(str(state), 'swing', 50_000)
    assert sw.cash == 0 and sw.contributed == pytest.approx(-1_682.0)
    assert sw.equity_history()['equity'].iloc[-1] == 0
    text = summary.read_text()
    assert 'Email alerts are OFF' in text and 'moved in from the retired swing sleeve** — invested at today' in text
    assert main(['momentum', '--state', str(state)]) == 0           # same session: nothing moves twice
    assert Ledger(str(state), 'momentum', 50_000).contributed == pytest.approx(101_682.0)


def test_progress_counts_a_transfer_as_neither_profit_nor_loss(tmp_path):
    def bot(mode, rows, flows=()):
        d = tmp_path / 'state' / mode
        d.mkdir(parents=True)
        (d / 'state.json').write_text(json.dumps({'initial_cash': 50_000, 'positions': [],
                                                  'flows': [{'session': s, 'amount': a} for s, a in flows]}))
        lines = ['session,equity,cash,open_positions,open_value'] + [f'{s},{e},{e},0,0' for s, e in rows]
        (d / 'equity.csv').write_text('\n'.join(lines) + '\n')
    bot('momentum', [('2026-11-02', 40_000), ('2026-11-03', 91_000), ('2026-11-04', 95_000)],
        [('2026-11-03', 51_000)])                                   # flat on the 3rd, then the swing cash
    bot('swing', [('2026-11-02', 51_000), ('2026-11-03', 0)], [('2026-11-03', -51_000)])
    bot('intraday', [('2026-11-02', 44_000), ('2026-11-03', 38_000)])
    (tmp_path / 'state' / 'sleeves.json').write_text(json.dumps({'swing': RETIRED}))
    bots = progress.load(str(tmp_path / 'state'))
    mom, swing = bots[0], bots[1]
    day = date(2026, 11, 4)
    assert mom.capital_on(day) == 101_000 and swing.capital_on(day) == -1_000
    assert mom.twr(day) == pytest.approx(0.8 * 1.0 * 95_000 / 91_000 - 1)        # the 51k is no gain
    assert swing.twr(day) == pytest.approx(51_000 / 50_000 - 1)                   # nor is it a loss
    before, after = progress.snapshot(bots, date(2026, 11, 2)), progress.snapshot(bots, date(2026, 11, 3))
    assert after.value - before.value == (91_000 - 40_000) + (0 - 51_000) + (38_000 - 44_000)
    block = progress.block(bots, progress.history(bots))
    assert '| Swing (retired) |' in block and '| Momentum | 2 Nov 2026 | ₹95,000 | −16.48% | +₹4,000 |' in block


# ═════════════════════════════════════════════════════════════════════════════
# V4 (kept off): the live filter matches the research
# ═════════════════════════════════════════════════════════════════════════════
def test_v4_is_off_in_the_deployed_config():
    assert BotConfig().momentum.max_jump == 0.0 and BotConfig().momentum.top_n == 10


def test_v4_never_buys_a_recent_jumper_but_never_sells_one_for_it(tmp_path):
    cfg = small_cfg(max_jump=0.15)
    u, idx, dates = small_world()
    jumpy = dict(u, S0=_scale(u['S0'], 50, 1.2))                   # +20% in one session, 10 bars ago
    rep = engine(tmp_path / 'a', cfg)[0].run(MomentumMarket(jumpy, idx, cfg.momentum), dates[60])
    assert [p[0] for p in rep['placed']] == ['S1', 'S2']
    eng, L = engine(tmp_path / 'b', cfg)
    for d in dates[40:42]:                                          # bought before the jump
        eng.run(MomentumMarket(jumpy, idx, cfg.momentum), d)
    assert 'S0' in {p.symbol for p in L.positions}
    for d in dates[42:61]:
        eng.run(MomentumMarket(jumpy, idx, cfg.momentum), d)
    assert 'S0' in {p.symbol for p in L.positions}


def test_live_v4_trades_what_the_research_filter_measured(tmp_path):
    u, idx = research_world()
    rng = np.random.default_rng(4)
    for s in rng.choice(sorted(u), 25, replace=False):             # one-day jumps the filter must see
        u[s] = _scale(u[s], int(rng.integers(300, 500)), 1.18)
    P = Panel(u)
    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    r = c / c.shift(1) - 1
    nojump = (r.rolling(63, min_periods=40).max() < 0.15).to_numpy()
    buy_ok, sell_ok = locked_bars(P)
    research, _, _ = diagnosis.rotate(P, mom, P.liquid, n=10, buy_ok=buy_ok, sell_ok=sell_ok, buy_filter=nojump)
    start = int(np.argmax(np.isfinite(mom).sum(axis=1) >= 20))
    cfg = BotConfig()
    cfg = dataclasses.replace(cfg, momentum=dataclasses.replace(cfg.momentum, max_jump=0.15))
    eng, L = engine(tmp_path, cfg)
    market = MomentumMarket(u, idx, cfg.momentum)
    for d in P.dates[start:]:
        eng.run(market, d)
    eq = L.equity_history().set_index('session')['equity']
    assert np.abs(eq.to_numpy() - research.loc[P.dates[start:]].round(2).to_numpy()).max() <= 0.011


# ═════════════════════════════════════════════════════════════════════════════
# Fixes 3 and 4: alerts and run orchestration
# ═════════════════════════════════════════════════════════════════════════════
def test_every_bot_checks_out_the_branch_tip_not_the_queued_commit():
    for f in ('main.yml', 'momentum.yml', 'intraday.yml'):
        text = open(os.path.join(ROOT, '.github', 'workflows', f)).read()
        assert 'uses: actions/checkout@v4\n        with:' in text and 'ref: ${{ github.ref_name }}' in text, f


def test_an_intraday_session_squares_off_before_the_job_is_killed(tmp_path):
    bars = intraday_sessions(breakout_time='10:00', side='LONG')
    now = {'t': _at(bars, '10:02')}

    class Feed:
        def daily(self, *a, **k):
            return {}

        def intraday(self, symbols, *a, **k):
            return {s: bars for s in symbols}

    def sleep(s):
        now['t'] += pd.Timedelta(minutes=5)

    L = Ledger(str(tmp_path), 'intraday', 50_000)
    eng = IntradayEngine(BotConfig(), PaperBroker(slippage_bps=5), L, workdir=str(tmp_path),
                         log=lambda m: None, universe=['AAA'])
    eng.run_session(Feed(), clock=lambda: now['t'], sleep=sleep, deadline=now['t'] + pd.Timedelta(minutes=25))
    t = L.closed_trades()
    assert len(t) == 1 and t['exit_reason'].iloc[0] == 'square-off before the job time limit'
    assert pd.Timestamp(t['exit_time'].iloc[0]).time() < pd.Timestamp('11:00').time() and L.positions == []
