"""Why does the full-engine backtest differ from the event study?

    python -m nsebot.research.attribution --out research_results

The S4b event study took every signal, at a fixed ₹15k notional, with no
slippage, no capacity limits and no regime scaling. The deployed engine
adds all of those. This runs the REAL SwingEngine under a ladder of
configurations, removing one difference at a time, so the gap can be
attributed rather than guessed:

  P0  parity     every signal, ₹15k fixed notional, no slippage, no regime,
                 unlimited slots — must reproduce the event study, or the
                 engine has a bug
  P1  + 5 bps slippage and tick snapping
  P2  + risk-based sizing (half-Kelly, caps) instead of ₹15k fixed
  P3  + capacity (5 slots, 2/sector, heat cap, regime entry budget)
  P4  + regime size dial  = the deployed configuration
  P5  deployed, but shallowest dip first (ranking check)

Research-only: patches are applied in-process to a throwaway engine; the
deployed modules are not modified.
"""

import argparse
import dataclasses
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from ..broker.paper import PaperBroker
from ..config import BotConfig
from ..engine import swing as swing_mod
from ..engine.market_view import DailyMarket
from ..engine.swing import SwingEngine
from ..ledger import Ledger
from ..risk.allocator import AllocationPlan, Order
from ..risk.sizing import SizeDecision

logger = logging.getLogger(__name__)
IS_FRACTION = 0.6
FIXED_NOTIONAL = 15_000.0


def _fixed_notional_allocate(signals, open_positions, cfg, **kw):
    """Research parity: every signal, ₹15k each, no caps."""
    plan = AllocationPlan()
    held = {p.symbol for p in open_positions}
    for s in signals:
        if s.symbol in held:
            plan.declined.append((s.symbol, 'already held'))
            continue
        qty = max(int(FIXED_NOTIONAL // s.ref_price), 1)
        size = SizeDecision(qty, qty * s.risk_per_share, qty * s.ref_price,
                            qty * s.risk_per_share / max(kw.get('equity', 1), 1), 'fixed ₹15k')
        plan.orders.append(Order(s.symbol, s.side, qty, s.ref_price, s.stop, size, s, 'CNC'))
        held.add(s.symbol)
    return plan


def make_notional_allocate(target_pct=0.20, min_notional=8_000.0):
    """Candidate fix: size by NOTIONAL (a fixed share of equity per slot, like
    the event study's ₹15k), cap the risk at cfg.risk_cap_pct, and SKIP a trade
    that cannot reach min_notional rather than shrinking it into fixed-cost
    territory. Capacity limits (slots, sector, daily budget, cash) still apply."""
    def _allocate(signals, open_positions, cfg, *, equity, cash, edge=None, regime_mult=1.0,
                  breaker_mult=1.0, max_new=None, sector_of=lambda s: 'OTHER', **kw):
        plan = AllocationPlan()
        held = {p.symbol for p in open_positions}
        sectors = {}
        for p in open_positions:
            sectors[sector_of(p.symbol)] = sectors.get(sector_of(p.symbol), 0) + 1
        slots = max(cfg.max_positions - len(open_positions), 0)
        budget = slots if max_new is None else min(slots, int(max_new))
        cash_left = float(cash)
        for s in signals:
            if s.symbol in held or budget <= 0:
                continue
            sec = sector_of(s.symbol)
            if sectors.get(sec, 0) >= cfg.max_per_sector:
                continue
            notional = min(equity * target_pct * regime_mult * breaker_mult, cash_left)
            qty = int(notional // s.ref_price)
            max_risk_qty = int(equity * cfg.risk_cap_pct // max(s.risk_per_share, 1e-9))
            qty = min(qty, max_risk_qty)
            if qty < 1 or qty * s.ref_price < min_notional:
                plan.declined.append((s.symbol, 'below minimum economic notional'))
                continue
            size = SizeDecision(qty, qty * s.risk_per_share, qty * s.ref_price,
                                qty * s.risk_per_share / equity, 'notional')
            plan.orders.append(Order(s.symbol, s.side, qty, s.ref_price, s.stop, size, s, 'CNC'))
            held.add(s.symbol)
            sectors[sec] = sectors.get(sec, 0) + 1
            cash_left -= qty * s.ref_price
            budget -= 1
        return plan
    return _allocate


class _Regime:
    """Wraps a DailyMarket. mode: 'real' (deployed), 'budget' (keep the regime's
    daily entry budget, drop its size dial), 'flat' (no regime at all)."""

    def __init__(self, market, mode='real', reverse=False):
        self.m, self.mode, self.reverse = market, mode, reverse

    def __getattr__(self, k):
        return getattr(self.m, k)

    def regime_at(self, asof):
        r = dict(self.m.regime_at(asof))
        if self.mode == 'flat':
            return {'score': 1.0, 'size_mult': 1.0, 'max_new_entries': 999, 'breadth': None}
        if self.mode == 'budget':
            r['size_mult'] = 1.0
        return r

    def signals(self, asof):
        s = self.m.signals(asof)
        return list(reversed(s)) if self.reverse else s


def run_variant(market, name, cfg, slippage_bps, fixed_notional, regime_mode, reverse=False,
                initial_cash=50_000.0, warmup=200, allocator=None):
    root = tempfile.mkdtemp(prefix='nsebot_attr_')
    ledger = Ledger(root, 'swing', 1e12 if fixed_notional else initial_cash)
    engine = SwingEngine(cfg, PaperBroker(slippage_bps=slippage_bps), ledger, workdir=root,
                         log=lambda m: None)
    view = _Regime(market, mode=regime_mode, reverse=reverse)
    original = swing_mod.allocate
    if fixed_notional:
        swing_mod.allocate = _fixed_notional_allocate
    elif allocator is not None:
        swing_mod.allocate = allocator
    try:
        for d in market.dates[warmup:]:
            engine.run(view, d)
    finally:
        swing_mod.allocate = original
    eq = ledger.equity_history()
    eq = pd.Series(eq['equity'].astype(float).to_numpy(), index=pd.to_datetime(eq['session']))
    t = ledger.closed_trades()
    if len(t):
        t['exit_time'] = pd.to_datetime(t['exit_time'])
        notional = t['entry_price'].astype(float) * t['qty'].astype(float)
        t['net_pct'] = t['net_pnl'].astype(float) / notional
        t['gross_pct'] = t['gross_pnl'].astype(float) / notional
    return {'name': name, 'eq': eq, 'trades': t, 'fixed': fixed_notional}


def _seg(res, a, b):
    t = res['trades']
    t = t[(t['exit_time'] >= a) & (t['exit_time'] < b)] if len(t) else t
    out = {'n': len(t)}
    if len(t):
        out.update({'gross_win': (t['gross_pnl'] > 0).mean(), 'net_win': (t['net_pnl'] > 0).mean(),
                    'gross_pct': t['gross_pct'].mean(), 'net_pct': t['net_pct'].mean(),
                    'net_r': t['net_r'].astype(float).mean(),
                    't': t['net_pct'].mean() / (t['net_pct'].std(ddof=1) / np.sqrt(len(t)))
                    if len(t) > 1 and t['net_pct'].std(ddof=1) > 0 else np.nan,
                    'avg_notional': float((t['entry_price'] * t['qty']).mean())})
    if not res['fixed']:
        eq = res['eq'].loc[a:b]
        if len(eq) > 5:
            out['ret'] = eq.iloc[-1] / eq.iloc[0] - 1
            out['maxdd'] = float((eq / eq.cummax() - 1).min())
    return out


def _fmt(s):
    if s['n'] == 0:
        return '0 | — | — | — | — | — | — | — |'
    ret = f"{s['ret'] * 100:+.1f}% / {s['maxdd'] * 100:.1f}%" if 'ret' in s else 'n/a (fixed ₹15k)'
    return (f"{s['n']} | {s['gross_win'] * 100:.0f}% | {s['net_win'] * 100:.0f}% | "
            f"{s['gross_pct'] * 100:+.2f}% | {s['net_pct'] * 100:+.2f}% | {s['t']:+.1f} | "
            f"₹{s['avg_notional']:,.0f} | {ret} |")


def run(out_dir):
    from ..data import YahooProvider
    from ..universe import INDEX_SYMBOL, SWING_UNIVERSE
    os.makedirs(out_dir, exist_ok=True)
    prov = YahooProvider()
    universe = prov.daily(SWING_UNIVERSE, lookback_days=520)
    index_df = prov.daily([INDEX_SYMBOL], lookback_days=520)[INDEX_SYMBOL]
    market = DailyMarket(universe, index_df)

    base = BotConfig()
    no_breakers = dataclasses.replace(base.swing_breakers, max_consecutive_losses=10**9,
                                      daily_loss_limit_pct=10.0, max_drawdown_pct=10.0)
    unlimited = dataclasses.replace(base.swing_sizing, max_positions=10_000, max_per_sector=10_000,
                                    max_portfolio_heat_pct=1e6)
    parity = dataclasses.replace(base, swing_breakers=no_breakers, swing_sizing=unlimited)
    sized = dataclasses.replace(base, swing_sizing=unlimited)
    variants = [
        # name, cfg, slippage bps, fixed ₹15k, regime mode, reverse ranking
        ('P0 parity: all signals, ₹15k fixed, 0 slippage, no regime, no breakers',
         parity, 0.0, True, 'flat', False),
        ('P1 + 5 bps slippage & tick snap', parity, 5.0, True, 'flat', False),
        ('P2 + half-Kelly sizing & breakers (no capacity limit)', sized, 5.0, False, 'flat', False),
        ('P3 + capacity: 5 slots, 2/sector, heat cap, regime entry budget', base, 5.0, False, 'budget', False),
        ('P4 = DEPLOYED (adds the regime size dial)', base, 5.0, False, 'real', False),
        ('P5 deployed, shallowest dip first', base, 5.0, False, 'real', True),
    ]
    no_streak = dataclasses.replace(base.swing_breakers, max_consecutive_losses=10**9)
    cand = dataclasses.replace(base, swing_breakers=no_streak)
    notional = make_notional_allocate(0.20, 8_000.0)
    candidates = [
        ('C1 CANDIDATE: 20%-of-equity notional, min ₹8k, 3% risk cap, capacity on, no size dial, '
         'no loss-streak breaker (daily-loss & drawdown latch kept)', cand, 'budget'),
        ('C2 = C1 + loss-streak breaker', base, 'budget'),
        ('C3 = C1 + regime size dial', cand, 'real'),
    ]
    results = []
    for name, cfg, slip, fixed, mode, rev in variants:
        r = run_variant(market, name, cfg, slip, fixed, mode, reverse=rev)
        results.append(r)
        logger.info(f'  done {name}: {len(r["trades"])} trades')
    for name, cfg, mode in candidates:
        r = run_variant(market, name, cfg, 5.0, False, mode, allocator=notional)
        results.append(r)
        logger.info(f'  done {name}: {len(r["trades"])} trades')

    dates = market.dates[200:]
    cut = dates[int(len(dates) * IS_FRACTION)]
    md = ['## Backtest attribution — where the event-study edge goes', '',
          f"Window {dates[0].date()} → {dates[-1].date()} ({len(dates)} sessions); "
          f"IS before {cut.date()}, OOS after. Win rates and returns per trade are on notional.", '',
          '| variant | segment | trades | gross win | net win | gross/trade | net/trade | t (net) | '
          'avg notional | book return / maxDD |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        for seg, (a, b) in (('ALL', (dates[0], dates[-1] + pd.Timedelta(days=1))),
                            ('IS', (dates[0], cut)), ('OOS', (cut, dates[-1] + pd.Timedelta(days=1)))):
            md.append(f"| {r['name']} | {seg} | {_fmt(_seg(r, a, b))}")
    # Exit-reason split for parity vs deployed
    for r in (results[0], results[4]):
        t = r['trades']
        if len(t):
            md += ['', f"**{r['name']}** — by exit", '', '| exit | n | net/trade | gross win |',
                   '|---|---|---|---|']
            for reason, g in t.groupby(t['exit_reason'].str.split(':').str[0]):
                md.append(f"| {reason} | {len(g)} | {g['net_pct'].mean() * 100:+.2f}% | "
                          f"{(g['gross_pnl'] > 0).mean() * 100:.0f}% |")
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'ATTRIBUTION.md'), 'w') as fh:
        fh.write(report + '\n')
    step = os.environ.get('GITHUB_STEP_SUMMARY')
    if step:
        with open(step, 'a') as fh:
            fh.write(report + '\n')
    print(report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='research_results')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stdout)
    run(args.out)


if __name__ == '__main__':
    main()
