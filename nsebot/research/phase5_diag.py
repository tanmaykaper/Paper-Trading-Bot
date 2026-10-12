"""Phase 5 follow-up diagnostics — POST-HOC, not part of the pre-registered test.

    python -m nsebot.research.phase5_diag --out research_results

The Phase 5 run (run 37295540936) left two questions open. This answers them
without changing any design or rule from that run.

1. Why did the dip-reversion designs go flat? D1 and E1 stopped trading part-way
   through the window and E2 never traded in it. Two mechanisms can freeze the
   book: the 25% drawdown latch, and the ₹8k minimum notional. At 20% notional
   the floor binds once equity falls below ₹40k, after which every signal is
   skipped. This reports when each one first bit.
2. How much of the momentum result is the symbol list? An equal-weight,
   cost-free hold of every name in today's list that was liquid at the time,
   measured on the same window. Whatever it earns above the median fund is
   return the list hands every design for free (survivorship), not skill.
"""

import argparse
import dataclasses
import logging
import os
import sys

import numpy as np
import pandas as pd

from ..config import BotConfig
from .experiments import Panel, benchmark_equal_weight, simulate_rotation
from .hurdle import curve
from .phase5 import WIN_END, WIN_START, YEARS, fetch_range, yearly

logger = logging.getLogger(__name__)


def freeze_stats(eq, trades, floor_equity, dd_limit, win_start=WIN_START):
    """When (if ever) each freeze mechanism first applied, from an engine
    equity curve and its closed-trade journal."""
    eq = eq.dropna()
    dd = eq / eq.cummax() - 1
    below = eq.index[eq < floor_equity]
    latched = eq.index[dd <= -dd_limit]
    entries = pd.to_datetime(trades['entry_time']) if len(trades) else pd.Series([], dtype='datetime64[ns]')
    at_start = eq[eq.index >= win_start]
    return {
        'equity_at_window_start': float(at_start.iloc[0]) if len(at_start) else np.nan,
        'min_equity': float(eq.min()) if len(eq) else np.nan,
        'first_below_floor': below[0] if len(below) else None,
        'first_latch': latched[0] if len(latched) else None,
        'trades': int(len(trades)),
        'entries_before_window': int((entries < win_start).sum()),
        'entries_in_window': int((entries >= win_start).sum()),
        'last_entry': entries.max() if len(entries) else None,
    }


def _d(ts):
    return ts.date().isoformat() if ts is not None and pd.notna(ts) else 'never'


def run(out_dir):
    from ..backtest import run_backtest
    from ..universe import INDEX_SYMBOL, SWING_UNIVERSE

    os.makedirs(out_dir, exist_ok=True)
    universe = fetch_range(SWING_UNIVERSE)
    index_df = fetch_range([INDEX_SYMBOL])[INDEX_SYMBOL]

    base = BotConfig()
    configs = {
        'D1 deployed engine': base,
        'E1 33% x 3 slots': dataclasses.replace(base, swing_sizing=dataclasses.replace(
            base.swing_sizing, target_notional_pct=0.33, max_positions=3, max_portfolio_heat_pct=0.09)),
        'E2 5% trigger': dataclasses.replace(base, reversion=dataclasses.replace(base.reversion, drop_pct=0.05)),
    }
    md = ['# Phase 5 follow-up — post-hoc diagnostics', '',
          '_Not part of the pre-registered test. Explains the flat dip-reversion curves and sizes the '
          'survivorship bias in the symbol list. No design or rule from Phase 5 is changed._', '',
          '## 1. Why the dip-reversion designs froze', '',
          '| design | equity at window start | min equity | floor binds below | first below floor | '
          'first 25% drawdown | trades | entries before window | entries in window | last entry |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for label, cfg in configs.items():
        _, eq, trades, _ = run_backtest(universe, index_df, cfg)
        sz = cfg.swing_sizing
        floor_equity = sz.min_notional_inr / sz.target_notional_pct
        st = freeze_stats(eq, trades, floor_equity, cfg.swing_breakers.max_drawdown_pct)
        md.append(f"| {label} | ₹{st['equity_at_window_start']:,.0f} | ₹{st['min_equity']:,.0f} | "
                  f"₹{floor_equity:,.0f} | {_d(st['first_below_floor'])} | {_d(st['first_latch'])} | "
                  f"{st['trades']} | {st['entries_before_window']} | {st['entries_in_window']} | "
                  f"{_d(st['last_entry'])} |")

    P = Panel(universe)
    c = P.df['close']
    mom12_1 = (c.shift(21) / c.shift(252) - 1).to_numpy()
    nifty = index_df.set_index(pd.to_datetime(index_df['datetime']))['close'].astype(float)
    risk_on = (nifty.reindex(P.dates).ffill()
               .pipe(lambda s: s > s.rolling(200, min_periods=200).mean())).to_numpy()
    series = {
        'EW hold of liquid names in the list (no costs)': benchmark_equal_weight(P),
        'D4 12-1 momentum top 10, no filter': simulate_rotation(P, mom12_1, P.liquid, n=10)[0],
        'D2 12-1 momentum top 5 + 200-day filter': simulate_rotation(P, mom12_1, P.liquid, n=5,
                                                                    risk_on=risk_on)[0],
        'Nifty 50': nifty,
    }
    ylabels = [f"{a.year % 100:02d}-{b.year % 100:02d}" for a, b in YEARS]
    md += ['', '## 2. How much the symbol list gives away for free', '',
           '| series | CAGR | max DD | ' + ' | '.join(ylabels) + ' |',
           '|---|---|---|' + '---|' * len(YEARS)]
    ew_years = yearly(series['EW hold of liquid names in the list (no costs)'])
    stats = {}
    for name, s in series.items():
        st = curve(s.dropna(), WIN_START, WIN_END)
        stats[name] = st
        ys = yearly(s)
        md.append(f"| {name} | {st['cagr'] * 100:+.1f}% | {st['maxdd'] * 100:.1f}% | "
                  + ' | '.join(f'{v * 100:+.1f}%' if np.isfinite(v) else '—' for v in ys) + ' |'
                  if st else f'| {name} | — | — |' + ' — |' * len(YEARS))
    for name in ('D4 12-1 momentum top 10, no filter', 'D2 12-1 momentum top 5 + 200-day filter'):
        ys = yearly(series[name])
        st, ew = stats.get(name), stats.get('EW hold of liquid names in the list (no costs)')
        if st and ew:
            md.append(f"| {name.split(' ')[0]} minus EW | {(st['cagr'] - ew['cagr']) * 100:+.1f} pts | — | "
                      + ' | '.join(f'{(a - b) * 100:+.1f}' if np.isfinite(a) and np.isfinite(b) else '—'
                                   for a, b in zip(ys, ew_years)) + ' |')
    md += ['', '_Compare the EW row with the median active smallcap fund in PHASE5.md. If a cost-free '
               'equal-weight hold of the list already beats the median fund, the list itself is the '
               'advantage, and only a design\'s margin over EW is evidence of an edge._']

    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'PHASE5_DIAG.md'), 'w') as fh:
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
