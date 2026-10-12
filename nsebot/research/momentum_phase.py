"""How much of B4's result is the day it happens to rebalance on?

    python -m nsebot.research.momentum_phase --out research_results

Post-hoc diagnostic for round 7, where the live engine replayed on real data
beat the research curve by 4.7–6.3 points of CAGR although the two code paths
are identical on synthetic data. A rebalance every 5 sessions has 5 possible
phases; which one a run gets depends on its first day and on which calendar
it counts sessions on. This reruns the research simulator's B4 at all 5
phases on both calendars:

  union  every stock's trading days together (rounds 5–6)
  index  Nifty's sessions only (how the live engine counts)

It also counts the dates where those calendars disagree. If the phases
alone spread CAGR by several points, B4's single-run estimates (and the
round-7 gap) are timing noise, and the honest range is the spread.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from ..listing import nse_equities
from .experiments import Panel, locked_bars, simulate_rotation
from .hurdle import curve
from .phase5 import FETCH_START, WIN_END, WIN_START, fetch_range
from .phase5b import ever_eligible

logger = logging.getLogger(__name__)
PHASES = 5


def phase_runs(P, n=10):
    """B4 equity curves with the first rebalance delayed by 0..PHASES-1 sessions."""
    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    start = int(np.argmax(np.isfinite(mom).sum(axis=1) >= 2 * n))
    buy_ok, sell_ok = locked_bars(P)
    out = []
    for k in range(PHASES):
        m = mom.copy()
        m[start:start + k] = np.nan                    # fewer than 2n scores: the first rebalance moves k later
        out.append(simulate_rotation(P, m, P.liquid, n=n, buy_ok=buy_ok, sell_ok=sell_ok)[0])
    return out


def run(out_dir, fetch=None, get_list=None):
    from ..universe import INDEX_SYMBOL

    fetch = fetch or fetch_range
    os.makedirs(out_dir, exist_ok=True)
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    symbols, note = (get_list or (lambda: nse_equities(os.path.join(out_dir, 'nse_equity_list.json'))))()
    raw = fetch(symbols, FETCH_START, end)
    broad = {s: d for s, d in raw.items() if ever_eligible(d)}
    index_df = fetch([INDEX_SYMBOL], FETCH_START, end)[INDEX_SYMBOL]
    idx_dates = pd.DatetimeIndex(pd.to_datetime(index_df['datetime']))

    P_union = Panel(broad)
    on_index = {s: d[pd.to_datetime(d['datetime']).isin(idx_dates)].reset_index(drop=True) for s, d in broad.items()}
    P_index = Panel({s: d for s, d in on_index.items() if len(d) >= 60})
    span = (P_union.dates >= idx_dates[0]) & (P_union.dates <= idx_dates[-1])
    extra = P_union.dates[span].difference(idx_dates)
    missing = idx_dates.difference(P_union.dates)
    last = min(P_union.dates[-1], idx_dates[-1])

    md = ['# Momentum phase sensitivity — B4 at every rebalance phase', '',
          f'Universe: {note}; {len(broad)} ever eligible. Research simulator (rounds 5–6), ₹50k, CNC costs.', '',
          f'**Calendars:** {len(extra)} stock-trading dates are not Nifty sessions'
          + (f" (e.g. {', '.join(str(d.date()) for d in extra[:6])})" if len(extra) else '')
          + f'; {len(missing)} Nifty sessions have no stock bars'
          + (f" (e.g. {', '.join(str(d.date()) for d in missing[:6])})" if len(missing) else '') + '.', '',
          '| calendar | phase | CAGR Jul 2018 – Jul 2023 | CAGR Jul 2018 – latest | max DD |', '|---|---|---|---|---|']
    summary = {}
    for name, P in (('union (rounds 5–6)', P_union), ('Nifty sessions (live)', P_index)):
        rows = []
        for k, eq in enumerate(phase_runs(P)):
            a, b = curve(eq, WIN_START, WIN_END), curve(eq, WIN_START, last)
            if a and b:
                rows.append((a['cagr'], b['cagr'], b['maxdd']))
                md.append(f"| {name} | +{k} | {a['cagr'] * 100:+.1f}% | {b['cagr'] * 100:+.1f}% | {b['maxdd'] * 100:.1f}% |")
        summary[name] = rows
    md += ['', '| calendar | untouched CAGR: min / median / max | full-span CAGR: min / median / max | worst max DD |',
           '|---|---|---|---|']
    for name, rows in summary.items():
        if rows:
            u, f, d = (np.array(x) for x in zip(*rows))
            md.append(f'| {name} | {u.min() * 100:+.1f}% / {np.median(u) * 100:+.1f}% / {u.max() * 100:+.1f}% | '
                      f'{f.min() * 100:+.1f}% / {np.median(f) * 100:+.1f}% / {f.max() * 100:+.1f}% | {d.min() * 100:.1f}% |')
    md += ['', '_Round 7 for reference: the live engine replay made +28.9% (untouched) and +34.9% (full span); '
               'the round-6 research run +22.6% and +30.2%. Benchmarks over the same windows: median smallcap '
               'fund +21.7% / +18.5%, best +28.7% / +24.5%._']
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'MOMENTUM_PHASE.md'), 'w') as fh:
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
