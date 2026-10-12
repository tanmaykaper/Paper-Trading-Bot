"""Replay the LIVE momentum engine over real history and compare it with the
research simulator's B4 (docs/RESEARCH.md, round 6).

    python -m nsebot.research.momentum_replay --out research_results

The research curve and the deployed code agree to the rupee on synthetic data
built to remove their deliberate differences (tests/test_momentum.py). On real
data those differences come back, and this measures what they cost:

  * the live paper broker pays 5 bps slippage on every fill and snaps prices
    to the exchange tick in the adverse direction;
  * orders are sized at the decision day's close, not the fill day's open;
  * a buy that lands on a circuit-locked bar is cancelled and its slot waits
    for the next rebalance (research skipped to the next name with
    look-ahead it could not have had).

Same universe as Phase 5b: every NSE-listed equity (series EQ/BE/BZ) that is
ever eligible, Yahoo daily bars from Jul 2016, ₹50k, Zerodha CNC costs.
"""

import argparse
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from ..broker.paper import PaperBroker
from ..config import BotConfig
from ..engine.momentum import MomentumEngine, MomentumMarket
from ..ledger import Ledger
from ..listing import nse_equities
from .hurdle import curve
from .phase5 import FETCH_START, WIN_END, WIN_START, fetch_range
from .phase5b import by_year, ever_eligible, july_years, momentum_book

logger = logging.getLogger(__name__)


def replay(universe, index_df, start, workdir, cfg=None):
    """Run the live engine session by session from `start`. Returns
    (equity series, ledger, counts of what the live-data rules did)."""
    cfg = cfg or BotConfig()
    market = MomentumMarket(universe, index_df, cfg.momentum)
    L = Ledger(workdir, 'momentum', 50_000.0)
    eng = MomentumEngine(cfg, PaperBroker(), L, workdir=workdir, log=lambda m: None)
    counts = {'rebalances': 0, 'cancelled': 0, 'kept': 0, 'skipped_rebalances': 0}
    for d in market.calendar[market.calendar >= start]:
        rep = eng.run(market, d)
        counts['rebalances'] += int(rep['rebalanced'])
        counts['cancelled'] += len(rep['cancelled'])
        counts['kept'] += len([k for k in rep['kept'] if 'queued' not in k[1]])
        counts['skipped_rebalances'] += int(any(w.startswith('DATA:') for w in rep['warnings']))
    eq = L.equity_history()
    return pd.Series(eq['equity'].astype(float).to_numpy(), index=pd.to_datetime(eq['session'])), L, counts


def run(out_dir, fetch=None, get_list=None):
    from ..universe import INDEX_SYMBOL

    fetch = fetch or fetch_range
    os.makedirs(out_dir, exist_ok=True)
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    symbols, note = (get_list or (lambda: nse_equities(os.path.join(out_dir, 'nse_equity_list.json'))))()
    raw = fetch(symbols, FETCH_START, end)
    broad = {s: d for s, d in raw.items() if ever_eligible(d)}
    index_df = fetch([INDEX_SYMBOL], FETCH_START, end)[INDEX_SYMBOL]

    P, research, _ = momentum_book(broad)
    c = P.df['close']
    mom = (c.shift(21) / c.shift(252) - 1).to_numpy()
    start = P.dates[int(np.argmax(np.isfinite(mom).sum(axis=1) >= 20))]
    live, L, counts = replay(broad, index_df, start, tempfile.mkdtemp(prefix='nsebot_replay_'))
    last = min(live.index[-1], research.index[-1])
    trades = L.closed_trades()
    live_costs = float(trades['costs'].sum()) + sum(p.meta['entry_costs'] for p in L.positions)

    md = ['# Momentum replay — the live engine on real history', '',
          f'Universe: {note}; {len(broad)} ever eligible. Both curves start {start.date()} with ₹50k; '
          f'data to {last.date()}. Live = MomentumEngine with the paper broker (5 bps slippage, tick '
          f'snapping, close-day sizing, no look-ahead on circuit locks).', '',
          '| | research B4 | live engine | live − research |', '|---|---|---|---|']
    for label, a, b in (('Jul 2018 – Jul 2023 (untouched)', WIN_START, WIN_END),
                        (f'Jul 2018 – {last.date()}', WIN_START, last)):
        r, l = curve(research, a, b), curve(live, a, b)
        if r and l:
            md.append(f"| CAGR, {label} | {r['cagr'] * 100:+.1f}% | {l['cagr'] * 100:+.1f}% | "
                      f"{(l['cagr'] - r['cagr']) * 100:+.1f} pts |")
            md.append(f"| max drawdown, {label} | {r['maxdd'] * 100:.1f}% | {l['maxdd'] * 100:.1f}% | "
                      f"{(l['maxdd'] - r['maxdd']) * 100:+.1f} pts |")
    years = july_years(last)
    labels = [f"{a.year % 100:02d}-{(a.year + 1) % 100:02d}" for a, _ in years]
    ry, ly = by_year(research, years), by_year(live, years)
    md += ['', '| year | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(years),
           '| research B4 | ' + ' | '.join(f'{v * 100:+.1f}%' if np.isfinite(v) else '—' for v in ry) + ' |',
           '| live engine | ' + ' | '.join(f'{v * 100:+.1f}%' if np.isfinite(v) else '—' for v in ly) + ' |',
           '', f"Live engine: {counts['rebalances']} rebalances, {len(trades)} completed round trips, "
               f"costs ₹{live_costs:,.0f}; {counts['cancelled']} orders not filled (circuit locks, no "
               f"trade, cash), {counts['kept']} sales deferred, {counts['skipped_rebalances']} rebalances "
               f"skipped as data faults. Final equity ₹{live.iloc[-1]:,.0f}.", '',
           '_Synthetic tests prove the two code paths identical once slippage, tick snapping and '
           'close-day sizing are removed; the gap above is what those real-world frictions cost._']
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'MOMENTUM_REPLAY.md'), 'w') as fh:
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
