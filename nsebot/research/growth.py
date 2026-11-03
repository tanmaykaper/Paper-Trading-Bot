"""Growth — sizing the momentum book for the owner's 3-5 year horizon.

    python -m nsebot.research.growth --out research_results

The owner chose growth over consistency and hit rate (Nov 2026): the whole
₹1.5L book moves into the momentum sleeve, the only design with an
out-of-sample record. This study asks how that book should be built to
maximise what it is worth after 3-5 years.

PRE-REGISTERED — written and committed before any result was seen.

Designs: every combination of
  size       top 5, 7, 10 or 15 names, each kept until it leaves the top 3N
  buy rule   B4 (12-1 momentum alone) or V4 (B4, but no buy within 63
             sessions of a 15%+ single-day gain; docs/DIAGNOSIS.md §2.4)
on a ₹1,50,000 book, rebalanced every 5 sessions, Zerodha CNC costs, next-
open fills, no fills on circuit-locked bars, prices adjusted for
corporate-action gaps (research.diagnosis), every NSE-listed stock, Nifty's
calendar, all 5 rebalance phases.

Measures, per design, pooled over the 5 phases:
  rolling 3-year and 5-year CAGR, from every 21st session since Jul 2018
  whose window ends inside the data: median, 10th percentile, worst, and
  the share of windows that lost money; worst max drawdown; CAGR on the
  untouched Jul 2018 - Jul 2023 window per phase.

Decision rule:
  1 A design whose worst max drawdown (any phase, Jul 2018 to the latest
    bar) is deeper than the sleeve's 55% latch is out: the latch would have
    stopped its buying at the bottom of the crash it needed to recover from.
  2 Among the rest, the winner has the highest median rolling 3-year CAGR.
  3 The winner replaces the deployed B4 top 10 only if its median rolling
    3-year CAGR is at least 2 points higher AND it beats B4 top 10 at no
    fewer than 3 of the 5 phases on the untouched window. Otherwise B4 top
    10 stays as it is.

Known limits, stated up front: the windows overlap heavily (8 years of data
hold fewer than three independent 3-year periods); delisted stocks are
missing, which flatters every design; V4 was found on this same history.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from ..listing import nse_equities
from .diagnosis import adjusted, ca_events, rotate, scores
from .experiments import Panel, locked_bars
from .hurdle import curve
from .phase5 import FETCH_START, WIN_END, WIN_START, fetch_range
from .phase5b import ever_eligible

logger = logging.getLogger(__name__)

CAPITAL = 150_000.0
SIZES = (5, 7, 10, 15)
PHASES = 5
LATCH = 0.55
MIN_GAIN = 0.02                       # rule 3: points of median 3-year CAGR over B4 top 10
JUMP = 0.15
STEP = 21                             # rolling-window start spacing, sessions
BASELINE = ('B4', 10)


def designs(liquid, s):
    mom = s['12-1 momentum (B4)']
    with np.errstate(invalid='ignore'):
        nojump = -s['−largest daily gain, 63 sessions'] < JUMP
    return [((rule, n), mom, liquid, f, n) for rule, f in (('B4', None), ('V4', nojump)) for n in SIZES]


def phase_runs(P, score, eligible, buy_filter, n, capital=CAPITAL, phases=PHASES):
    start = int(np.argmax(np.isfinite(score).sum(axis=1) >= 2 * n))
    buy_ok, sell_ok = locked_bars(P)
    out = []
    for k in range(phases):
        m = score.copy()
        m[start:start + k] = np.nan
        out.append(rotate(P, m, eligible, n=n, capital=capital, buy_ok=buy_ok, sell_ok=sell_ok,
                          buy_filter=buy_filter)[0])
    return out


def rolling_cagr(curves, years, start=WIN_START, step=STEP):
    """CAGR over every `years`-long window starting each `step` sessions from
    `start`, pooled over the curves."""
    span = int(round(252 * years))
    out = []
    for e in curves:
        e = e[e.index >= start].dropna()
        for i in range(0, len(e) - span, step):
            out.append((e.iloc[i + span] / e.iloc[i]) ** (1 / years) - 1)
    return np.asarray(out, dtype=float)


def summarise(curves, last):
    r3, r5 = rolling_cagr(curves, 3), rolling_cagr(curves, 5)
    full = [curve(e, WIN_START, last) for e in curves]
    untouched = np.array([(curve(e, WIN_START, WIN_END) or {}).get('cagr', np.nan) for e in curves])
    return {'r3': r3, 'r5': r5, 'untouched': untouched,
            'maxdd': min(f['maxdd'] for f in full if f),
            'full': np.median([f['cagr'] for f in full if f])}


def decide(results):
    """(winner key, deployed key, reasons) by the pre-registered rule."""
    base = results[BASELINE]
    ok = {k: v for k, v in results.items() if v['maxdd'] >= -LATCH}
    if not ok:
        return None, BASELINE, ['every design breaches the 55% latch; B4 top 10 stays']
    win = max(ok, key=lambda k: np.median(ok[k]['r3']))
    gain = np.median(ok[win]['r3']) - np.median(base['r3'])
    phases = int(np.sum(ok[win]['untouched'] > base['untouched']))
    why = [f'highest median rolling 3-year CAGR within the latch: {win[0]} top {win[1]} '
           f'({np.median(ok[win]["r3"]) * 100:+.1f}% against {np.median(base["r3"]) * 100:+.1f}% for B4 top 10, '
           f'{gain * 100:+.1f} points; beats it at {phases} of {PHASES} phases on the untouched window)']
    if win == BASELINE:
        return win, BASELINE, why + ['the winner is the deployed design']
    if gain >= MIN_GAIN and phases >= 3:
        return win, win, why + ['both conditions of rule 3 hold: deploy it']
    return win, BASELINE, why + ['rule 3 not met (needs +2 points and 3 of 5 phases): B4 top 10 stays']


def _p(v):
    return '—' if v is None or not np.isfinite(v) else f'{v * 100:+.1f}%'


def run(out_dir, fetch=None, get_list=None):
    from ..universe import INDEX_SYMBOL

    fetch = fetch or fetch_range
    os.makedirs(out_dir, exist_ok=True)
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    symbols, note = (get_list or (lambda: nse_equities(os.path.join(out_dir, 'nse_equity_list.json'))))()
    raw = fetch(symbols, FETCH_START, end)
    broad = {s_: d for s_, d in raw.items() if ever_eligible(d)}
    index_df = fetch([INDEX_SYMBOL], FETCH_START, end)[INDEX_SYMBOL]
    idx_dates = pd.DatetimeIndex(pd.to_datetime(index_df['datetime']))
    on_index = {s_: d[pd.to_datetime(d['datetime']).isin(idx_dates)].reset_index(drop=True)
                for s_, d in broad.items()}
    P = Panel({s_: d for s_, d in on_index.items() if len(d) >= 60})
    Q = adjusted(P, ca_events(P))
    s = scores(Q, P.liquid)
    last = P.dates[-1]

    results = {}
    for key, score, elig, bf, n in designs(P.liquid, s):
        curves = phase_runs(Q, score, elig, bf, n)
        results[key] = summarise(curves, last)
        results[key]['curves'] = curves
    win, deploy, why = decide(results)

    md = ['# Growth — sizing the momentum book for a 3–5 year horizon', '',
          '**Pre-registered** (see the module docstring for the rule, committed before this run). '
          f'Universe: {note}; {len(broad)} ever eligible. ₹1,50,000 book, CNC costs, corporate-action-adjusted '
          f'prices, 5 rebalance phases, data to {last.date()}. Delisted stocks are missing, which flatters '
          'every row; the rolling windows overlap heavily.', '',
          '| design | median 3-yr CAGR | 10th pct 3-yr | worst 3-yr | 3-yr windows losing | median 5-yr CAGR | '
          'worst 5-yr | CAGR full (median) | worst max DD | untouched phases beating B4 top 10 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    base = results[BASELINE]
    for key, v in results.items():
        r3, r5 = v['r3'], v['r5']
        beats = '—' if key == BASELINE else f"{int(np.sum(v['untouched'] > base['untouched']))}/{PHASES}"
        flag = ' ⛔' if v['maxdd'] < -LATCH else ''
        md.append(f'| {key[0]} top {key[1]}{flag} | {_p(np.median(r3))} | {_p(np.percentile(r3, 10))} | '
                  f'{_p(r3.min())} | {(r3 < 0).mean():.0%} | {_p(np.median(r5))} | {_p(r5.min())} | '
                  f'{_p(v["full"])} | {_p(v["maxdd"])} | {beats} |')
    md += ['', '_⛔ worst drawdown deeper than the 55% latch: excluded by rule 1._', '', '## Decision', '']
    md += [f'- {w}' for w in why]
    md += ['', f'**Deploy: {deploy[0]} top {deploy[1]}.**', '']

    chosen = results[deploy]['curves']
    md += [f'## What ₹1,50,000 became in {deploy[0]} top {deploy[1]}, every start date since Jul 2018', '',
           '| horizon | windows | worst | 10th pct | median | 90th pct | best |', '|---|---|---|---|---|---|---|']
    for yrs in (3, 5):
        r = rolling_cagr(chosen, yrs)
        if len(r):
            vals = [CAPITAL * (1 + x) ** yrs for x in (r.min(), np.percentile(r, 10), np.median(r),
                                                       np.percentile(r, 90), r.max())]
            md.append(f'| {yrs} years | {len(r)} | ' + ' | '.join(f'₹{v:,.0f}' for v in vals) + ' |')
    md.append('')
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'GROWTH.md'), 'w') as fh:
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
