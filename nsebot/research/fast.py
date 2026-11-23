"""Fast — can the momentum book be built to show strong results inside 2-3 years?

    python -m nsebot.research.fast --out research_results

The owner (Nov 2026) wants to study the book for at most 6 months, then
trade it with real money for aggressive growth that is plainly visible
within 2-3 years. Rounds 9-10 already tested the score (raw 12-1 beat every
alternative), buy filters (V4 goes to a forward test), monthly rebalancing
(worse) and book size (fewer than 10 names breaches the latch). This study
tests the levers that are left, one change at a time from the deployed book.

PRE-REGISTERED — written and committed before any result was seen.

Designs (each one change from B4 top 10: 12-1 momentum, rebalance every 5
sessions, keep a holding until it leaves the top 30, fully invested):
  F1  6-1 momentum            the score reacts to the last 6 months only
  F2  rebalance every 2 sessions
  F3  hold band 2N            sell once a holding leaves the top 20
  F4  volatility-targeted     exposure = min(1, target / realised volatility)
  F5  volatility-targeted, up to 1.5x on margin
  F6  constant 1.3x on margin
on a ₹1,50,000 book, Zerodha CNC costs, next-open fills, no fills on
circuit-locked bars, corporate-action-adjusted prices, every NSE-listed
stock, Nifty's calendar, all 5 rebalance phases.

Volatility targeting (F4, F5): the realised volatility of the unscaled B4
book's daily returns over the last 63 sessions, read every 5 sessions; the
target is the median of that volatility over the untouched Jul 2018 - Jul
2023 window, phase 0. Exposure changes under 0.1 are skipped; each change
costs 0.15% of the rupees moved. Idle cash earns nothing (as in the paper
ledger). Margin (F5, F6) is charged 15% a year on the borrowed rupees, by
calendar day (Zerodha MTF is 0.04% a day), and the leverage is assumed to
be reset every session: an approximation.

Measures, per design, pooled over the 5 phases:
  rolling 2-year and 3-year CAGR from every 21st session since Jul 2018
  (median, 10th percentile, worst, share of windows losing); worst max
  drawdown; CAGR on the untouched Jul 2018 - Jul 2023 window per phase;
  against an equal-weight hold of every eligible stock: the share of
  126-session (6-month) windows the book was ahead, and the information
  ratio of its monthly excess return with the years a live record needs
  to show it at t = 2.

Decision rule:
  1 A design whose worst max drawdown (any phase) is deeper than the 55%
    latch is out.
  2 Among the rest, the winner has the highest median rolling 2-year CAGR.
  3 The winner replaces B4 top 10 only if (a) its median rolling 2-year
    CAGR is at least 3 points higher, (b) its 10th-percentile rolling
    2-year CAGR is no lower than B4 top 10's, and (c) it beats B4 top 10
    at no fewer than 3 of the 5 phases on the untouched window. The bar
    is a point higher than round 10's because six challengers are tried.
  4 A margin design (F5, F6) that wins is not switched on in the paper
    book: margin (MTF) covers only broker-approved stocks, never series BE
    or BZ, which hold many of B4's names. It opens a follow-up that models
    MTF eligibility first.
  5 Two challengers that each pass are not combined here; the combination
    would be a new pre-registered test.

Known limits, stated up front: the windows overlap heavily; delisted stocks
are missing, which flatters every design; the untouched window has been
read by rounds 5-10, so it is no longer truly untouched; six challengers on
one history invite a lucky winner, which rule 3 only partly guards against.
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
from .growth import rolling_cagr
from .hurdle import curve
from .phase5 import FETCH_START, WIN_END, WIN_START, fetch_range
from .phase5b import ever_eligible

logger = logging.getLogger(__name__)

CAPITAL = 150_000.0
PHASES = 5
LATCH = 0.55
MIN_GAIN = 0.03                       # rule 3a: points of median 2-year CAGR over B4 top 10
VOL_WINDOW = 63
VOL_EVERY = 5
MIN_STEP = 0.10                       # exposure changes smaller than this are skipped
SHIFT_COST = 0.0015                   # per rupee moved when exposure changes
BORROW = 0.15                         # margin interest a year, by calendar day
SIX_MONTHS = 126
STEP = 21
BASELINE = 'B4 top 10 (deployed)'
MARGIN = ('F5', 'F6')

DESIGNS = [
    # (name, score key, every, band, overlay)
    (BASELINE, '12-1 momentum (B4)', 5, 30, None),
    ('F1 6-1 momentum', '6-1 momentum', 5, 30, None),
    ('F2 rebalance every 2 sessions', '12-1 momentum (B4)', 2, 30, None),
    ('F3 hold band 2N (sell below rank 20)', '12-1 momentum (B4)', 5, 20, None),
    ('F4 volatility-targeted, no margin', '12-1 momentum (B4)', 5, 30, ('vol', 1.0)),
    ('F5 volatility-targeted, up to 1.5x margin', '12-1 momentum (B4)', 5, 30, ('vol', 1.5)),
    ('F6 constant 1.3x margin', '12-1 momentum (B4)', 5, 30, ('lev', 1.3)),
]


# ═════════════════════════════════════════════════════════════════════════════
# Books
# ═════════════════════════════════════════════════════════════════════════════
def phase_runs(P, score, eligible, n=10, every=5, band=30, capital=CAPITAL, phases=PHASES):
    start = int(np.argmax(np.isfinite(score).sum(axis=1) >= 2 * n))
    buy_ok, sell_ok = locked_bars(P)
    out = []
    for k in range(phases):
        m = score.copy()
        m[start:start + k] = np.nan
        out.append(rotate(P, m, eligible, n=n, every=every, band=band, capital=capital,
                          buy_ok=buy_ok, sell_ok=sell_ok)[0])
    return out


def realised_vol(e, window=VOL_WINDOW):
    r = e.pct_change()
    return r.rolling(window, min_periods=window).std() * np.sqrt(252)


def vol_target(curve0, start=WIN_START, end=WIN_END):
    """The volatility target: median realised volatility of the unscaled book
    over the untouched window."""
    v = realised_vol(curve0)
    return float(v[(v.index >= start) & (v.index <= end)].median())


def overlay(e, kind, level, target=None, every=VOL_EVERY, rate=BORROW, cost=SHIFT_COST, min_step=MIN_STEP):
    """The book `e` held at exposure k instead of 1: k = level ('lev'), or
    min(level, target / realised volatility) read every `every` sessions
    ('vol'). Below 1 the rest is idle cash; above 1 the borrowed rupees pay
    `rate` a year by calendar day. Starts at k = 1 until the book has a full
    volatility window of its own returns; margin ('lev') starts with the
    book's first trade."""
    r = e.pct_change().fillna(0.0).to_numpy()
    vol = realised_vol(e).to_numpy()
    days = np.r_[0, np.diff(e.index.values).astype('timedelta64[D]').astype(float)]
    moving = np.flatnonzero(r != 0)
    first = moving[0] - 1 if len(moving) else len(r)          # the close before the first move
    live_from = first + 1 + VOL_WINDOW
    out = np.empty(len(r))
    E, k = float(e.iloc[0]), 1.0
    for t in range(len(r)):
        if t:
            E = E * (1.0 + k * r[t]) - max(k - 1.0, 0.0) * E * rate * days[t] / 365.0
            E = max(E, 1.0)
        k_new = k
        if kind == 'lev' and t == first:
            k_new = level
        elif kind == 'vol' and t >= live_from and (t - live_from) % every == 0 \
                and np.isfinite(vol[t]) and vol[t] > 0:
            k_new = min(level, target / vol[t])
        if abs(k_new - k) >= min_step:
            E -= abs(k_new - k) * E * cost
            k = k_new
        out[t] = E
    return pd.Series(out, index=e.index)


def market(Q, liquid):
    """Equal-weight hold of every eligible stock, rebalanced daily, no costs:
    the universe the book picks from."""
    c = Q.a['close']
    with np.errstate(invalid='ignore', divide='ignore'):
        r = c[1:] / c[:-1] - 1
    m = liquid[:-1] & np.isfinite(r)
    r = np.clip(np.where(m, r, np.nan), -0.5, 1.0)
    with np.errstate(invalid='ignore'):
        cnt = m.sum(axis=1)
        ew = np.where(cnt > 0, np.nansum(np.where(m, r, 0.0), axis=1) / np.maximum(cnt, 1), 0.0)
    return pd.Series(np.r_[1.0, np.cumprod(1 + ew)], index=Q.dates)


# ═════════════════════════════════════════════════════════════════════════════
# Measures and the rule
# ═════════════════════════════════════════════════════════════════════════════
def versus_market(curves, mkt, start=WIN_START, span=SIX_MONTHS, step=5):
    """(share of `span`-session windows ahead of the market, information
    ratio of 21-session excess returns, years to t = 2), pooled over curves."""
    ahead, excess = [], []
    for e in curves:
        e = e[e.index >= start].dropna()
        m = mkt.reindex(e.index).ffill()
        for i in range(0, len(e) - span, step):
            ahead.append(e.iloc[i + span] / e.iloc[i] > m.iloc[i + span] / m.iloc[i])
        for i in range(0, len(e) - STEP, STEP):
            excess.append(e.iloc[i + STEP] / e.iloc[i] - m.iloc[i + STEP] / m.iloc[i])
    x = np.asarray(excess, dtype=float)
    ir = x.mean() / x.std(ddof=1) * np.sqrt(12) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan
    years = (2.0 / ir) ** 2 if np.isfinite(ir) and ir > 0 else np.inf
    return (float(np.mean(ahead)) if ahead else np.nan), ir, years


def summarise(curves, last, mkt):
    full = [curve(e, WIN_START, last) for e in curves]
    untouched = np.array([(curve(e, WIN_START, WIN_END) or {}).get('cagr', np.nan) for e in curves])
    ahead, ir, years = versus_market(curves, mkt)
    return {'r2': rolling_cagr(curves, 2), 'r3': rolling_cagr(curves, 3), 'untouched': untouched,
            'maxdd': min(f['maxdd'] for f in full if f), 'full': np.median([f['cagr'] for f in full if f]),
            'ahead6m': ahead, 'ir': ir, 'years_t2': years}


def decide(results):
    """(winner, deployed, reasons) by the pre-registered rule."""
    base = results[BASELINE]
    ok = {k: v for k, v in results.items() if v['maxdd'] >= -LATCH}
    if not ok:
        return None, BASELINE, ['every design breaches the 55% latch; B4 top 10 stays']
    win = max(ok, key=lambda k: np.median(ok[k]['r2']))
    w = ok[win]
    gain = np.median(w['r2']) - np.median(base['r2'])
    tail = np.percentile(w['r2'], 10) - np.percentile(base['r2'], 10)
    phases = int(np.sum(w['untouched'] > base['untouched']))
    why = [f'highest median rolling 2-year CAGR within the latch: {win} ({np.median(w["r2"]) * 100:+.1f}% '
           f'against {np.median(base["r2"]) * 100:+.1f}% for B4 top 10, {gain * 100:+.1f} points; 10th '
           f'percentile {tail * 100:+.1f} points; beats it at {phases} of {PHASES} phases on the untouched window)']
    if win == BASELINE:
        return win, BASELINE, why + ['the winner is the deployed design']
    if not (gain >= MIN_GAIN and tail >= 0 and phases >= 3):
        return win, BASELINE, why + ['rule 3 not met (needs +3 points, no worse 10th percentile, and 3 of 5 '
                                     'phases): B4 top 10 stays']
    if win.split()[0] in MARGIN:
        return win, BASELINE, why + ['rule 3 holds, but it is a margin design (rule 4): B4 top 10 stays on '
                                     'paper and MTF eligibility is modelled next']
    return win, win, why + ['all three conditions of rule 3 hold: deploy it']


def _p(v):
    return '—' if v is None or not np.isfinite(v) else f'{v * 100:+.1f}%'


# ═════════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════════
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
    mkt = market(Q, P.liquid)

    books, results, target = {}, {}, None
    for name, key, every, band, ov in DESIGNS:
        base_key = (key, every, band)
        if base_key not in books:
            books[base_key] = phase_runs(Q, s[key], P.liquid, every=every, band=band)
        curves = books[base_key]
        if ov:
            if target is None:
                target = vol_target(books[('12-1 momentum (B4)', 5, 30)][0])
            curves = [overlay(e, ov[0], ov[1], target=target) for e in curves]
        results[name] = summarise(curves, last, mkt)
        results[name]['curves'] = curves
    win, deploy, why = decide(results)

    md = ['# Fast — building the momentum book to show results inside 2–3 years', '',
          '**Pre-registered** (see the module docstring for the rule, committed before this run). '
          f'Universe: {note}; {len(broad)} ever eligible. ₹1,50,000 book, CNC costs, corporate-action-adjusted '
          f'prices, 5 rebalance phases, data to {last.date()}. Volatility target {_p(target)} a year. '
          'Delisted stocks are missing, which flatters every row; the rolling windows overlap heavily.', '',
          '| design | median 2-yr CAGR | 10th pct 2-yr | worst 2-yr | 2-yr windows losing | median 3-yr CAGR | '
          'CAGR full (median) | worst max DD | untouched phases beating B4 | 6-month windows ahead of market | '
          'information ratio | years to show it at t=2 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    base = results[BASELINE]
    for name, v in results.items():
        r2, r3 = v['r2'], v['r3']
        beats = '—' if name == BASELINE else f"{int(np.sum(v['untouched'] > base['untouched']))}/{PHASES}"
        flag = ' ⛔' if v['maxdd'] < -LATCH else ''
        md.append(f'| {name}{flag} | {_p(np.median(r2))} | {_p(np.percentile(r2, 10))} | {_p(r2.min())} | '
                  f'{(r2 < 0).mean():.0%} | {_p(np.median(r3))} | {_p(v["full"])} | {_p(v["maxdd"])} | {beats} | '
                  f'{v["ahead6m"]:.0%} | {v["ir"]:.2f} | {v["years_t2"]:.1f} |')
    md += ['', '_⛔ worst drawdown deeper than the 55% latch: excluded by rule 1. "Market" is an equal-weight '
               'hold of every eligible NSE stock, no costs._', '', '## Decision', '']
    md += [f'- {w}' for w in why]
    md += ['', f'**Deploy: {deploy}.**', '']

    chosen = results[deploy]['curves']
    md += [f'## What ₹1,50,000 became in {deploy}, every start date since Jul 2018', '',
           '| horizon | windows | worst | 10th pct | median | 90th pct | best |', '|---|---|---|---|---|---|---|']
    for yrs in (2, 3):
        r = rolling_cagr(chosen, yrs)
        if len(r):
            vals = [CAPITAL * (1 + x) ** yrs for x in (r.min(), np.percentile(r, 10), np.median(r),
                                                       np.percentile(r, 90), r.max())]
            md.append(f'| {yrs} years | {len(r)} | ' + ' | '.join(f'₹{v:,.0f}' for v in vals) + ' |')
    md.append('')
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'FAST.md'), 'w') as fh:
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
