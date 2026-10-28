"""Diagnosis — where the live book's first weeks went, and how much each
candidate fix would have changed on real NSE data.

    python -m nsebot.research.diagnosis --out research_results

POST-HOC AND EXPLORATORY. Written in Oct 2026, after the live paper book lost
money in its first three weeks. Every variant below was chosen because of
what the live book did, and 2018-26 has been looked at before (rounds 2-8),
so nothing here is a clean test. Read the numbers as the size of each effect
on this history (which fixes matter and which don't), not as a promise that
a variant beats B4 from here. A variant earns deployment only through a
pre-registered forward test.

  1 LIVE WINDOW   the momentum sleeve's first decision date to the latest
                  close, against Nifty indices and the equal-weight eligible
                  market; what B4 and each variant would have bought then.
  2 CORPORATE     one-session collapses at the open (a 30%+ gap down that
    ACTIONS       also closes 30%+ down). Under NSE's circuit bands these are
                  demergers, unadjusted splits or data faults, not markets.
                  How often B4 held one, what its ledger booked, and B4
                  rerun on prices adjusted for them (value kept across the
                  gap, the way a data vendor adjusts a demerger).
  3 PREDICTIVE    rank IC of each candidate score against the next 21
    POWER         sessions' return (non-overlapping samples), the top 10's
                  excess return, forward returns by 12-1 level, and a
                  re-ranking test inside B4's top 30.
  4 VARIANTS      each candidate book at all 5 rebalance phases on Nifty's
                  calendar (the live engine's): CAGR on the untouched
                  2018-23 window, the full span and the recent window, worst
                  drawdown, Sharpe, costs.
"""

import argparse
import io
import json
import logging
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..costs import DEFAULT_CHARGES
from ..listing import SOURCES as LIST_URLS, _get_text, nse_equities
from .experiments import ROTATION_CAPITAL, Panel, curve_stats, locked_bars
from .hurdle import curve
from .phase5 import FETCH_START, WIN_END, WIN_START, fetch_range
from .phase5b import RECENT_START, ever_eligible

logger = logging.getLogger(__name__)

LIVE_START = pd.Timestamp('2026-10-05')      # the sleeve's first decision date (₹50k cash at that close)
RESEARCHED_START = pd.Timestamp('2023-08-14')
GAP = -0.30                                  # open AND close this far under the previous close
PHASES = 5
FWD = 21                                     # forward window for the predictive tests, sessions
JUMP = 0.15                                  # V4: no single-day gain this large in the last 63 sessions
NEAR_HIGH = 0.80                             # V5: close within 20% of the 252-session high
BENCHMARKS = {'^NSEI': 'Nifty 50', '^CRSLDX': 'Nifty 500', '^CNXSC': 'Nifty Smallcap 100',
              '^NSEMDCP50': 'Nifty Midcap 50'}
LEVELS = [(-np.inf, 0.0, '< 0%'), (0.0, 0.5, '0–50%'), (0.5, 1.0, '50–100%'),
          (1.0, 2.0, '100–200%'), (2.0, np.inf, '> 200%')]


# ═════════════════════════════════════════════════════════════════════════════
# Book simulator
# ═════════════════════════════════════════════════════════════════════════════
def rotate(P, score, eligible, n=10, band=None, every=5, capital=ROTATION_CAPITAL,
           buy_ok=None, sell_ok=None, buy_filter=None, log=False):
    """experiments.simulate_rotation plus two things it does not do: a filter
    that gates BUYS only (a holding still leaves on the rank band or on
    eligibility, as in B4), and an optional log of the shares held at the
    start of each session. With buy_filter=None it reproduces
    simulate_rotation exactly (tests/test_diagnosis.py)."""
    o, c = P.a['open'], P.a['close']
    T, N = c.shape
    band = band or 3 * n
    cash, qty = capital, np.zeros(N, dtype=int)
    last_px = np.full(N, np.nan)
    equity = np.full(T, np.nan)
    held = np.zeros((T, N), dtype=np.int32) if log else None
    costs = 0.0
    start = int(np.argmax(np.isfinite(score).sum(axis=1) >= 2 * n))
    for t in range(T):
        last_px = np.where(np.isfinite(c[t]), c[t], last_px)
        equity[t] = cash + np.nansum(qty * last_px)
        if log:
            held[t] = qty
        if t < start or (t - start) % every or t + 1 >= T:
            continue
        sc = np.where(eligible[t] & np.isfinite(score[t]), score[t], -np.inf)
        order = np.argsort(-sc)
        rank = np.empty(N, dtype=int)
        rank[order] = np.arange(N)
        valid = np.isfinite(sc)
        for j in np.flatnonzero(qty > 0):
            if (not valid[j] or rank[j] >= band) and np.isfinite(o[t + 1, j]) \
                    and (sell_ok is None or sell_ok[t + 1, j]):
                v = qty[j] * o[t + 1, j]
                fee = DEFAULT_CHARGES.cnc(0.0, v)
                cash += v - fee
                costs += fee
                qty[j] = 0
        slots = n - int((qty > 0).sum())
        target = equity[t] / n
        for j in order:
            if slots <= 0 or not valid[j]:
                break
            if qty[j] > 0 or not np.isfinite(o[t + 1, j]) or (buy_ok is not None and not buy_ok[t + 1, j]) \
                    or (buy_filter is not None and not buy_filter[t, j]):
                continue
            px = o[t + 1, j]
            q = int(min(target, cash) // px)
            if q < 1:
                continue
            fee = DEFAULT_CHARGES.cnc(q * px, 0.0, dp_charged=False)
            cash -= q * px + fee
            costs += fee
            qty[j] = q
            slots -= 1
    return pd.Series(equity, index=P.dates), costs, held


def phase_curves(P, score, eligible, buy_filter=None, n=10, every=5, phases=PHASES):
    """One run per rebalance phase: the first rebalance delayed 0..phases-1
    sessions (research.momentum_phase), each (equity, costs)."""
    start = int(np.argmax(np.isfinite(score).sum(axis=1) >= 2 * n))
    buy_ok, sell_ok = locked_bars(P)
    out = []
    for k in range(phases):
        m = score.copy()
        m[start:start + k] = np.nan
        eq, costs, _ = rotate(P, m, eligible, n=n, every=every, buy_ok=buy_ok, sell_ok=sell_ok,
                              buy_filter=buy_filter)
        out.append((eq, costs))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Corporate actions
# ═════════════════════════════════════════════════════════════════════════════
def ca_events(P, gap=GAP):
    """(date x symbol) bool: the session opened AND closed at least |gap| under
    the previous close. NSE's 2-20% circuit bands rule that out for a market
    move in all but a few F&O names; it is a demerger, an unadjusted split or
    bonus, or a data fault."""
    o, c = P.a['open'], P.a['close']
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    with np.errstate(invalid='ignore', divide='ignore'):
        return (o / prev - 1 <= gap) & (c / prev - 1 <= gap)


def adjusted(P, events):
    """A light copy of the panel with every price before each event scaled by
    the event's open over the previous close, so value is kept across the
    gap. Eligibility stays the raw panel's: a decision may not use a
    corporate action that has not happened yet."""
    px = {k: P.a[k].copy() for k in ('open', 'high', 'low', 'close')}
    for t, j in zip(*np.nonzero(events)):
        f = px['open'][t, j] / px['close'][t - 1, j]
        for a in px.values():
            a[:t, j] *= f
    return SimpleNamespace(dates=P.dates, symbols=P.symbols, liquid=P.liquid, a=px,
                           df={'close': pd.DataFrame(px['close'], index=P.dates, columns=P.symbols)})


def ca_hits(P, events, held):
    """Events that hit a holding: (date, symbol, shares, rupees the ledger
    booked across the gap)."""
    c, o = P.a['close'], P.a['open']
    rows = []
    for t, j in zip(*np.nonzero(events)):
        q = int(held[t - 1, j]) if t >= 1 else 0
        if q > 0:
            rows.append((P.dates[t], P.symbols[j], q, q * (c[t - 1, j] - o[t, j]),
                         o[t, j] / c[t - 1, j] - 1))
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Scores
# ═════════════════════════════════════════════════════════════════════════════
def _xs_z(x, mask):
    """Cross-sectional z-score per date over `mask`, clipped at ±3."""
    X = np.where(mask & np.isfinite(x), x, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        z = (X - mu) / sd
    return np.clip(z, -3.0, 3.0)


def scores(P, liquid):
    """Every candidate score as a (date x symbol) array, oriented so that
    higher is better."""
    c = P.df['close']
    r = c / c.shift(1) - 1
    with np.errstate(invalid='ignore', divide='ignore'):
        mom = c.shift(21) / c.shift(252) - 1
        mom6 = c.shift(21) / c.shift(126) - 1
        vol = r.rolling(252, min_periods=200).std() * np.sqrt(252)
        up = (r > 0).astype(float).where(r.notna()).rolling(231, min_periods=180).mean().shift(21)
        down = (r < 0).astype(float).where(r.notna()).rolling(231, min_periods=180).mean().shift(21)
        fip = np.sign(mom) * (down - up)                       # Da, Gurun & Warachka: lower = smoother
        s = {'12-1 momentum (B4)': mom.to_numpy(),
             '6-1 momentum': mom6.to_numpy(),
             '12-1 / volatility': (mom / vol).to_numpy(),
             '6m+12m risk-adjusted composite': None,
             'smoothness (−frog-in-the-pan ID)': (-fip).to_numpy(),
             '52-week-high proximity': (c / c.rolling(252, min_periods=200).max()).to_numpy(),
             '−largest daily gain, 63 sessions': (-r.rolling(63, min_periods=40).max()).to_numpy(),
             '−volatility': (-vol).to_numpy()}
    s['6m+12m risk-adjusted composite'] = (_xs_z(s['12-1 / volatility'], liquid)
                                           + _xs_z((mom6 / vol).to_numpy(), liquid)) / 2.0
    return s


def variants(P, liquid, s):
    """(name, score, hold eligibility, buy-only filter, n, every)."""
    mom, fip = s['12-1 momentum (B4)'], -s['smoothness (−frog-in-the-pan ID)']
    with warnings.catch_warnings(), np.errstate(invalid='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        med_fip = np.nanmedian(np.where(liquid & np.isfinite(mom), fip, np.nan), axis=1, keepdims=True)
        smooth = fip <= med_fip
        nojump = -s['−largest daily gain, 63 sessions'] < JUMP
        near = s['52-week-high proximity'] >= NEAR_HIGH
    return [
        ('B4 deployed: 12-1, top 10, every 5 sessions', mom, liquid, None, 10, 5),
        ('V1 12-1 / volatility', s['12-1 / volatility'], liquid, None, 10, 5),
        ('V2 6m+12m risk-adjusted composite (NSE momentum-index style)',
         s['6m+12m risk-adjusted composite'], liquid, None, 10, 5),
        ('V3 buy only smooth winners (frog-in-the-pan ID in the better half)', mom, liquid, smooth, 10, 5),
        (f'V4 buy only without a {JUMP:.0%}+ day in the last 63 sessions', mom, liquid, nojump, 10, 5),
        (f'V5 buy only within {1 - NEAR_HIGH:.0%} of the 52-week high', mom, liquid, near, 10, 5),
        ('V6 rebalance every 21 sessions', mom, liquid, None, 10, 21),
        ('V7 top 20', mom, liquid, None, 20, 5),
        ('V8 = V1 + V4', s['12-1 / volatility'], liquid, nojump, 10, 5),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Predictive power
# ═════════════════════════════════════════════════════════════════════════════
def sample_rows(P, s, liquid, step=FWD):
    mom = s['12-1 momentum (B4)']
    ok = (liquid & np.isfinite(mom)).sum(axis=1) >= 30
    first = int(np.argmax(ok))
    return [t for t in range(first, len(P.dates) - FWD, step) if ok[t]]


def predictive(P, s, liquid, rows):
    """{score: DataFrame(date, ic, top10_excess, top10_hit)} on `rows`."""
    c = P.df['close']
    fwd = (c.shift(-FWD) / c - 1).to_numpy()
    out = {}
    for name, x in s.items():
        recs = []
        for t in rows:
            m = liquid[t] & np.isfinite(x[t]) & np.isfinite(fwd[t])
            if m.sum() < 30:
                continue
            xs, ys = x[t][m], fwd[t][m]
            ic = pd.Series(xs).rank().corr(pd.Series(ys).rank())
            top = np.argsort(-xs)[:10]
            recs.append((P.dates[t], ic, ys[top].mean() - ys.mean(), (ys[top] > np.median(ys)).mean()))
        out[name] = pd.DataFrame(recs, columns=['date', 'ic', 'excess', 'hit']).set_index('date')
    return out


def by_level(P, s, liquid, rows):
    """Mean forward excess return of eligible stocks by their 12-1 level."""
    c = P.df['close']
    fwd = (c.shift(-FWD) / c - 1).to_numpy()
    mom = s['12-1 momentum (B4)']
    acc = {lab: [] for *_, lab in LEVELS}
    for t in rows:
        m = liquid[t] & np.isfinite(mom[t]) & np.isfinite(fwd[t])
        if m.sum() < 30:
            continue
        ex = fwd[t][m] - fwd[t][m].mean()
        for lo, hi, lab in LEVELS:
            k = (mom[t][m] >= lo) & (mom[t][m] < hi)
            acc[lab] += list(ex[k])
    return acc


def rerank(P, s, liquid, rows, pool=30, n=10):
    """Inside B4's top `pool` by 12-1, the mean forward return of the top `n`
    by each other score, minus B4's own top `n`."""
    c = P.df['close']
    fwd = (c.shift(-FWD) / c - 1).to_numpy()
    mom = s['12-1 momentum (B4)']
    out = {k: [] for k in s if k != '12-1 momentum (B4)'}
    for t in rows:
        m = liquid[t] & np.isfinite(mom[t]) & np.isfinite(fwd[t])
        idx = np.flatnonzero(m)
        if len(idx) < pool:
            continue
        cand = idx[np.argsort(-mom[t][idx])[:pool]]
        base = fwd[t][cand[:n]].mean()
        for k in out:
            x = s[k][t][cand]
            pick = cand[np.argsort(-np.where(np.isfinite(x), x, -np.inf))[:n]]
            out[k].append(fwd[t][pick].mean() - base)
    return out


def _tstat(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


# ═════════════════════════════════════════════════════════════════════════════
# Live window
# ═════════════════════════════════════════════════════════════════════════════
def equity_series_map(get=_get_text):
    """{symbol: series} from NSE's EQUITY_L.csv; {} when NSE can't be reached."""
    for url in LIST_URLS:
        text = get(url)
        if not text:
            continue
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            continue
        df.columns = [str(col).strip().upper() for col in df.columns]
        if {'SYMBOL', 'SERIES'} <= set(df.columns):
            return {str(a).strip(): str(b).strip() for a, b in zip(df['SYMBOL'], df['SERIES'])}
    return {}


def live_ledger(state_dir):
    """(equity curve, positions, closed trades) of the live momentum sleeve, or Nones."""
    def read(name):
        p = os.path.join(state_dir, name)
        return pd.read_csv(p) if os.path.exists(p) else None
    eq = read('equity.csv')
    trades = read('trades.csv')
    positions = []
    p = os.path.join(state_dir, 'state.json')
    if os.path.exists(p):
        with open(p) as fh:
            positions = json.load(fh).get('positions', [])
    return eq, positions, trades


def live_ca(P, events, positions, trades):
    """Corporate-action gaps the live ledger carried a holding through:
    (symbol, date, shares, rupees booked across the gap)."""
    spans = [(p['symbol'], int(p['qty']), pd.Timestamp(p['entry_time']).normalize(), P.dates[-1])
             for p in positions]
    if trades is not None:
        spans += [(r.symbol, int(r.qty), pd.Timestamp(r.entry_time).normalize(),
                   pd.Timestamp(r.exit_time).normalize()) for r in trades.itertuples()]
    out = []
    o, c = P.a['open'], P.a['close']
    for sym, q, a, b in spans:
        if sym not in P.symbols:
            continue
        j = P.symbols.index(sym)
        for t in np.flatnonzero(events[:, j]):
            if a < P.dates[t] <= b:
                out.append((sym, P.dates[t].date(), q, q * (c[t - 1, j] - o[t, j])))
    return out


def picks_at(P, Q, score, liquid, buy_filter, t0, n=10):
    """The top n buyable names at t0, and their equal-weight return from the
    next open to the last close on corporate-action-adjusted prices."""
    sc = np.where(liquid[t0] & np.isfinite(score[t0]), score[t0], -np.inf)
    if buy_filter is not None:
        sc = np.where(buy_filter[t0], sc, -np.inf)
    top = [j for j in np.argsort(-sc)[:n] if np.isfinite(sc[j])]
    if t0 + 1 >= len(P.dates) or not top:
        return [], np.nan
    o, c = Q.a['open'], Q.a['close']
    last = np.array([c[np.flatnonzero(np.isfinite(c[:, j]))[-1], j] for j in top])
    ret = np.nanmean(last / o[t0 + 1, top] - 1)
    return [P.symbols[j] for j in top], ret


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════
def _p(v, signed=True, nd=1):
    if v is None or not np.isfinite(v):
        return '—'
    return f'{v * 100:+.{nd}f}%' if signed else f'{v * 100:.{nd}f}%'


def run(out_dir, fetch=None, get_list=None, get_series=None, state_dir=os.path.join('state', 'momentum')):
    from ..universe import INDEX_SYMBOL

    fetch = fetch or fetch_range
    os.makedirs(out_dir, exist_ok=True)
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    symbols, note = (get_list or (lambda: nse_equities(os.path.join(out_dir, 'nse_equity_list.json'))))()
    series_of = (get_series or equity_series_map)()
    raw = fetch(symbols, FETCH_START, end)
    broad = {s_: d for s_, d in raw.items() if ever_eligible(d)}
    bench = fetch(list(BENCHMARKS), FETCH_START, end)
    index_df = bench.get(INDEX_SYMBOL)
    if index_df is None:
        index_df = fetch([INDEX_SYMBOL], FETCH_START, end)[INDEX_SYMBOL]
    idx_dates = pd.DatetimeIndex(pd.to_datetime(index_df['datetime']))
    on_index = {s_: d[pd.to_datetime(d['datetime']).isin(idx_dates)].reset_index(drop=True)
                for s_, d in broad.items()}
    P = Panel({s_: d for s_, d in on_index.items() if len(d) >= 60})
    liquid = P.liquid
    last = P.dates[-1]

    events = ca_events(P)
    Q = adjusted(P, events)
    s = scores(P, liquid)
    s_adj = scores(Q, liquid)
    rows = sample_rows(P, s_adj, liquid)

    md = ['# Diagnosis — the live book\'s first weeks, and what each fix would have changed', '',
          '**Post-hoc and exploratory.** Written after the live losses were seen; every variant was chosen '
          'because of them, and 2018–26 has been studied before. Effect sizes, not evidence: a variant earns '
          'deployment only through a pre-registered forward test.', '',
          f'Universe: {note}; {len(broad)} ever eligible; Nifty\'s calendar (as the live engine counts). '
          f'Data to {last.date()}. ₹50k books, Zerodha CNC costs, next-open fills, no fills on circuit-locked '
          f'bars. Delisted stocks are missing, which flatters every momentum figure.', '']

    # ── 1 live window ────────────────────────────────────────────────────────
    md += ['## 1. The live window', '']
    eq, positions, trades = live_ledger(state_dir)
    t0 = int(np.searchsorted(P.dates, LIVE_START))
    if t0 < len(P.dates) and P.dates[t0] == LIVE_START and t0 + 1 < len(P.dates):
        md += [f'From the {LIVE_START.date()} close (the sleeve\'s first decision) to {last.date()}:', '',
               '| | return |', '|---|---|']
        if eq is not None and len(eq):
            e = eq.set_index(pd.to_datetime(eq['session']))['equity'].astype(float)
            md.append(f'| Momentum sleeve, as booked | {_p(e.iloc[-1] / e.iloc[0] - 1)} |')
            hits = live_ca(P, events, positions, trades)
            if hits:
                booked = sum(h[3] for h in hits)
                md.append(f'| Momentum sleeve, value kept across corporate-action gaps (est.) | '
                          f'{_p((e.iloc[-1] + booked) / e.iloc[0] - 1)} |')
        for sym, label in BENCHMARKS.items():
            d = bench.get(sym)
            if d is not None and len(d):
                b = d.set_index(pd.to_datetime(d['datetime']))['close'].astype(float)
                if LIVE_START in b.index:
                    md.append(f'| {label} | {_p(b.iloc[-1] / b.loc[LIVE_START] - 1)} |')
        cq = Q.a['close']
        lastpx = pd.DataFrame(cq, index=P.dates).ffill().to_numpy()[-1]
        m = liquid[t0] & np.isfinite(cq[t0]) & np.isfinite(lastpx)
        md.append(f'| Equal-weight hold of every eligible NSE stock (no costs, {int(m.sum())} names) | '
                  f'{_p(np.mean(lastpx[m] / cq[t0][m] - 1))} |')
        md += ['', '_Buy-and-hold of each design\'s picks on that date, equal weight, next-open entry, no costs, '
                   'prices adjusted for corporate-action gaps. Ten stocks over a few weeks: an anecdote, not a test._',
               '', '| design | picks on ' + str(LIVE_START.date()) + ' | return to ' + str(last.date()) + ' |',
               '|---|---|---|']
        for name, sc, _, bf, n, every in variants(Q, liquid, s_adj):
            if n != 10 or every != 5:                   # same picks as B4 on any one date
                continue
            picks, ret = picks_at(P, Q, sc, liquid, bf, t0)
            flag = [f'{p}{"*" if series_of.get(p, "EQ") != "EQ" else ""}' for p in picks]
            md.append(f'| {name} | {", ".join(flag)} | {_p(ret)} |')
        md += ['', '_* not in series EQ today (BE = trade-for-trade surveillance, BZ = non-compliant)._', '']
        if eq is not None:
            hits = live_ca(P, events, positions, trades)
            if hits:
                md += ['**Live holdings carried through a corporate-action gap:**', '',
                       '| symbol | ex-date | shares | booked across the gap |', '|---|---|---|---|']
                md += [f'| {a} | {b} | {q} | ₹{-v:+,.0f} |' for a, b, q, v in hits]
                md.append('')
        held_series = {}
        for p in positions:
            held_series.setdefault(series_of.get(p['symbol'], '?'), []).append(p['symbol'])
        if held_series:
            md += ['Live holdings by NSE series today: ' + '; '.join(
                f'{k}: {", ".join(v)}' for k, v in sorted(held_series.items())), '']
    else:
        md += [f'_{LIVE_START.date()} is not in the data — live window skipped._', '']

    # ── 2 corporate actions ──────────────────────────────────────────────────
    buy_ok, sell_ok = locked_bars(P)
    mom_raw = s['12-1 momentum (B4)']
    eq_raw, costs_raw, held = rotate(P, mom_raw, liquid, buy_ok=buy_ok, sell_ok=sell_ok, log=True)
    hits = ca_hits(P, events, held)
    ev_per_year = pd.Series(P.dates[np.nonzero(events)[0]].year).value_counts().sort_index()
    q_ok, q_sell = locked_bars(Q)
    eq_adj, _, _ = rotate(Q, s_adj['12-1 momentum (B4)'], liquid, buy_ok=q_ok, sell_ok=q_sell)
    md += ['## 2. Corporate actions the ledger books as losses', '',
           f'Sessions that opened **and** closed {abs(GAP):.0%}+ under the previous close: **{int(events.sum())}** '
           f'across {int(events.any(axis=0).sum())} stocks '
           f'({", ".join(f"{y}: {n}" for y, n in ev_per_year.items())}).', '',
           f'B4 (phase 0) held a stock through **{len(hits)}** of them; the ledger booked '
           f'**₹{-sum(h[3] for h in hits):+,.0f}** across those gaps, on a ₹50k book.', '']
    if hits:
        md += ['| ex-date | symbol | shares | gap | booked |', '|---|---|---|---|---|']
        md += [f'| {d.date()} | {sym} | {q} | {_p(g)} | ₹{-v:+,.0f} |' for d, sym, q, v, g in hits]
        md.append('')
    md += ['| B4, phase 0 | CAGR Jul 2018 – Jul 2023 | CAGR Jul 2018 – latest | CAGR Jun 2024 – latest | max DD |',
           '|---|---|---|---|---|']
    for name, e in (('raw Yahoo prices (as researched and as the ledger books)', eq_raw),
                    ('prices adjusted for the gaps (value kept, as a holder gets it)', eq_adj)):
        a, b, r = curve(e, WIN_START, WIN_END), curve(e, WIN_START, last), curve(e, RECENT_START, last)
        md.append(f'| {name} | {_p(a and a["cagr"])} | {_p(b and b["cagr"])} | {_p(r and r["cagr"])} | '
                  f'{_p(b and b["maxdd"], False)} |')
    md.append('')

    # ── 3 predictive power ───────────────────────────────────────────────────
    periods = (('untouched 2018–23', WIN_START, WIN_END), ('researched 2023–', RESEARCHED_START, last))
    pred = predictive(Q, s_adj, liquid, rows)
    md += ['## 3. Which score predicts the next month', '',
           f'{len(rows)} non-overlapping {FWD}-session windows; every eligible stock; corporate-action-adjusted '
           f'prices. IC = rank correlation of the score with the forward return. Top-10 excess = the 10 '
           f'best-scored names\' forward return minus the eligible average (what a 10-stock book earns '
           f'over the market, before costs, per {FWD} sessions).', '',
           '| score | ' + ' | '.join(f'IC {p} (t) | top-10 excess {p} (t)' for p, *_ in periods) + ' |',
           '|---|' + '---|---|' * len(periods)]
    for name, df in pred.items():
        cells = []
        for _, a, b in periods:
            d = df[(df.index >= a) & (df.index <= b)]
            cells += [f'{d["ic"].mean():+.3f} ({_tstat(d["ic"]):+.1f})' if len(d) else '—',
                      f'{_p(d["excess"].mean(), nd=2)} ({_tstat(d["excess"]):+.1f})' if len(d) else '—']
        md.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lv = by_level(Q, s_adj, liquid, rows)
    md += ['', f'**Forward {FWD}-session excess return by 12-1 momentum level** (all windows):', '',
           '| 12-1 level | stock-windows | mean excess (t) | share beating the average |', '|---|---|---|---|']
    for *_, lab in LEVELS:
        x = np.asarray(lv[lab], dtype=float)
        if len(x):
            md.append(f'| {lab} | {len(x):,} | {_p(x.mean(), nd=2)} ({_tstat(x):+.1f}) | {(x > 0).mean():.0%} |')
    rr = rerank(Q, s_adj, liquid, rows)
    md += ['', f'**Re-ranking inside B4\'s top 30**: forward {FWD}-session return of the top 10 by each '
               f'score, minus B4\'s own top 10 (positive = a better 10 from the same 30):', '',
           '| re-rank by | mean difference (t) | windows better |', '|---|---|---|']
    for k, x in rr.items():
        x = np.asarray(x, dtype=float)
        if len(x):
            md.append(f'| {k} | {_p(np.nanmean(x), nd=2)} ({_tstat(x):+.1f}) | {(x > 0).mean():.0%} |')
    md.append('')

    # ── 4 variants ───────────────────────────────────────────────────────────
    md += ['## 4. Candidate books at every rebalance phase', '',
           f'{PHASES} runs each (the first rebalance delayed 0–{PHASES - 1} sessions); median [min – max]. '
           f'Corporate-action-adjusted prices for every row, so the B4 row here is the fair baseline. '
           f'Untouched = Jul 2018 – Jul 2023; full = Jul 2018 – {last.date()}; recent = '
           f'{RECENT_START.date()} – {last.date()}. Costs are Zerodha CNC charges per year as a share of '
           f'average equity; short-term capital-gains tax is not modelled and hits a faster rotation harder.', '',
           '| design | CAGR untouched | CAGR full | CAGR recent | worst max DD | Sharpe (full) | costs/yr |',
           '|---|---|---|---|---|---|---|']
    csv_rows = []
    for name, sc, elig, bf, n, every in variants(Q, liquid, s_adj):
        runs = phase_curves(Q, sc, elig, buy_filter=bf, n=n, every=every)
        stats = []
        for k, (e, costs) in enumerate(runs):
            a, b, r = curve(e, WIN_START, WIN_END), curve(e, WIN_START, last), curve(e, RECENT_START, last)
            cs = curve_stats(e[e.index >= WIN_START])
            span = e[e.index >= WIN_START].dropna()
            yrs = max(len(e.dropna()) / 252.0, 1e-9)
            stats.append((a and a['cagr'], b and b['cagr'], r and r['cagr'], b and b['maxdd'],
                          cs.get('sharpe', np.nan), costs / yrs / span.mean() if len(span) else np.nan))
            csv_rows.append({'design': name, 'phase': k, 'cagr_untouched': stats[-1][0],
                             'cagr_full': stats[-1][1], 'cagr_recent': stats[-1][2], 'maxdd': stats[-1][3],
                             'sharpe': stats[-1][4], 'costs_per_year': stats[-1][5]})
        cols = [np.array([x[i] if x[i] is not None else np.nan for x in stats], dtype=float) for i in range(6)]

        def mm(v):
            v = v[np.isfinite(v)]
            return f'{_p(np.median(v))} [{_p(v.min())} – {_p(v.max())}]' if len(v) else '—'
        md.append(f'| {name} | {mm(cols[0])} | {mm(cols[1])} | {mm(cols[2])} | '
                  f'{_p(np.nanmin(cols[3]), False)} | {np.nanmedian(cols[4]):.2f} | {_p(np.nanmedian(cols[5]), False, 2)} |')
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, 'diagnosis_variants.csv'), index=False)
    md += ['', '_Nine designs on one history: the best of them is flattered by selection. A design that beats '
               'B4 at every phase, in both windows, with a shallower drawdown is a candidate for a pre-registered '
               'forward test — nothing more._', '']
    return _write(out_dir, md)


def _write(out_dir, md):
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'DIAGNOSIS.md'), 'w') as fh:
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
