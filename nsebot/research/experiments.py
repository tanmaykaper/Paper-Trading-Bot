"""Phase 2b — pre-registered signal experiments with an out-of-sample split.

    python -m nsebot.research.experiments --out research_results

Every variant below was written down BEFORE any result was seen, and each is
judged twice: on the first 60% of the history (in-sample, where a variant
would be chosen) and on the last 40% (out-of-sample, which never informed the
choice). Only the out-of-sample column is evidence. Reporting every variant —
including the losers — is what keeps a parameter sweep from turning into a
curve fit.

Acceptance rules (also fixed in advance):
  event strategies  IS net/trade > 0 on >= 100 trades, AND OOS net/trade > 0
                    with t-stat >= 1.0
  rotation          beats the equal-weight liquid universe on CAGR AND Sharpe
                    in BOTH halves
  intraday          OOS net/trade > 0 on >= 30 trades — labelled low
                    confidence regardless, because Yahoo serves only ~59 days

All entries fill at the NEXT bar's open; every exit is gap-aware; Zerodha
costs are charged on every leg.
"""

import argparse
import logging
import os
import sys
from datetime import time

import numpy as np
import pandas as pd

from ..config import IntradaySignalConfig
from ..costs import DEFAULT_CHARGES
from ..data import YahooProvider
from ..indicators import atr, ema, median_turnover
from ..regime import RegimeDial
from ..signals import IntradaySignalEngine, SwingSignalEngine
from ..universe import INDEX_SYMBOL, INTRADAY_UNIVERSE, SWING_UNIVERSE

logger = logging.getLogger(__name__)

IS_FRACTION = 0.6
EVENT_NOTIONAL = 15_000.0
ROTATION_CAPITAL = 50_000.0
INTRADAY_NOTIONAL = 50_000.0
SQUARE_OFF = time(15, 5)          # the 15:05 bar closes at 15:10


# ═════════════════════════════════════════════════════════════════════════════
# Shared statistics
# ═════════════════════════════════════════════════════════════════════════════
def trade_stats(t):
    if t is None or len(t) == 0:
        return {'n': 0}
    r, net = t['r'].astype(float), t['net_pct'].astype(float)
    wins, losses = r[r > 0], r[r <= 0]
    payoff = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) and losses.mean() else np.nan
    p = (r > 0).mean()
    kelly = p - (1 - p) / payoff if payoff and np.isfinite(payoff) and payoff > 0 else np.nan
    sd = net.std(ddof=1)
    return {'n': len(t), 'win': p, 'payoff': payoff, 'avg_r': r.mean(), 'net': net.mean(),
            't': net.mean() / (sd / np.sqrt(len(t))) if len(t) > 1 and sd > 0 else np.nan,
            'kelly': kelly}


def _fmt_stats(s):
    if not s or s.get('n', 0) == 0:
        return '0 | — | — | — | — | —'
    return (f"{s['n']} | {s['win'] * 100:.0f}% | {s['avg_r']:+.2f} | {s['net'] * 100:+.2f}% | "
            f"{s['t']:+.1f} | {s['kelly']:+.3f}")


def curve_stats(equity):
    eq = equity.dropna()
    if len(eq) < 20:
        return {}
    ret = eq.pct_change().dropna()
    years = len(ret) / 252.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return {'cagr': cagr, 'vol': vol, 'sharpe': (ret.mean() * 252) / vol if vol > 0 else np.nan,
            'maxdd': dd}


def _fmt_curve(s):
    if not s:
        return '— | — | —'
    return f"{s['cagr'] * 100:+.1f}% | {s['sharpe']:.2f} | {s['maxdd'] * 100:.1f}%"


# ═════════════════════════════════════════════════════════════════════════════
# Swing data panel
# ═════════════════════════════════════════════════════════════════════════════
class Panel:
    """Wide (date x symbol) numpy arrays, one alignment for every experiment."""

    def __init__(self, universe):
        frames = {s: df.set_index(pd.to_datetime(df['datetime'])) for s, df in universe.items()
                  if df is not None and len(df) >= 60}
        self.symbols = sorted(frames)
        idx = sorted(set().union(*[f.index for f in frames.values()]))
        self.dates = pd.DatetimeIndex(idx)

        def wide(fn):
            return pd.DataFrame({s: fn(frames[s]) for s in self.symbols}).reindex(self.dates)

        self.df = {k: wide(lambda f, k=k: f[k].astype(float))
                   for k in ('open', 'high', 'low', 'close', 'volume')}
        self.df['atr'] = wide(lambda f: atr(f.reset_index(drop=True)).set_axis(f.index))
        self.df['ema5'] = wide(lambda f: ema(f['close'].astype(float), 5))
        self.df['ema200'] = wide(lambda f: ema(f['close'].astype(float), 200))
        self.df['turnover'] = wide(lambda f: median_turnover(f.reset_index(drop=True)).set_axis(f.index))
        self.a = {k: v.to_numpy() for k, v in self.df.items()}
        c = self.df['close']
        self.liquid = ((self.df['turnover'] >= 5e7) & (c >= 50)).to_numpy()
        self.cut = int(len(self.dates) * IS_FRACTION)

    def segment(self, i):
        return 'IS' if i < self.cut else 'OOS'


# ═════════════════════════════════════════════════════════════════════════════
# Event-trade simulator (one position per symbol at a time)
# ═════════════════════════════════════════════════════════════════════════════
def simulate_events(P, signal, stop, exit_rule, hold=15, max_hold=40, notional=EVENT_NOTIONAL):
    o, h, l, c = P.a['open'], P.a['high'], P.a['low'], P.a['close']
    at, e5 = P.a['atr'], P.a['ema5']
    T, N = signal.shape
    rows = []
    for j in range(N):
        busy_until = -1
        for t in np.flatnonzero(signal[:, j]):
            if t <= busy_until or t + 1 >= T:
                continue
            entry, s0 = o[t + 1, j], stop[t, j]
            if not (np.isfinite(entry) and np.isfinite(s0)) or entry <= s0:
                continue
            risk, s, best = entry - s0, s0, entry
            limit = hold if exit_rule in ('time', 'reversion') else max_hold
            exit_px, k_exit, last_c = None, None, None
            for k in range(t + 1, min(T, t + 1 + limit)):
                ok, hk, lk, ck = o[k, j], h[k, j], l[k, j], c[k, j]
                if not np.isfinite(ck):
                    continue
                last_c, k_exit = ck, k
                if k > t + 1 and ok <= s:
                    exit_px = ok
                    break
                if lk <= s:
                    exit_px = s
                    break
                held = k - t
                if exit_rule == 'reversion' and ck > e5[k, j]:
                    exit_px = ck
                    break
                if exit_rule == 'trail':
                    best = max(best, hk)
                    br = (best - entry) / risk
                    if held >= 10 and (ck - entry) / risk < 0.5:
                        exit_px = ck
                        break
                    if br >= 1.0 and np.isfinite(at[k, j]):
                        mult = 2.0 if br >= 3.0 else 3.0
                        s = min(max(s, entry * 1.001, best - mult * at[k, j]), ck * 0.999)
            if exit_px is None:
                if last_c is None or k_exit is None or k_exit >= T - 1:
                    continue                      # truncated by the end of data
                exit_px = last_c                  # time limit reached
            busy_until = k_exit
            qty = max(int(notional // entry), 1)
            cost = DEFAULT_CHARGES.cnc(entry * qty, exit_px * qty)
            rows.append({'date': P.dates[t], 'seg': P.segment(t), 'symbol': P.symbols[j],
                         'entry': entry, 'exit': exit_px, 'r': (exit_px - entry) / risk,
                         'net_pct': ((exit_px - entry) * qty - cost) / (entry * qty),
                         'held': k_exit - t})
    return pd.DataFrame(rows)


def top_k_per_day(signal, score, k):
    """Keep only the k best-scored signals each day — the book's real capacity."""
    out = np.zeros_like(signal)
    sc = np.where(signal, np.nan_to_num(score, nan=-1e9), -np.inf)
    for t in np.flatnonzero(signal.any(axis=1)):
        idx = np.argsort(-sc[t])[:k]
        out[t, idx] = signal[t, idx]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Rotation simulator (equal weight, integer shares, CNC costs)
# ═════════════════════════════════════════════════════════════════════════════
def simulate_rotation(P, score, eligible, n=5, band=None, every=5, capital=ROTATION_CAPITAL,
                      risk_on=None):
    """risk_on: optional per-date bool array. On a rebalance date where it is
    False the book goes to cash (sells everything at the next open, buys
    nothing) — the classic absolute-momentum / 200-day market filter."""
    o, c = P.a['open'], P.a['close']
    T, N = c.shape
    band = band or 3 * n
    cash, qty = capital, np.zeros(N, dtype=int)
    last_px = np.full(N, np.nan)
    equity = np.full(T, np.nan)
    costs = 0.0
    start = int(np.argmax(np.isfinite(score).sum(axis=1) >= 2 * n))
    for t in range(T):
        last_px = np.where(np.isfinite(c[t]), c[t], last_px)
        equity[t] = cash + np.nansum(qty * last_px)
        if t < start or (t - start) % every or t + 1 >= T:
            continue
        sc = np.where(eligible[t] & np.isfinite(score[t]), score[t], -np.inf)
        order = np.argsort(-sc)
        rank = np.empty(N, dtype=int)
        rank[order] = np.arange(N)
        valid = np.isfinite(sc)
        off = risk_on is not None and not bool(risk_on[t])
        # Sell: market filter off, no longer eligible, or fallen out of the band.
        for j in np.flatnonzero(qty > 0):
            if (off or not valid[j] or rank[j] >= band) and np.isfinite(o[t + 1, j]):
                v = qty[j] * o[t + 1, j]
                fee = DEFAULT_CHARGES.cnc(0.0, v)
                cash += v - fee
                costs += fee
                qty[j] = 0
        # Buy: fill empty slots from the top of the ranking.
        slots = 0 if off else n - int((qty > 0).sum())
        target = equity[t] / n
        for j in order:
            if slots <= 0 or not valid[j]:
                break
            if qty[j] > 0 or not np.isfinite(o[t + 1, j]):
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
    return pd.Series(equity, index=P.dates), costs


def benchmark_equal_weight(P):
    held = pd.DataFrame(P.liquid, index=P.dates, columns=P.symbols).shift(1, fill_value=False)
    r = P.df['close'].pct_change().where(held.astype(bool))
    return (1 + r.mean(axis=1).fillna(0)).cumprod()


# ═════════════════════════════════════════════════════════════════════════════
# Swing experiments
# ═════════════════════════════════════════════════════════════════════════════
def swing_experiments(universe, index_df):
    P = Panel(universe)
    eng = SwingSignalEngine()
    feats = {s: eng.features(universe[s]).set_index('datetime') for s in P.symbols}
    trig = pd.DataFrame({s: feats[s]['trigger'] != '' for s in P.symbols}).reindex(P.dates) \
        .fillna(False).to_numpy(bool)
    eng_stop = pd.DataFrame({s: feats[s]['stop'] for s in P.symbols}).reindex(P.dates).to_numpy()
    events, _ = eng.panel(universe)
    score = np.full(trig.shape, np.nan)
    if len(events):
        sc = events.pivot_table(index='datetime', columns='symbol', values='score', aggfunc='last')
        score = sc.reindex(index=P.dates, columns=P.symbols).to_numpy()
    regime = RegimeDial().panel(index_df, universe)['score'].reindex(P.dates).ffill().to_numpy()
    weak = (regime < 0.33)[:, None]

    c, at = P.a['close'], P.a['atr']
    ret3 = c / np.roll(c, 3, axis=0) - 1.0
    ret3[:3] = np.nan
    uptrend = P.liquid & (c > P.a['ema200'])
    rev_stop = c - 3.0 * at

    event_variants = {
        'S1 breakout/thrust, 15-day time exit (Phase 2 baseline)': (trig, eng_stop, 'time'),
        'S1t breakout/thrust, chandelier trail': (trig, eng_stop, 'trail'),
        'S1t breakout/thrust, trail, top 2/day': (top_k_per_day(trig, score, 2), eng_stop, 'trail'),
        'S2 breakout/thrust in WEAK tape, time exit': (trig & weak, eng_stop, 'time'),
        'S2t breakout/thrust in WEAK tape, trail': (trig & weak, eng_stop, 'trail'),
        'S4a reversion: >5% 3-day drop above EMA200, exit > EMA5 (max 7d)':
            (uptrend & (ret3 <= -0.05), rev_stop, 'reversion'),
        'S4b reversion: >8% 3-day drop above EMA200, exit > EMA5 (max 7d)':
            (uptrend & (ret3 <= -0.08), rev_stop, 'reversion'),
    }
    event_rows = []
    for name, (sig, stop, rule) in event_variants.items():
        t = simulate_events(P, sig, stop, rule, hold=7 if rule == 'reversion' else 15)
        is_s = trade_stats(t[t['seg'] == 'IS']) if len(t) else {'n': 0}
        oos_s = trade_stats(t[t['seg'] == 'OOS']) if len(t) else {'n': 0}
        accepted = (is_s.get('n', 0) >= 100 and is_s.get('net', -1) > 0
                    and oos_s.get('n', 0) > 0 and oos_s.get('net', -1) > 0
                    and (oos_s.get('t') or 0) >= 1.0)
        event_rows.append((name, is_s, oos_s, accepted))

    # Rotation
    cdf = P.df['close']
    mom126 = (cdf / cdf.shift(126) - 1).to_numpy()
    mom12_1 = (cdf.shift(21) / cdf.shift(252) - 1).to_numpy()
    above200 = P.liquid & (c > P.a['ema200'])
    rot_variants = {
        'R1 6-month momentum, top 5, weekly': (mom126, P.liquid, 5),
        'R2 6-month momentum, top 10, weekly': (mom126, P.liquid, 10),
        'R3 12-1 momentum, top 5, weekly': (mom12_1, P.liquid, 5),
        'R4 12-1 momentum, top 10, weekly': (mom12_1, P.liquid, 10),
        'R5 6-month momentum, top 5, only names above EMA200': (mom126, above200, 5),
    }
    bench = benchmark_equal_weight(P)
    nifty = index_df.set_index(pd.to_datetime(index_df['datetime']))['close'].reindex(P.dates).ffill()
    cut_date = P.dates[P.cut]
    halves = {'IS': lambda s: s[s.index < cut_date], 'OOS': lambda s: s[s.index >= cut_date]}
    bench_s = {k: curve_stats(f(bench)) for k, f in halves.items()}
    nifty_s = {k: curve_stats(f(nifty)) for k, f in halves.items()}
    rot_rows = []
    for name, (sc, elig, n) in rot_variants.items():
        eq, cost = simulate_rotation(P, sc, elig, n=n)
        s = {k: curve_stats(f(eq)) for k, f in halves.items()}
        beats = all(s[k] and bench_s[k] and s[k]['cagr'] > bench_s[k]['cagr']
                    and s[k]['sharpe'] > bench_s[k]['sharpe'] for k in halves)
        rot_rows.append((name, s, beats, cost))

    md = ['## Swing — event strategies (₹15k per trade, CNC costs)', '',
          f"History {P.dates[0].date()} → {P.dates[-1].date()} ({len(P.dates)} sessions, "
          f"{len(P.symbols)} symbols). In-sample to {cut_date.date()}, out-of-sample after.", '',
          '| variant | IS n | IS win | IS avg R | IS net/trade | IS t | IS Kelly | '
          'OOS n | OOS win | OOS avg R | OOS net/trade | OOS t | OOS Kelly | accepted |',
          '|---|' + '---|' * 13]
    for name, a, b, ok in event_rows:
        md.append(f"| {name} | {_fmt_stats(a)} | {_fmt_stats(b)} | {'**YES**' if ok else 'no'} |")
    md += ['', f'## Swing — rotation (₹{ROTATION_CAPITAL:,.0f} book, integer shares, CNC costs)', '',
           '| variant | IS CAGR | IS Sharpe | IS maxDD | OOS CAGR | OOS Sharpe | OOS maxDD | costs ₹ | accepted |',
           '|---|---|---|---|---|---|---|---|---|',
           f"| benchmark: equal-weight liquid universe (no costs) | {_fmt_curve(bench_s['IS'])} | "
           f"{_fmt_curve(bench_s['OOS'])} | — | — |",
           f"| benchmark: Nifty 50 | {_fmt_curve(nifty_s['IS'])} | {_fmt_curve(nifty_s['OOS'])} | — | — |"]
    for name, s, ok, cost in rot_rows:
        md.append(f"| {name} | {_fmt_curve(s['IS'])} | {_fmt_curve(s['OOS'])} | {cost:,.0f} | "
                  f"{'**YES**' if ok else 'no'} |")
    return '\n'.join(md), event_rows, rot_rows


# ═════════════════════════════════════════════════════════════════════════════
# Intraday experiments
# ═════════════════════════════════════════════════════════════════════════════
def _simulate_intraday(f, entries, stops):
    """entries: list of (row index, side). Entry next bar open; exit at stop or 15:10."""
    rows = []
    for i, side in entries:
        e = f.loc[i]
        day = f[(f['session'] == e['session']) & (f.index > i)]
        if day.empty:
            continue
        entry = float(day['open'].iloc[0])
        stop = float(stops[i])
        sgn = 1.0 if side == 'LONG' else -1.0
        if sgn * (entry - stop) <= 0:
            continue
        exit_px = None
        for _, b in day.iterrows():
            if side == 'LONG' and b['low'] <= stop:
                exit_px = min(stop, b['open'])
                break
            if side == 'SHORT' and b['high'] >= stop:
                exit_px = max(stop, b['open'])
                break
            exit_px = b['close']
            if b['datetime'].time() >= SQUARE_OFF:
                break
        qty = max(int(INTRADAY_NOTIONAL // entry), 1)
        cost = DEFAULT_CHARGES.mis(entry * qty, exit_px * qty)
        rows.append({'session': e['session'], 'side': side, 'r': sgn * (exit_px - entry) / abs(entry - stop),
                     'net_pct': (sgn * (exit_px - entry) * qty - cost) / (entry * qty)})
    return rows


def _first_per_session(mask, f):
    return mask & (mask.groupby(f['session']).cumsum() == 1)


def intraday_experiments(bars_by_symbol, daily_by_symbol, index_bars):
    wide_cfg = IntradaySignalConfig(entry_end=time(14, 0))
    variants = {'I1 ORB baseline (Phase 2)': [], 'I2 ORB, stop at range opposite side, window to 14:00': [],
                'I3 VWAP pullback in trend, 10:30-14:00': []}
    sessions = set()
    for sym, bars in bars_by_symbol.items():
        if bars is None or len(bars) < 150:
            continue
        daily = daily_by_symbol.get(sym)
        f = IntradaySignalEngine().session_features(bars, daily, index_bars)
        sessions.update(f['session'].unique())
        # I1
        ent = [(i, 'LONG') for i in f.index[f['first_long']]] + [(i, 'SHORT') for i in f.index[f['first_short']]]
        stops = np.where(f['first_long'], f['long_stop'], f['short_stop'])
        variants['I1 ORB baseline (Phase 2)'] += _simulate_intraday(f, ent, pd.Series(stops, index=f.index))
        # I2
        g = IntradaySignalEngine(wide_cfg).session_features(bars, daily, index_bars)
        close = g['close']
        lstop = close - np.clip(close - g['or_low'], 0.003 * close, 0.025 * close)
        sstop = close + np.clip(g['or_high'] - close, 0.003 * close, 0.025 * close)
        ent = [(i, 'LONG') for i in g.index[g['first_long']]] + [(i, 'SHORT') for i in g.index[g['first_short']]]
        stops = pd.Series(np.where(g['first_long'], lstop, sstop), index=g.index)
        variants['I2 ORB, stop at range opposite side, window to 14:00'] += _simulate_intraday(g, ent, stops)
        # I3 — trend day (beyond the opening range), pullback touches VWAP, bar closes back with conviction
        tod = f['datetime'].dt.time
        win = (tod >= time(10, 30)) & (tod < time(14, 0))
        base = win & (f['or_rvol'] >= 1.5) & f['or_complete'] & (f['turnover'] >= 5e8)
        if index_bars is not None and len(index_bars):
            bias = IntradaySignalEngine._index_bias(index_bars)
            b = pd.merge_asof(f[['datetime']], bias, on='datetime', direction='backward')['index_bias'].to_numpy()
        else:
            b = np.zeros(len(f))
        long_ = base & (f['low'] <= f['vwap'] * 1.0015) & (f['close'] > f['vwap']) & (f['clv'] >= 0.3) \
            & (f['close'] > f['or_high']) & ~(b < 0)
        short = base & (f['high'] >= f['vwap'] * 0.9985) & (f['close'] < f['vwap']) & (f['clv'] <= -0.3) \
            & (f['close'] < f['or_low']) & ~(b > 0)
        long_, short = _first_per_session(long_, f), _first_per_session(short, f)
        c = f['close']
        ls = c - np.clip(c - np.minimum(f['low'], f['vwap']) * 0.999, 0.003 * c, 0.015 * c)
        ss = c + np.clip(np.maximum(f['high'], f['vwap']) * 1.001 - c, 0.003 * c, 0.015 * c)
        ent = [(i, 'LONG') for i in f.index[long_]] + [(i, 'SHORT') for i in f.index[short]]
        stops = pd.Series(np.where(long_, ls, ss), index=f.index)
        variants['I3 VWAP pullback in trend, 10:30-14:00'] += _simulate_intraday(f, ent, stops)

    ordered = sorted(sessions)
    cut = ordered[int(len(ordered) * IS_FRACTION)] if ordered else None
    md = ['## Intraday — MIS (₹50k notional, MIS costs) — LOW CONFIDENCE: ~59 days of data', '',
          f"{len(ordered)} sessions; in-sample before {cut}, out-of-sample from it.", '',
          '| variant | IS n | IS win | IS avg R | IS net/trade | IS t | IS Kelly | '
          'OOS n | OOS win | OOS avg R | OOS net/trade | OOS t | OOS Kelly | accepted |',
          '|---|' + '---|' * 13]
    rows_out = []
    for name, rows in variants.items():
        t = pd.DataFrame(rows)
        if len(t):
            a, b2 = trade_stats(t[t['session'] < cut]), trade_stats(t[t['session'] >= cut])
            s_all = trade_stats(t)
        else:
            a = b2 = s_all = {'n': 0}
        ok = b2.get('n', 0) >= 30 and b2.get('net', -1) > 0
        rows_out.append((name, a, b2, ok, s_all))
        md.append(f"| {name} | {_fmt_stats(a)} | {_fmt_stats(b2)} | {'**YES**' if ok else 'no'} |")
    return '\n'.join(md), rows_out


# ═════════════════════════════════════════════════════════════════════════════
def run(out_dir, suites=('swing', 'intraday')):
    os.makedirs(out_dir, exist_ok=True)
    prov = YahooProvider()
    md = ['# Phase 2b — pre-registered experiments with out-of-sample split', '',
          'Only the OOS columns are evidence. Every variant is shown, including the losers.', '']
    if 'swing' in suites:
        universe = prov.daily(SWING_UNIVERSE, lookback_days=520)
        index_df = prov.daily([INDEX_SYMBOL], lookback_days=520).get(INDEX_SYMBOL)
        text, _, _ = swing_experiments(universe, index_df)
        md += [text, '']
    if 'intraday' in suites:
        bars = prov.intraday(INTRADAY_UNIVERSE, 5, lookback_days=59)
        daily = prov.daily(INTRADAY_UNIVERSE, lookback_days=120)
        index_bars = prov.intraday([INDEX_SYMBOL], 5, lookback_days=59).get(INDEX_SYMBOL)
        text, _ = intraday_experiments(bars, daily, index_bars)
        md += [text, '']
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'EXPERIMENTS.md'), 'w') as fh:
        fh.write(report + '\n')
    step = os.environ.get('GITHUB_STEP_SUMMARY')
    if step:
        with open(step, 'a') as fh:
            fh.write(report + '\n')
    print(report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description='Phase 2b pre-registered experiments')
    ap.add_argument('--out', default='research_results')
    ap.add_argument('--suite', choices=['swing', 'intraday', 'all'], default='all')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stdout)
    run(args.out, ('swing', 'intraday') if args.suite == 'all' else (args.suite,))


if __name__ == '__main__':
    main()
