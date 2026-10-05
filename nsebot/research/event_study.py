"""Does the trigger predict anything? Measured on real NSE data, net of costs.

    python -m nsebot.research.event_study --mode swing
    python -m nsebot.research.event_study --mode intraday
    python -m nsebot.research.event_study --mode all --out research_results

Swing: every historical trigger from the live SwingSignalEngine (the exact
code the bot trades), scored against the universe on its own date. Measured
two ways:
  1. Event study — return from the NEXT open to close at +5/+10/+20 sessions,
     against the average liquid stock over the same window (excess return).
  2. Baseline trade — enter next open, initial stop as signalled, exit at the
     stop or after 15 sessions, CNC costs on a ₹15k position. No trailing,
     no targets: this isolates the entry. Phase 3 adds the exits.

Intraday: every first-of-session trigger from IntradaySignalEngine over the
~59 days of 5-minute history Yahoo serves. Enter next bar's open, exit at the
signal stop or the 15:10 close, MIS costs.

Every entry is at a price that existed AFTER the signal bar closed. Nothing
here reads a bar the live bot could not have seen.
"""

import argparse
import logging
import os
import sys
from datetime import time

import numpy as np
import pandas as pd

from ..costs import DEFAULT_CHARGES
from ..data import YahooProvider
from ..regime import RegimeDial
from ..signals import IntradaySignalEngine, SwingSignalEngine
from ..universe import INDEX_SYMBOL, INTRADAY_UNIVERSE, SWING_UNIVERSE

logger = logging.getLogger(__name__)

SWING_NOTIONAL = 15_000.0
INTRADAY_NOTIONAL = 50_000.0
SWING_HOLD = 15
HORIZONS = (5, 10, 20)
SQUARE_OFF = time(15, 5)        # the 15:05 bar closes at 15:10


# ═════════════════════════════════════════════════════════════════════════════
# SWING
# ═════════════════════════════════════════════════════════════════════════════
def swing_study(universe, index_df, cfg=None):
    eng = SwingSignalEngine(cfg)
    events, feats = eng.panel(universe)
    if events.empty:
        return pd.DataFrame(), {}

    opens = pd.DataFrame({s: f.set_index('datetime')['open'] for s, f in feats.items()})
    closes = pd.DataFrame({s: f.set_index('datetime')['close'] for s, f in feats.items()})
    lows = pd.DataFrame({s: f.set_index('datetime')['low'] for s, f in feats.items()})
    liquid = pd.DataFrame({s: f.set_index('datetime')['liquid'] for s, f in feats.items()}) \
        .fillna(False).astype(bool)
    dates = closes.index
    pos = {d: i for i, d in enumerate(dates)}

    # Universe baseline: average liquid stock, same entry rule, same horizon.
    base = {}
    for h in HORIZONS:
        fwd = closes.shift(-h) / opens.shift(-1) - 1.0
        base[h] = fwd.where(liquid).mean(axis=1)

    regime = RegimeDial().panel(index_df, universe)['score'].reindex(dates).ffill()

    rows = []
    for _, e in events.iterrows():
        sym, t = e['symbol'], pos.get(e['datetime'])
        if t is None or t + 1 >= len(dates):
            continue
        entry = opens[sym].iloc[t + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        r = {'symbol': sym, 'date': e['datetime'], 'trigger': e['trigger'],
             'score': e['score'], 'regime': regime.iloc[t], 'entry': entry,
             'slip_pct': entry / e['close'] - 1.0}
        for h in HORIZONS:
            if t + h < len(dates) and np.isfinite(closes[sym].iloc[t + h]):
                r[f'fwd_{h}'] = closes[sym].iloc[t + h] / entry - 1.0
                r[f'xs_{h}'] = r[f'fwd_{h}'] - base[h].iloc[t]
        # Baseline trade: stop or time, nothing else.
        stop = float(e['stop'])
        if entry <= stop:
            r['trade_r'] = np.nan                       # gapped through the stop pre-entry
        else:
            exit_px, held = None, 0
            for k in range(t + 1, min(t + 1 + SWING_HOLD, len(dates))):
                held += 1
                o, lo, c = opens[sym].iloc[k], lows[sym].iloc[k], closes[sym].iloc[k]
                if not np.isfinite(c):
                    continue
                if k > t + 1 and o <= stop:
                    exit_px = o                         # gap through the stop: fill at the open
                    break
                if lo <= stop:
                    exit_px = stop
                    break
                exit_px = c
            if exit_px is not None and held >= 1:
                qty = max(int(SWING_NOTIONAL // entry), 1)
                gross = (exit_px - entry) * qty
                cost = DEFAULT_CHARGES.cnc(entry * qty, exit_px * qty)
                r['trade_r'] = (exit_px - entry) / (entry - stop)
                r['trade_net_pct'] = (gross - cost) / (entry * qty)
                r['trade_cost_pct'] = cost / (entry * qty)
                r['held'] = held
        rows.append(r)

    ev = pd.DataFrame(rows)
    days = len(dates[dates >= ev['date'].min()]) if len(ev) else 0
    summary = {'events': len(ev), 'sessions': days,
               'events_per_session': round(len(ev) / max(days, 1), 2)}
    return ev, summary


def _swing_table(ev):
    lines = ['| slice | n | ' + ' | '.join(f'fwd {h}d' for h in HORIZONS) + ' | '
             + ' | '.join(f'excess {h}d' for h in HORIZONS) + ' | hit 10d | trade win | avg R | net/trade |',
             '|---|---|' + '---|' * (2 * len(HORIZONS) + 4)]

    def row(name, g):
        if len(g) == 0:
            return
        cells = [name, str(len(g))]
        cells += [f"{g[f'fwd_{h}'].mean() * 100:+.2f}%" for h in HORIZONS]
        cells += [f"{g[f'xs_{h}'].mean() * 100:+.2f}%" for h in HORIZONS]
        cells.append(f"{(g['fwd_10'] > 0).mean() * 100:.0f}%")
        t = g.dropna(subset=['trade_r'])
        cells.append(f"{(t['trade_r'] > 0).mean() * 100:.0f}%" if len(t) else '—')
        cells.append(f"{t['trade_r'].mean():+.2f}" if len(t) else '—')
        cells.append(f"{t['trade_net_pct'].mean() * 100:+.2f}%" if len(t) else '—')
        lines.append('| ' + ' | '.join(cells) + ' |')

    row('ALL', ev)
    for trig, g in ev.groupby('trigger'):
        row(f'trigger={trig}', g)
    for name, g in ev.groupby(pd.cut(ev['regime'], [-0.01, 0.33, 0.66, 1.01],
                                     labels=['weak', 'mixed', 'strong']), observed=True):
        row(f'regime={name}', g)
    top = ev[ev['score'] >= ev['score'].quantile(0.5)]
    row('score top half', top)
    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# INTRADAY
# ═════════════════════════════════════════════════════════════════════════════
def intraday_study(bars_by_symbol, daily_by_symbol, index_bars, cfg=None):
    eng = IntradaySignalEngine(cfg)
    rows = []
    sessions_seen = set()
    for sym, bars in bars_by_symbol.items():
        if bars is None or len(bars) < 150:
            continue
        f = eng.session_features(bars, daily_by_symbol.get(sym), index_bars)
        sessions_seen.update(f['session'].unique())
        trig = f[f['first_long'] | f['first_short']]
        for session, g in trig.groupby('session'):
            e = g.iloc[0]
            side = 'LONG' if e['first_long'] else 'SHORT'
            day = f[f['session'] == session]
            after = day[day['datetime'] > e['datetime']]
            if after.empty:
                continue
            entry = float(after['open'].iloc[0])
            stop = float(e['long_stop'] if side == 'LONG' else e['short_stop'])
            sgn = 1.0 if side == 'LONG' else -1.0
            if sgn * (entry - stop) <= 0:
                continue                                 # opened beyond the stop
            exit_px = None
            for _, b in after.iterrows():
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
            gross = sgn * (exit_px - entry) * qty
            cost = DEFAULT_CHARGES.mis(entry * qty, exit_px * qty)
            rows.append({'symbol': sym, 'session': session, 'side': side,
                         'time': e['datetime'].strftime('%H:%M'), 'entry': entry,
                         'stop': stop, 'exit': exit_px,
                         'r': sgn * (exit_px - entry) / abs(entry - stop),
                         'net_pct': (gross - cost) / (entry * qty),
                         'cost_pct': cost / (entry * qty),
                         'or_rvol': e['or_rvol'], 'score': e['score']})
    ev = pd.DataFrame(rows)
    summary = {'trades': len(ev), 'sessions': len(sessions_seen),
               'trades_per_session': round(len(ev) / max(len(sessions_seen), 1), 2)}
    return ev, summary


def _intraday_table(ev):
    lines = ['| slice | n | win | avg R | payoff | expectancy (R) | avg net/trade | cost/trade |',
             '|---|---|---|---|---|---|---|---|']

    def row(name, g):
        if len(g) == 0:
            return
        w, l = g[g['r'] > 0]['r'], g[g['r'] <= 0]['r']
        payoff = (w.mean() / abs(l.mean())) if len(w) and len(l) and l.mean() != 0 else float('nan')
        lines.append(f"| {name} | {len(g)} | {(g['r'] > 0).mean() * 100:.0f}% | "
                     f"{g['r'].mean():+.2f} | {payoff:.2f} | {g['r'].mean():+.3f} | "
                     f"{g['net_pct'].mean() * 100:+.3f}% | {g['cost_pct'].mean() * 100:.3f}% |")

    row('ALL', ev)
    for side, g in ev.groupby('side'):
        row(f'side={side}', g)
    row('score top half', ev[ev['score'] >= ev['score'].quantile(0.5)])
    early = ev['time'] < '11:00'
    row('entry before 11:00', ev[early])
    row('entry 11:00+', ev[~early])
    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
def _kelly_line(r):
    r = r.dropna()
    if len(r) < 20:
        return 'insufficient sample for a Kelly estimate'
    p = (r > 0).mean()
    w, l = r[r > 0].mean(), abs(r[r <= 0].mean())
    b = w / l if l else float('nan')
    f = p - (1 - p) / b if b and np.isfinite(b) else float('nan')
    return f'win {p:.1%}, payoff {b:.2f}, full-Kelly fraction of R {f:+.3f} (n={len(r)})'


def run(mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prov = YahooProvider()
    md = ['# nsebot signal research — real NSE data', '']

    if mode in ('swing', 'all'):
        logger.info('Swing: downloading daily bars...')
        universe = prov.daily(SWING_UNIVERSE, lookback_days=520)
        index_df = prov.daily([INDEX_SYMBOL], lookback_days=520).get(INDEX_SYMBOL)
        ev, s = swing_study(universe, index_df)
        md += ['## Swing (CNC) — momentum continuation', '',
               f"Universe resolved: {len(universe)}/{len(SWING_UNIVERSE)} symbols. "
               f"{s.get('events', 0)} triggers over {s.get('sessions', 0)} sessions "
               f"= **{s.get('events_per_session', 0)} candidates per session** "
               f"(V2: 0 per session since 14 Sep).", '']
        if len(ev):
            md += [_swing_table(ev), '',
                   f"Baseline trade (stop or 15 sessions, CNC costs on ₹{SWING_NOTIONAL:,.0f}): "
                   f"{_kelly_line(ev.get('trade_r', pd.Series(dtype=float)))}", '',
                   f"Median next-open slippage vs signal close: {ev['slip_pct'].median() * 100:+.2f}%", '']
            ev.to_csv(os.path.join(out_dir, 'swing_events.csv'), index=False)

    if mode in ('intraday', 'all'):
        logger.info('Intraday: downloading 5-minute bars...')
        bars = prov.intraday(INTRADAY_UNIVERSE, 5, lookback_days=59)
        daily = prov.daily(INTRADAY_UNIVERSE, lookback_days=120)
        index_bars = prov.intraday([INDEX_SYMBOL], 5, lookback_days=59).get(INDEX_SYMBOL)
        ev, s = intraday_study(bars, daily, index_bars)
        md += ['## Intraday (MIS) — opening-range breakout + VWAP + volume', '',
               f"Symbols with 5m data: {len(bars)}/{len(INTRADAY_UNIVERSE)}. "
               f"{s['trades']} trades over {s['sessions']} sessions = "
               f"**{s['trades_per_session']} per session**.", '']
        if len(ev):
            md += [_intraday_table(ev), '',
                   f"Baseline (signal stop or 15:10 close, MIS costs on ₹{INTRADAY_NOTIONAL:,.0f}): "
                   f"{_kelly_line(ev['r'])}", '']
            ev.to_csv(os.path.join(out_dir, 'intraday_trades.csv'), index=False)

    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'SUMMARY.md'), 'w') as fh:
        fh.write(report + '\n')
    step = os.environ.get('GITHUB_STEP_SUMMARY')
    if step:
        with open(step, 'a') as fh:
            fh.write(report + '\n')
    print(report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mode', choices=['swing', 'intraday', 'all'], default='all')
    ap.add_argument('--out', default='research_results')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stdout)
    run(args.mode, args.out)


if __name__ == '__main__':
    main()
