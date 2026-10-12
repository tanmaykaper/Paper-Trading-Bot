"""Phase 5 — validation on UNTOUCHED history: Jul 2018 to Jul 2023.

    python -m nsebot.research.phase5 --out research_results

Every design in research rounds 1-4 was selected on data from 14 Aug 2023
onward. This window ends before that date, so nothing below was chosen on
it. It spans a smallcap bear market (2018-19), the COVID crash and
recovery (2020), the 2020-21 boom and the 2022 chop.

PRE-REGISTERED — written and committed before any result was seen.

Frozen designs (no parameter changes from rounds 2-4):
  D1  the deployed engine (Phase 4b: S4b dip reversion, 20% notional, ₹8k
      floor, 3% risk cap, 5 slots)
  D2  12-1 momentum, top 5, weekly, held only while Nifty > its 200-day SMA
  D3  same, top 10
  D4  12-1 momentum, top 10, no market filter

Return boosters, specified now (all untested until this run):
  E1  concentrated dip: D1 with 33% notional per position, 3 slots
  E2  broader dip: D1 triggered on a 5% (not 8%) 3-day drop — more trades
  E3  combined book: 60% D2 + 40% D1, separate sleeves
  E4  leveraged momentum: D2 at 1.5x exposure, the borrowed 0.5x charged
      15%/yr on every day D2 holds stocks (approximates Zerodha MTF; MTF
      eligibility varies by stock and leverage deepens drawdowns 1.5x)

Benchmarks over the same window: every active Direct-Growth smallcap fund
with full NAV history (mfapi.in), Nifty 50, and the Smallcap 100 index if
Yahoo serves it.

Acceptance rules:
  QUALIFIES     window CAGR > median active smallcap fund AND beats the
                median fund in at least 3 of the 5 July-June years
  MEETS TARGET  window CAGR > the best active smallcap fund

The engine designs (D1, E1, E2) carry the deployed 25% drawdown latch; if it
trips, the design stops opening trades until a human resets it. The report
says when that happened — a tripped latch is a result, not a bug.

Known bias, stated up front: the universe is TODAY's 255-name list applied
to 2018 data. Names that later grew into it are included (survivorship /
look-ahead), which flatters every bot design. A bot result that loses to
funds here is strong evidence; a bot result that wins is weak evidence.
"""

import argparse
import dataclasses
import logging
import os
import sys
import time as _time

import numpy as np
import pandas as pd

from ..config import BotConfig
from ..data import YahooProvider, sanitize, to_yahoo
from .experiments import Panel, simulate_rotation
from .hurdle import curve, smallcap_funds

logger = logging.getLogger(__name__)

FETCH_START, FETCH_END = '2016-07-01', '2023-08-12'      # strictly before the selection data
WIN_START, WIN_END = pd.Timestamp('2018-07-02'), pd.Timestamp('2023-07-31')
YEARS = [(pd.Timestamp(f'{y}-07-01'), pd.Timestamp(f'{y + 1}-06-30')) for y in range(2018, 2022)] + \
        [(pd.Timestamp('2022-07-01'), WIN_END)]
SMALLCAP_INDEX = '^CNXSC'                                  # NIFTY SMALLCAP 100 on Yahoo, if served


def fetch_range(symbols, start=FETCH_START, end=FETCH_END, chunk=40):
    import yfinance as yf

    prov, out = YahooProvider(), {}
    symbols = list(dict.fromkeys(symbols))
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        tickers = [to_yahoo(s) for s in part]
        raw = prov._with_retry(
            lambda: yf.download(tickers=tickers, start=start, end=end, interval='1d',
                                group_by='ticker', auto_adjust=True, threads=True, progress=False),
            what=f'range chunk {i // chunk + 1}')
        for sym, frame in YahooProvider._split(raw, tickers).items():
            clean = sanitize(frame, is_index=sym.startswith('^'))
            if clean is not None:
                out[sym] = clean
        _time.sleep(1.0)
    return out


def designs(universe, index_df):
    from ..backtest import run_backtest

    base = BotConfig()
    P = Panel(universe)
    c = P.df['close']
    mom12_1 = (c.shift(21) / c.shift(252) - 1).to_numpy()
    nifty = index_df.set_index(pd.to_datetime(index_df['datetime']))['close'].astype(float)
    nifty = nifty.reindex(P.dates).ffill()
    risk_on = (nifty > nifty.rolling(200, min_periods=200).mean()).to_numpy()

    out, latches = {}, {}

    def engine(label, cfg):
        _, eq, _, led = run_backtest(universe, index_df, cfg)
        latch = os.path.join(os.path.dirname(led.dir), 'BREAKER_TRIPPED_swing')
        latches[label] = open(latch).read().strip() if os.path.exists(latch) else None
        out[label] = eq

    engine('D1 deployed engine (dip reversion, Phase 4b)', base)
    for name, n, flt in (('D2 12-1 momentum top 5 + 200-day filter', 5, True),
                         ('D3 12-1 momentum top 10 + 200-day filter', 10, True),
                         ('D4 12-1 momentum top 10, no filter', 10, False)):
        out[name] = simulate_rotation(P, mom12_1, P.liquid, n=n, risk_on=risk_on if flt else None)[0]

    e1 = dataclasses.replace(base, swing_sizing=dataclasses.replace(
        base.swing_sizing, target_notional_pct=0.33, max_positions=3, max_portfolio_heat_pct=0.09))
    engine('E1 concentrated dip: 33% x 3 slots', e1)
    e2 = dataclasses.replace(base, reversion=dataclasses.replace(base.reversion, drop_pct=0.05))
    engine('E2 broader dip: 5% trigger', e2)

    d1, d2 = out['D1 deployed engine (dip reversion, Phase 4b)'], out['D2 12-1 momentum top 5 + 200-day filter']
    idx = d1.dropna().index.intersection(d2.dropna().index)
    idx = idx[idx >= WIN_START]
    if len(idx) > 20:
        out['E3 combined: 60% D2 + 40% D1'] = 50_000 * (0.6 * d2.reindex(idx) / d2.reindex(idx).iloc[0]
                                                        + 0.4 * d1.reindex(idx) / d1.reindex(idx).iloc[0])
    r = d2.dropna().pct_change().fillna(0.0)
    invested = (r != 0).astype(float)          # all-cash days have exactly zero return: nothing borrowed
    out['E4 leveraged momentum: D2 at 1.5x, 15%/yr financing'] = \
        50_000 * (1 + 1.5 * r - invested * 0.5 * 0.15 / 252).cumprod()
    return out, P, latches


def yearly(series):
    vals = []
    for a, b in YEARS:
        st = curve(series.dropna(), a, b)
        vals.append(st['ret'] if st else np.nan)
    return vals


def run(out_dir):
    from ..universe import INDEX_SYMBOL, SWING_UNIVERSE

    os.makedirs(out_dir, exist_ok=True)
    universe = fetch_range(SWING_UNIVERSE)
    extra = fetch_range([INDEX_SYMBOL, SMALLCAP_INDEX])
    index_df = extra.get(INDEX_SYMBOL)
    alive_2018 = sum(1 for d in universe.values()
                     if len(d) and pd.to_datetime(d['datetime']).iloc[0] <= WIN_START)

    curves, P, latches = designs(universe, index_df)

    funds = smallcap_funds()
    active = []
    for name, nav in funds.items():
        low = name.lower()
        if 'index' in low or 'etf' in low:
            continue
        st = curve(nav, WIN_START, WIN_END)
        if st and st['start'] <= WIN_START + pd.Timedelta(days=10) and st['end'] >= WIN_END - pd.Timedelta(days=10):
            active.append((name, st, nav))
    active.sort(key=lambda x: -x[1]['cagr'])

    md = ['# Phase 5 — untouched history: Jul 2018 to Jul 2023', '',
          f'Window **{WIN_START.date()} → {WIN_END.date()}**, never used to select anything. '
          f'Universe: {len(universe)} of {len(SWING_UNIVERSE)} symbols resolved, {alive_2018} trading by Jul 2018. '
          f'₹50k book, Zerodha CNC costs, next-open fills.', '',
          '**Bias warning:** today\'s symbol list applied to 2018 data includes names that later grew into it, '
          'which flatters every bot design.', '']

    med_name, med_nav, best_cagr, med_cagr = None, None, np.nan, np.nan
    if active:
        cagrs = [s['cagr'] for _, s, _ in active]
        best_cagr, med_cagr = max(cagrs), float(np.median(cagrs))
        mi = len(active) // 2
        med_name, med_nav = active[mi][0], active[mi][2]
        md += [f'**{len(active)} active smallcap funds** with full-window history: best '
               f'**{best_cagr * 100:+.1f}% CAGR**, top quartile {np.percentile(cagrs, 75) * 100:+.1f}%, '
               f'median {med_cagr * 100:+.1f}%, worst {min(cagrs) * 100:+.1f}%.', '',
               '| fund | total | CAGR | max DD |', '|---|---|---|---|']
        for i, (n, s, _) in enumerate(active):
            if i < 5 or i == mi:
                md.append(f"| {n[:68]}{' (median)' if i == mi else ''} | {s['ret'] * 100:+.1f}% | "
                          f"{s['cagr'] * 100:+.1f}% | {s['maxdd'] * 100:.1f}% |")
    else:
        md += ['_No active smallcap fund NAV history retrieved — acceptance cannot be judged this run._']

    bench = {'Nifty 50 (buy & hold)': index_df.set_index(pd.to_datetime(index_df['datetime']))['close']}
    if extra.get(SMALLCAP_INDEX) is not None:
        sc = extra[SMALLCAP_INDEX]
        bench['Nifty Smallcap 100 index'] = sc.set_index(pd.to_datetime(sc['datetime']))['close']
    md += ['', '| benchmark | total | CAGR | max DD |', '|---|---|---|---|']
    for n, s in bench.items():
        st = curve(s.astype(float), WIN_START, WIN_END)
        if st:
            md.append(f"| {n} | {st['ret'] * 100:+.1f}% | {st['cagr'] * 100:+.1f}% | {st['maxdd'] * 100:.1f}% |")

    med_years = yearly(med_nav) if med_nav is not None else [np.nan] * len(YEARS)
    ylabels = [f"{a.year % 100:02d}-{b.year % 100:02d}" for a, b in YEARS]
    md += ['', '## Designs on the untouched window', '',
           '| design | total | CAGR | max DD | ' + ' | '.join(ylabels) + ' | yrs > median fund | QUALIFIES | MEETS TARGET |',
           '|---|---|---|---|' + '---|' * len(YEARS) + '---|---|---|']
    md.append('| median active fund | — | ' + (f'{med_cagr * 100:+.1f}%' if np.isfinite(med_cagr) else '—')
              + ' | — | ' + ' | '.join(f'{v * 100:+.1f}%' if np.isfinite(v) else '—' for v in med_years)
              + ' | — | — | — |')
    for name, eq in curves.items():
        st = curve(eq.dropna(), WIN_START, WIN_END)
        if not st:
            md.append(f'| {name} | — | — | — |' + ' — |' * len(YEARS) + ' — | — | — |')
            continue
        ys = yearly(eq)
        wins = sum(1 for a, b in zip(ys, med_years) if np.isfinite(a) and np.isfinite(b) and a > b)
        qual = np.isfinite(med_cagr) and st['cagr'] > med_cagr and wins >= 3
        meets = np.isfinite(best_cagr) and st['cagr'] > best_cagr
        md.append(f"| {name} | {st['ret'] * 100:+.1f}% | {st['cagr'] * 100:+.1f}% | {st['maxdd'] * 100:.1f}% | "
                  + ' | '.join(f'{v * 100:+.1f}%' if np.isfinite(v) else '—' for v in ys)
                  + f" | {wins}/{len(YEARS)} | {'**YES**' if qual else 'no'} | {'**YES**' if meets else 'no'} |")
    tripped = {k: v for k, v in latches.items() if v}
    md += ['', '**Drawdown latch (25%):** ' + ('; '.join(f'{k} — tripped: {v}' for k, v in tripped.items())
                                            if tripped else 'never tripped in D1, E1 or E2.')]
    md += ['', '_Rules (pre-registered): QUALIFIES = CAGR above the median active fund and beating it in at '
               'least 3 of 5 years. MEETS TARGET = CAGR above the best active fund. Survivorship in the symbol '
               'list biases bot results upward._']
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'PHASE5.md'), 'w') as fh:
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
