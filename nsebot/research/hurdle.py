"""The hurdle: beat the best smallcap mutual fund — measured, not assumed.

    python -m nsebot.research.hurdle --out research_results

1. Every smallcap equity fund (Direct, Growth) from mfapi.in's free NAV
   history, measured over the SAME window as the full-engine backtest: best
   fund, top quartile, median, and the passive smallcap index funds.
2. Higher-return portfolio designs run on that same window and universe,
   ₹50k book, integer shares, Zerodha CNC costs, next-open fills:

     M1  12-1 momentum, top 10, weekly, held only while Nifty > its 200-day SMA
     M2  same, top 5
     M3  6-month momentum, top 10, same market filter
     M4  12-1 momentum, top 10, NO filter (reference)
     B1  70% M1 + 30% the deployed dip-reversion engine (C1)
     C1  the deployed dip-reversion engine on its own

Honesty notes, printed with the results:
  * "The best fund" is picked with hindsight from ~30 funds. Beating the
    ex-post maximum is a far higher bar than beating a fund chosen in advance.
  * The 200-day market filter is a textbook rule (Faber 2007; Antonacci's
    absolute momentum), but choosing to test it here was informed by having
    seen momentum rotation crash in late 2024 (docs/RESEARCH.md round 2). Its
    result is therefore not out-of-sample evidence; forward trading is.
  * Fund NAVs are after fund expenses; the bot is after Zerodha costs but
    before tax. Fund investors pay LTCG; an active CNC book mostly pays STCG.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from .experiments import Panel, simulate_rotation

logger = logging.getLogger(__name__)

MFAPI = 'https://api.mfapi.in'
EXCLUDE = ('idcw', 'dividend', 'bonus', 'payout', 'reinvest', 'regular', 'fof', 'fund of fund')


def _get_json(url, params=None, attempts=3):
    import time

    import requests
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:                      # network: retry, then give up loudly
            if i == attempts - 1:
                logger.warning(f'  {url}: {e}')
                return None
            time.sleep(2 * (i + 1))


# mfapi.in's search returns a capped list, so one broad query is dominated by
# the newer index-fund schemes. Query per fund house as well.
FUND_HOUSES = ['Nippon India', 'Quant', 'SBI', 'Axis', 'HDFC', 'Kotak', 'DSP', 'Tata',
               'Canara Robeco', 'Bandhan', 'Invesco India', 'Franklin India', 'ICICI Prudential',
               'Edelweiss', 'Union', 'Sundaram', 'Aditya Birla Sun Life', 'Mahindra Manulife', 'HSBC',
               'PGIM India', 'Motilal Oswal', 'Bank of India', 'LIC MF', 'Baroda BNP Paribas', 'ITI',
               'Quantum', 'WhiteOak Capital', 'Groww', 'Helios', 'UTI', 'Mirae Asset', 'Bajaj Finserv',
               'JM Financial', 'Shriram', 'Samco', 'Navi', 'Zerodha', 'Trust', 'Old Bridge', 'NJ']
SMALLCAP_WORDS = ('small cap', 'smallcap', 'small-cap', 'smaller companies')


def smallcap_funds(fetch=_get_json, stats=None):
    """{scheme name: NAV series} for Direct-Growth smallcap funds."""
    stats = stats if stats is not None else {}
    queries = ['small cap', 'smallcap', 'smaller companies'] + \
        [f'{h} small cap' for h in FUND_HOUSES] + [f'{h} smallcap' for h in FUND_HOUSES]
    seen, funds = set(), {}
    for q in queries:
        hits = fetch(f'{MFAPI}/mf/search', {'q': q}) or []
        stats['search_hits'] = stats.get('search_hits', 0) + len(hits)
        for row in hits:
            name, code = str(row.get('schemeName', '')), row.get('schemeCode')
            low = name.lower()
            if code in seen or not any(w in low for w in SMALLCAP_WORDS):
                continue
            seen.add(code)
            stats['smallcap_named'] = stats.get('smallcap_named', 0) + 1
            if 'direct' not in low or 'growth' not in low or any(x in low for x in EXCLUDE):
                continue
            stats['direct_growth'] = stats.get('direct_growth', 0) + 1
            data = fetch(f'{MFAPI}/mf/{code}') or {}
            rows = data.get('data') or []
            if len(rows) < 100:
                stats['short_history'] = stats.get('short_history', 0) + 1
                continue
            s = pd.Series({pd.to_datetime(r['date'], format='%d-%m-%Y'): float(r['nav'])
                           for r in rows if r.get('nav') not in (None, '', 'N.A.')}).sort_index()
            funds[name] = s
    return funds


def curve(series, start, end):
    s = series[(series.index >= start) & (series.index <= end)].dropna()
    if len(s) < 20:
        return None
    years = (s.index[-1] - s.index[0]).days / 365.25
    return {'ret': s.iloc[-1] / s.iloc[0] - 1,
            'cagr': (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0.2 else np.nan,
            'maxdd': float((s / s.cummax() - 1).min()), 'start': s.index[0], 'end': s.index[-1]}


def bot_variants(universe, index_df, window_start):
    from ..backtest import run_backtest

    P = Panel(universe)
    c = P.df['close']
    mom12_1 = (c.shift(21) / c.shift(252) - 1).to_numpy()
    mom6 = (c / c.shift(126) - 1).to_numpy()
    nifty = index_df.set_index(pd.to_datetime(index_df['datetime']))['close'].astype(float)
    nifty = nifty.reindex(P.dates).ffill()
    risk_on = (nifty > nifty.rolling(200, min_periods=200).mean()).to_numpy()

    curves = {}
    for name, score, n, flt in (('M1 12-1 momentum, top 10, Nifty > 200-day SMA filter', mom12_1, 10, True),
                                ('M2 12-1 momentum, top 5, same filter', mom12_1, 5, True),
                                ('M3 6-month momentum, top 10, same filter', mom6, 10, True),
                                ('M4 12-1 momentum, top 10, no filter (reference)', mom12_1, 10, False)):
        eq, _ = simulate_rotation(P, score, P.liquid, n=n, risk_on=risk_on if flt else None)
        curves[name] = eq

    latches = {}
    for label, uni, idx in (('C1 deployed engine, started on the full history', universe, index_df),
                            ('C1 deployed engine, started 520 days back (as in its own backtest)',
                             {s: d.iloc[-520:] for s, d in universe.items()}, index_df.iloc[-520:])):
        _, eq, _, led = run_backtest(uni, idx)
        curves[label] = eq
        latch = os.path.join(os.path.dirname(led.dir), 'BREAKER_TRIPPED_swing')
        latches[label] = open(latch).read().strip() if os.path.exists(latch) else None
    c1 = curves['C1 deployed engine, started on the full history']
    m1 = curves['M1 12-1 momentum, top 10, Nifty > 200-day SMA filter']
    common = c1.index.intersection(m1.dropna().index)
    common = common[common >= window_start]
    if len(common) > 20:
        a = m1.reindex(common) / m1.reindex(common).iloc[0]
        b = c1.reindex(common) / c1.reindex(common).iloc[0]
        curves['B1 70% M1 + 30% C1'] = 50_000 * (0.7 * a + 0.3 * b)
    return curves, P, latches


def run(out_dir):
    from ..data import YahooProvider
    from ..universe import INDEX_SYMBOL, SWING_UNIVERSE

    os.makedirs(out_dir, exist_ok=True)
    prov = YahooProvider()
    universe = prov.daily(SWING_UNIVERSE, lookback_days=900)
    index_df = prov.daily([INDEX_SYMBOL], lookback_days=900)[INDEX_SYMBOL]

    # Same window as the full-engine backtest reported to date.
    window_start = pd.Timestamp('2024-06-11')
    curves, P, latches = bot_variants(universe, index_df, window_start)
    window_end = P.dates[-1]

    fund_stats = {}
    funds = smallcap_funds(stats=fund_stats)
    rows = []
    for name, nav in funds.items():
        st = curve(nav, window_start, window_end)
        if st and st['start'] <= window_start + pd.Timedelta(days=10):
            rows.append((name, st))
    rows.sort(key=lambda x: -x[1]['cagr'])
    active = [(n, s) for n, s in rows if 'index' not in n.lower() and 'etf' not in n.lower()]
    passive = [(n, s) for n, s in rows if (n, s) not in active]

    md = ['# The hurdle: best smallcap fund vs the bot, same window', '',
          f'Window **{window_start.date()} → {window_end.date()}**. ₹50k bot book, Zerodha CNC costs, '
          f'next-open fills. Fund NAVs are after expense ratio (Direct, Growth).', '']
    if active:
        cagrs = [s['cagr'] for _, s in active]
        md += [f'**{len(active)} active smallcap funds** with full history in the window: best '
               f'**{max(cagrs) * 100:+.1f}% CAGR**, top quartile {np.percentile(cagrs, 75) * 100:+.1f}%, '
               f'median {np.median(cagrs) * 100:+.1f}%, worst {min(cagrs) * 100:+.1f}%.', '',
               '| smallcap fund (top 5 + median) | total | CAGR | max DD |', '|---|---|---|---|']
        med_i = len(active) // 2
        for i, (n, s) in enumerate(active):
            if i < 5 or i == med_i:
                tag = ' (median)' if i == med_i else ''
                md.append(f"| {n[:70]}{tag} | {s['ret'] * 100:+.1f}% | {s['cagr'] * 100:+.1f}% | "
                          f"{s['maxdd'] * 100:.1f}% |")
    else:
        md += ['_No ACTIVE smallcap fund with full-window history was retrieved this run._']
    md += ['', f'Fund retrieval: {fund_stats}', '']
    if passive:
        md += ['', '| passive smallcap index fund | total | CAGR | max DD |', '|---|---|---|---|']
        for n, s in passive[:3]:
            md.append(f"| {n[:70]} | {s['ret'] * 100:+.1f}% | {s['cagr'] * 100:+.1f}% | {s['maxdd'] * 100:.1f}% |")

    best = active[0][1]['cagr'] if active else np.nan
    median = float(np.median([s['cagr'] for _, s in active])) if active else np.nan
    md += ['', '## Bot designs on the same window', '',
           '| design | total | CAGR | max DD | 1st half | 2nd half | beats median fund | beats best fund |',
           '|---|---|---|---|---|---|---|---|']
    for name, eq in curves.items():
        st = curve(eq.dropna(), window_start, window_end)
        if not st:
            md.append(f'| {name} | — | — | — | — | — | — | — |')
            continue
        e = eq.dropna()
        e = e[(e.index >= window_start) & (e.index <= window_end)]
        mid = e.index[len(e) // 2]
        h1, h2 = curve(e, e.index[0], mid), curve(e, mid, e.index[-1])
        md.append(f"| {name} | {st['ret'] * 100:+.1f}% | {st['cagr'] * 100:+.1f}% | {st['maxdd'] * 100:.1f}% | "
                  f"{h1['ret'] * 100:+.1f}% | {h2['ret'] * 100:+.1f}% | "
                  f"{'yes' if st['cagr'] > median else 'no'} | {'**YES**' if st['cagr'] > best else 'no'} |")
    for label, txt in latches.items():
        md.append(f"- {label}: drawdown latch {'**TRIPPED** — ' + txt.splitlines()[0] if txt else 'not tripped'}")
    md += ['', '_The best fund is chosen with hindsight; the market-filter variants were designed after '
               'seeing momentum crash in late 2024, so their numbers are not out-of-sample evidence._']
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'HURDLE.md'), 'w') as fh:
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
