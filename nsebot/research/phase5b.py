"""Phase 5b — momentum without hindsight in the symbol list.

    python -m nsebot.research.phase5b --out research_results

Phase 5 found one design that beat every smallcap fund on the untouched
2018-23 window: D4, 12-1 momentum, top 10, weekly, no market filter (+37.5%
CAGR against +28.7% for the best fund). But D4 picked from TODAY's 260-name
list, and a cost-free equal-weight hold of that list alone made +24.0%,
beating the median fund (+21.7%). This test asks whether D4's edge survives
on a symbol list that carries no hindsight.

PRE-REGISTERED — written and committed before any result was seen.

Universe: every NSE-listed equity in series EQ, BE or BZ (NSE's EQUITY_L.csv;
the narrower Nifty Total Market list is a fallback, flagged in the report). A
stock is eligible on a date only if, on that date, its 20-day median traded
value is at least ₹5 cr and its price at least ₹50 (D4's rule, unchanged),
and it has the 252 bars 12-1 momentum needs. Inclusion no longer depends on
having grown big by 2026. Still missing: companies delisted before today,
which Yahoo does not serve. That residual bias is smaller, but it still
flatters the result.

Design B4 (the one judged): D4 unchanged — 12-1 momentum, top 10, rebalanced
every 5 sessions with a 3n rank band, ₹50k, Zerodha CNC costs, next-open
fills — plus one realism rule fixed now: no buy fill on a bar frozen at its
upper circuit, no sale on a bar frozen at its lower circuit
(experiments.locked_bars).

References, reported and not judged: L4 is the same code on the 260-name list,
so B4 against L4 isolates the universe; cost-free equal-weight holds of each
universe's eligible names; Nifty 50; every active Direct-Growth smallcap fund
with NAV history over the window.

Windows:
  PRIMARY    Jul 2018 – Jul 2023, untouched, as in Phase 5.
  SECONDARY  Jul 2018 – latest bar. Adds 2023-26, where momentum was
             researched (so not clean) and where D4 lost to the funds.
  ROUND 4    Jun 2024 – latest bar, for comparison with round 4.

Decision rule:
  PASS  B4 QUALIFIES on the primary window (CAGR above the median fund AND
        beating it in at least 3 of 5 years) AND B4's secondary-window CAGR is
        above the median fund's over the same span.
  NOT JUDGED  fewer than 70% of the listed symbols return data.
  Only on PASS is a momentum paper sleeve proposed, and even then only with
  the owner's confirmation. On FAIL, Phase 5's D4 result is attributed to the
  symbol list and momentum is not deployed.
  MEETS TARGET (CAGR above the best fund) is reported for both windows.
"""

import argparse
import io
import logging
import os
import sys

import numpy as np
import pandas as pd

from ..indicators import median_turnover
from .experiments import Panel, benchmark_equal_weight, locked_bars, simulate_rotation
from .hurdle import curve, smallcap_funds
from .phase5 import FETCH_START, WIN_END, WIN_START, YEARS, fetch_range

logger = logging.getLogger(__name__)

LIST_SOURCES = [
    ('NSE equity list (EQUITY_L)', 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv'),
    ('NSE equity list (EQUITY_L)', 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'),
    ('Nifty Total Market list — NARROWER fallback',
     'https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv'),
    ('Nifty Total Market list — NARROWER fallback',
     'https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv'),
]
SERIES = ('EQ', 'BE', 'BZ')
MIN_LIST = 300                         # an error page or a truncated file must not pass as the market
MIN_RESOLVED = 0.70
MIN_TURNOVER, MIN_PRICE = 5e7, 50.0    # D4's point-in-time eligibility (Panel.liquid), unchanged
RECENT_START = pd.Timestamp('2024-06-11')
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/124.0 Safari/537.36',
           'Accept': 'text/csv,text/plain,*/*', 'Referer': 'https://www.nseindia.com/'}


def _get_text(url, attempts=3):
    import time

    import requests
    for i in range(attempts):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:                      # network: retry, then give up loudly
            if i == attempts - 1:
                logger.warning(f'  {url}: {e}')
                return None
            time.sleep(2 * (i + 1))


def listed_equities(get=_get_text):
    """(symbols, source label, url) from the first source that answers with a
    real list; ([], None, None) when none does."""
    for label, url in LIST_SOURCES:
        text = get(url)
        if not text:
            continue
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            logger.warning(f'  {url}: not a CSV ({e})')
            continue
        df.columns = [str(c).strip().upper() for c in df.columns]
        if 'SYMBOL' not in df.columns:
            continue
        if 'SERIES' in df.columns:
            df = df[df['SERIES'].astype(str).str.strip().isin(SERIES)]
        syms = sorted({str(s).strip() for s in df['SYMBOL'].dropna() if str(s).strip()})
        if len(syms) >= MIN_LIST:
            return syms, label, url
        logger.warning(f'  {url}: only {len(syms)} symbols — not accepted as the market')
    return [], None, None


def ever_eligible(df):
    """Could this stock EVER pass D4's eligibility? Names that never can are
    dropped before the panel is built; that cannot change any trade."""
    t = median_turnover(df, 20)
    return bool(((t >= MIN_TURNOVER) & (df['close'] >= MIN_PRICE)).any())


def momentum_book(universe):
    P = Panel(universe)
    c = P.df['close']
    mom12_1 = (c.shift(21) / c.shift(252) - 1).to_numpy()
    buy_ok, sell_ok = locked_bars(P)
    eq = simulate_rotation(P, mom12_1, P.liquid, n=10, buy_ok=buy_ok, sell_ok=sell_ok)[0]
    return P, eq, benchmark_equal_weight(P)


def july_years(last):
    """July-June years from 2018 to the last bar; the final one may be partial."""
    out = []
    for y in range(2018, last.year + 1):
        a, b = pd.Timestamp(f'{y}-07-01'), min(pd.Timestamp(f'{y + 1}-06-30'), last)
        if a < last - pd.Timedelta(days=30):
            out.append((a, b))
    return out


def by_year(series, years):
    vals = []
    for a, b in years:
        st = curve(series.dropna(), a, b)
        vals.append(st['ret'] if st else np.nan)
    return vals


def active_funds(funds, start, end):
    """Active (non-index) funds with NAVs covering [start, end], best CAGR first."""
    rows = []
    for name, nav in funds.items():
        low = name.lower()
        if 'index' in low or 'etf' in low:
            continue
        st = curve(nav, start, end)
        if st and st['start'] <= start + pd.Timedelta(days=10) and st['end'] >= end - pd.Timedelta(days=10):
            rows.append((name, st, nav))
    rows.sort(key=lambda x: -x[1]['cagr'])
    return rows


def _ylabel(a, b):
    """'18-19' for a July-June year; '*' marks one that is partial or runs past June."""
    if (b - a).days > 400:
        return f'{a:%b %y}–{b:%b %y}'
    return f'{a.year % 100:02d}-{(a.year + 1) % 100:02d}' + ('' if (b.month, b.day) == (6, 30) else '*')


def _pct(v, signed=True):
    return '—' if v is None or not np.isfinite(v) else (f'{v * 100:+.1f}%' if signed else f'{v * 100:.1f}%')


def window_table(title, series, judged, start, end, years, funds, year_rule=True):
    """Markdown rows for one window plus (median CAGR, best CAGR, judged-row stats).
    year_rule=False: the window is judged on CAGR only, so QUALIFIES is not shown."""
    fs = active_funds(funds, start, end)
    labels = [_ylabel(a, b) for a, b in years]
    md = [f'## {title}', '']
    if not fs:
        return md + ['_No active fund NAV history covers this window — cannot be judged._', ''], None, None, {}
    cagrs = [s['cagr'] for _, s, _ in fs]
    med = fs[len(fs) // 2]
    med_cagr, best_cagr = med[1]['cagr'], max(cagrs)
    med_years = by_year(med[2], years)
    md += [f'{len(fs)} active funds: best **{fs[0][0][:60]}** {_pct(best_cagr)} CAGR, median '
           f'**{med[0][:60]}** {_pct(med_cagr)}.', '',
           '| series | total | CAGR | max DD | ' + ' | '.join(labels) + ' | yrs > median fund | QUALIFIES | MEETS TARGET |',
           '|---|---|---|---|' + '---|' * len(years) + '---|---|---|',
           f"| median fund | {_pct(med[1]['ret'])} | {_pct(med_cagr)} | {_pct(med[1]['maxdd'], False)} | "
           + ' | '.join(_pct(v) for v in med_years) + ' | — | — | — |']
    judged_stats = {}
    for name, s in series.items():
        st = curve(s.dropna(), start, end)
        if not st:
            md.append(f'| {name} | — | — | — |' + ' — |' * len(years) + ' — | — | — |')
            continue
        ys = by_year(s, years)
        wins = sum(1 for a, b in zip(ys, med_years) if np.isfinite(a) and np.isfinite(b) and a > b)
        if name in judged:
            qual = st['cagr'] > med_cagr and wins >= 3 if year_rule else None
            meets = st['cagr'] > best_cagr
            judged_stats[name] = {'cagr': st['cagr'], 'wins': wins, 'qualifies': qual, 'meets': meets}
            q = '—' if qual is None else ('**YES**' if qual else 'no')
            m = '**YES**' if meets else 'no'
        else:
            q = m = '—'
        md.append(f"| {name} | {_pct(st['ret'])} | {_pct(st['cagr'])} | {_pct(st['maxdd'], False)} | "
                  + ' | '.join(_pct(v) for v in ys) + f' | {wins}/{len(years)} | {q} | {m} |')
    return md + ['', '_* partial year, or one that runs past June._', ''], med_cagr, best_cagr, judged_stats


def run(out_dir, fetch=None, get_list=None, get_funds=None):
    from ..universe import INDEX_SYMBOL, SWING_UNIVERSE

    fetch = fetch or fetch_range
    os.makedirs(out_dir, exist_ok=True)
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    symbols, label, url = (get_list or listed_equities)()
    md = ['# Phase 5b — momentum without hindsight in the symbol list', '']
    if not symbols:
        md += ['**NOT RUN:** no NSE equity list could be retrieved from any source. No result.']
        return _write(out_dir, md)

    raw = fetch(symbols, FETCH_START, end)
    listed = set(symbols)
    resolved = sum(1 for s in symbols if s in raw)
    broad = {s: d for s, d in raw.items() if s in listed and ever_eligible(d)}
    narrow = {s: raw[s] for s in SWING_UNIVERSE if s in raw}
    narrow.update(fetch([s for s in SWING_UNIVERSE if s not in raw], FETCH_START, end))
    index_df = fetch([INDEX_SYMBOL], FETCH_START, end).get(INDEX_SYMBOL)

    Pb, b4, ew_b = momentum_book(broad)
    Pn, l4, ew_n = momentum_book(narrow)
    last = Pb.dates[-1]
    i0 = int(np.searchsorted(Pb.dates, WIN_START))
    in_win = (Pb.dates >= WIN_START) & (Pb.dates <= WIN_END)
    eligible_now = Pb.liquid.sum(axis=1)
    md += [f'**Universe:** {label} — {len(symbols)} symbols in series {"/".join(SERIES)}; '
           f'{resolved} returned data ({resolved / len(symbols):.0%}); {len(broad)} were ever eligible '
           f'(₹5 cr median traded value, price ≥ ₹50). Eligible on {Pb.dates[min(i0, len(Pb.dates) - 1)].date()}: '
           f'{int(eligible_now[min(i0, len(eligible_now) - 1)])} (260-name list: '
           f'{int(Pn.liquid[min(int(np.searchsorted(Pn.dates, WIN_START)), len(Pn.dates) - 1)].sum())}); '
           f'median eligible per session in the primary window: {int(np.median(eligible_now[in_win])) if in_win.any() else 0}.',
           '', f'Source: `{url}`. Data to {last.date()}. ₹50k book, Zerodha CNC costs, next-open fills, '
           f'no fills on circuit-locked bars.', '',
           '**Still biased upward:** companies delisted before today are absent (Yahoo does not serve them).', '']

    series = {'B4 momentum, broad list (judged)': b4, 'L4 momentum, 260-name list': l4,
              'EW hold, broad list (no costs)': ew_b, 'EW hold, 260-name list (no costs)': ew_n}
    if index_df is not None:
        series['Nifty 50'] = index_df.set_index(pd.to_datetime(index_df['datetime']))['close'].astype(float)
    judged = {'B4 momentum, broad list (judged)'}

    funds = (get_funds or smallcap_funds)()
    t1, med1, best1, j1 = window_table('Primary: untouched window, Jul 2018 – Jul 2023', series, judged,
                                       WIN_START, WIN_END, YEARS, funds)
    full = july_years(last)
    t2, med2, best2, j2 = window_table(f'Secondary: Jul 2018 – {last.date()}', series, judged,
                                       WIN_START, last, full, funds, year_rule=False)
    t3, _, _, _ = window_table(f'Round-4 window: {RECENT_START.date()} – {last.date()}', series,
                               set(), RECENT_START, last, [(RECENT_START, last)], funds)
    md += t1 + t2 + t3

    md += ['## Margin over the equal-weight hold of the same list (skill net of the list)', '',
           '| | ' + ' | '.join(_ylabel(a, b) for a, b in full) + ' |',
           '|---|' + '---|' * len(full)]
    for name, mom, ew in (('B4 − EW broad', b4, ew_b), ('L4 − EW list', l4, ew_n)):
        md.append(f'| {name} | ' + ' | '.join(
            f'{(a - b) * 100:+.1f}' if np.isfinite(a) and np.isfinite(b) else '—'
            for a, b in zip(by_year(mom, full), by_year(ew, full))) + ' |')

    b = 'B4 momentum, broad list (judged)'
    md += ['', '## Verdict (pre-registered)', '']
    if resolved / len(symbols) < MIN_RESOLVED:
        md.append(f'**NOT JUDGED** — only {resolved / len(symbols):.0%} of listed symbols returned data '
                  f'(rule: at least {MIN_RESOLVED:.0%}).')
    elif b not in j1 or b not in j2:
        md.append('**NOT JUDGED** — fund history or B4 curve missing for a window.')
    else:
        passed = j1[b]['qualifies'] and j2[b]['cagr'] > med2
        md += [f"- Primary: B4 {_pct(j1[b]['cagr'])} CAGR vs median fund {_pct(med1)}, beat it in "
               f"{j1[b]['wins']}/{len(YEARS)} years → QUALIFIES **{'yes' if j1[b]['qualifies'] else 'no'}**; "
               f"MEETS TARGET (best {_pct(best1)}) **{'yes' if j1[b]['meets'] else 'no'}**",
               f"- Secondary: B4 {_pct(j2[b]['cagr'])} CAGR vs median fund {_pct(med2)} → "
               f"**{'above' if j2[b]['cagr'] > med2 else 'not above'}**; MEETS TARGET (best {_pct(best2)}) "
               f"**{'yes' if j2[b]['meets'] else 'no'}**",
               '', f"**{'PASS' if passed else 'FAIL'}**" + (
                   ' — a momentum paper sleeve may be proposed to the owner.' if passed else
                   ' — Phase 5\'s D4 result is attributed to the symbol list; momentum is not deployed.')]
    return _write(out_dir, md)


def _write(out_dir, md):
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'PHASE5B.md'), 'w') as fh:
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
