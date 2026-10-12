"""Symbol check — every name in the live universes against NSE's own list.

    python -m nsebot.research.symbol_check --out research_results

Research runs log five SWING_UNIVERSE names as 'possibly delisted' on Yahoo
every time (JKBANK, BIRLASOFT, SUVENPHAR, APOLLOMICRO, ORIENTGREEN), so the
live bot scans fewer stocks than its list says. This finds every universe
symbol missing from NSE's EQUITY_L.csv, searches the list's company names for
the likely current symbol, and asks Yahoo whether each candidate has data.
Read-only: it changes nothing and trades nothing.
"""

import argparse
import io
import logging
import os
import sys

import pandas as pd

from .phase5b import LIST_SOURCES, _get_text

logger = logging.getLogger(__name__)

# Likely successors, to be CONFIRMED by this run (renames, NSE symbols that
# differ from the brand name). Name hints search NSE's company-name column.
CANDIDATES = {'JKBANK': ['J&KBANK'], 'BIRLASOFT': ['BSOFT'], 'APOLLOMICRO': ['APOLLO'],
              'ORIENTGREEN': ['GREENPOWER'], 'SUVENPHAR': ['COHANCE']}
NAME_HINTS = {'JKBANK': ['JAMMU'], 'BIRLASOFT': ['BIRLASOFT'], 'APOLLOMICRO': ['APOLLO MICRO'],
              'ORIENTGREEN': ['ORIENT GREEN'], 'SUVENPHAR': ['SUVEN', 'COHANCE']}


def equity_table(get=_get_text):
    """NSE's EQUITY_L as a DataFrame (SYMBOL, NAME, SERIES), or None."""
    for _, url in LIST_SOURCES[:2]:                  # EQUITY_L only: the index lists have no full names
        text = get(url)
        if not text:
            continue
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            continue
        df.columns = [str(c).strip().upper() for c in df.columns]
        if {'SYMBOL', 'NAME OF COMPANY'} <= set(df.columns):
            df = df.rename(columns={'NAME OF COMPANY': 'NAME'})
            df['SERIES'] = df.get('SERIES', pd.Series('', index=df.index)).astype(str).str.strip()
            df['SYMBOL'] = df['SYMBOL'].astype(str).str.strip()
            return df[['SYMBOL', 'NAME', 'SERIES']]
    return None


def yahoo_rows(symbols):
    """{symbol: number of daily bars Yahoo returns for the last month}."""
    from ..data import YahooProvider
    got = YahooProvider(chunk_size=20).daily(list(symbols), lookback_days=20)
    return {s: (len(got[s]) if s in got else 0) for s in symbols}


def run(out_dir, table=None, rows=None, get=_get_text):
    from ..universe import INTRADAY_UNIVERSE, SWING_UNIVERSE

    os.makedirs(out_dir, exist_ok=True)
    table = table if table is not None else equity_table(get)
    md = ['# Symbol check — live universes against NSE EQUITY_L', '']
    if table is None:
        md.append('**NOT RUN:** NSE\'s equity list could not be retrieved.')
        return _write(out_dir, md)
    listed = set(table['SYMBOL'])
    universe = list(dict.fromkeys(SWING_UNIVERSE + INTRADAY_UNIVERSE))
    missing = [s for s in universe if s not in listed]
    md += [f'{len(universe)} universe symbols, {len(missing)} missing from EQUITY_L '
           f'({len(table)} listed rows).', '']
    if not missing:
        return _write(out_dir, md + ['Every universe symbol is listed.'])

    probe = set(missing)
    matches = {}
    for s in missing:
        hints = NAME_HINTS.get(s, [s])
        hit = table[table['NAME'].str.upper().apply(lambda n: any(h in n for h in hints))]
        matches[s] = list(zip(hit['SYMBOL'], hit['NAME'], hit['SERIES']))
        probe |= set(CANDIDATES.get(s, [])) | {m[0] for m in matches[s]}
    rows = rows if rows is not None else yahoo_rows(sorted(probe))

    md += ['| universe symbol | Yahoo bars (1 month) | candidates (in EQUITY_L? · Yahoo bars) | '
           'company-name matches in EQUITY_L |', '|---|---|---|---|']
    for s in missing:
        cands = ', '.join(f"{c} ({'listed' if c in listed else 'NOT listed'} · {rows.get(c, 0)})"
                          for c in CANDIDATES.get(s, [])) or '—'
        names = '; '.join(f'{sym} = {name} [{ser}] · {rows.get(sym, 0)} bars'
                          for sym, name, ser in matches[s][:6]) or '—'
        md.append(f'| {s} | {rows.get(s, 0)} | {cands} | {names} |')
    return _write(out_dir, md)


def _write(out_dir, md):
    report = '\n'.join(md)
    with open(os.path.join(out_dir, 'SYMBOL_CHECK.md'), 'w') as fh:
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
