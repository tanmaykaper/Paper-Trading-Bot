"""Every NSE-listed equity, from NSE's own file, with a committed fallback copy.

The momentum sleeve picks from the whole market, not a hand-made list: a
list drawn up today carries hindsight that was worth about 15 points a year
in backtests (docs/RESEARCH.md, round 6). NSE publishes the list as
EQUITY_L.csv, which GitHub Actions can download. Each good download is saved
under the sleeve's state directory, and the workflow commits it. When NSE
can't be reached, that copy is used for up to `max_age_days`; after that the
run stops rather than trade a stale market.
"""

import io
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

SOURCES = ('https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv',
           'https://archives.nseindia.com/content/equities/EQUITY_L.csv')
MIN_SYMBOLS = 300           # an error page or a truncated file must never pass as the market
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/124.0 Safari/537.36',
           'Accept': 'text/csv,text/plain,*/*', 'Referer': 'https://www.nseindia.com/'}


class ListUnavailable(RuntimeError):
    """Neither NSE nor a fresh enough saved copy could supply the list."""


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


def parse_equity_list(text, series):
    """Symbols in the wanted series from EQUITY_L.csv text; [] if it isn't one."""
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return []
    df.columns = [str(c).strip().upper() for c in df.columns]      # NSE pads its headers
    if 'SYMBOL' not in df.columns or 'SERIES' not in df.columns:
        return []
    df = df[df['SERIES'].astype(str).str.strip().isin(series)]
    return sorted({str(s).strip() for s in df['SYMBOL'].dropna() if str(s).strip()})


def nse_equities(cache_path, series=('EQ', 'BE', 'BZ'), max_age_days=30, get=_get_text, today=None):
    """(symbols, note). Downloads the list, saving it to `cache_path`; falls
    back to the saved copy while it is younger than `max_age_days`."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz='Asia/Kolkata').date())
    for url in SOURCES:
        text = get(url)
        syms = parse_equity_list(text, series) if text else []
        if len(syms) >= MIN_SYMBOLS:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            tmp = cache_path + '.tmp'
            with open(tmp, 'w') as fh:
                json.dump({'fetched': str(today.date()), 'source': url, 'series': list(series),
                           'symbols': syms}, fh, indent=0)
            os.replace(tmp, cache_path)
            return syms, f'NSE EQUITY_L ({len(syms)} symbols, {"/".join(series)})'
        if text:
            logger.warning(f'  {url}: {len(syms)} symbols — not accepted as the market')
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            saved = json.load(fh)
        age = (today - pd.Timestamp(saved['fetched'])).days
        if age <= max_age_days and len(saved.get('symbols', [])) >= MIN_SYMBOLS:
            return saved['symbols'], f"saved NSE list from {saved['fetched']} ({age} days old; NSE unreachable)"
        raise ListUnavailable(f"NSE unreachable and the saved list is {age} days old (limit {max_age_days})")
    raise ListUnavailable('NSE unreachable and no saved list exists')
