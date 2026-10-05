"""Free market data.

YahooProvider is the default and only required source: daily bars for swing,
5-minute bars for intraday, both via yfinance with chunked bulk downloads,
retries with exponential backoff and a polite pause between chunks (Yahoo
throttles GitHub Actions IPs that hammer it).

Every frame comes back in one schema: a 'datetime' column plus lowercase
open/high/low/close/volume, sorted, de-duplicated, with broken rows dropped.
Intraday timestamps are tz-aware Asia/Kolkata.

Caveat worth knowing: Yahoo's NSE intraday feed can lag the exchange by
several minutes. For paper trading that only delays when a bar is seen — fills
are still simulated at the next bar's price, never at a price the bot could
not have seen.
"""

import logging
import time as _time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IST = 'Asia/Kolkata'
_COLS = ['open', 'high', 'low', 'close', 'volume']


def to_yahoo(symbol):
    return symbol if symbol.startswith('^') else f'{symbol}.NS'


def from_yahoo(ticker):
    return ticker[:-3] if ticker.endswith('.NS') else ticker


def sanitize(df, is_index=False, intraday=False):
    """Normalise one symbol's frame. Returns None when nothing usable is left."""
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    d.columns = [str(c).lower().replace(' ', '_') for c in d.columns]
    if 'adj_close' in d.columns and 'close' not in d.columns:
        d['close'] = d['adj_close']
    missing = [c for c in _COLS if c not in d.columns]
    if missing:
        return None
    d = d[_COLS].apply(pd.to_numeric, errors='coerce')
    idx = pd.to_datetime(d.index)
    if intraday:
        idx = idx.tz_localize('UTC').tz_convert(IST) if idx.tz is None else idx.tz_convert(IST)
    elif idx.tz is not None:
        idx = idx.tz_localize(None)
    d.index = idx
    d = d[~d.index.duplicated(keep='last')].sort_index()
    d = d.dropna(subset=['open', 'high', 'low', 'close'])
    d = d[(d['close'] > 0) & (d['high'] >= d['low'])]
    d['volume'] = d['volume'].fillna(0.0)
    if not is_index:
        d = d[d['volume'] > 0]              # zero-volume bars are halts or stale prints
    if len(d) == 0:
        return None
    d = d.reset_index(names='datetime')
    return d


def completed_bars(df, now, interval_minutes):
    """Drop the bar still forming at `now` — a decision may only read closed bars."""
    if df is None or len(df) == 0:
        return df
    end = pd.to_datetime(df['datetime']) + pd.Timedelta(minutes=interval_minutes)
    now = pd.Timestamp(now)
    if end.dt.tz is not None and now.tzinfo is None:
        now = now.tz_localize(IST)
    return df[end <= now].reset_index(drop=True)


class DataProvider:
    """Interface every market-data source implements."""

    def daily(self, symbols, lookback_days=400):          # pragma: no cover - interface
        raise NotImplementedError

    def intraday(self, symbols, interval_minutes=5, lookback_days=5):  # pragma: no cover
        raise NotImplementedError


class YahooProvider(DataProvider):

    def __init__(self, chunk_size=40, pause_s=1.0, attempts=3, base_delay_s=2.0):
        self.chunk_size = chunk_size
        self.pause_s = pause_s
        self.attempts = attempts
        self.base_delay_s = base_delay_s

    # ── public ──────────────────────────────────────────────────────────────
    def daily(self, symbols, lookback_days=400):
        period = f'{max(int(lookback_days * 1.5), 30)}d'    # calendar days for trading days
        return self._bulk(symbols, period=period, interval='1d', intraday=False)

    def intraday(self, symbols, interval_minutes=5, lookback_days=5):
        # Yahoo serves 5m bars for the last 60 days only.
        period = f'{min(max(int(lookback_days), 1), 59)}d'
        return self._bulk(symbols, period=period, interval=f'{interval_minutes}m', intraday=True)

    # ── internals ───────────────────────────────────────────────────────────
    def _bulk(self, symbols, period, interval, intraday):
        import yfinance as yf

        symbols = list(dict.fromkeys(symbols))
        out = {}
        for i in range(0, len(symbols), self.chunk_size):
            chunk = symbols[i:i + self.chunk_size]
            tickers = [to_yahoo(s) for s in chunk]
            raw = self._with_retry(
                lambda: yf.download(tickers=tickers, period=period, interval=interval,
                                    group_by='ticker', auto_adjust=True, threads=True,
                                    progress=False),
                what=f'{interval} chunk {i // self.chunk_size + 1} ({len(chunk)} symbols)')
            for sym, frame in self._split(raw, tickers).items():
                clean = sanitize(frame, is_index=sym.startswith('^'), intraday=intraday)
                if clean is not None:
                    out[sym] = clean
            if i + self.chunk_size < len(symbols):
                _time.sleep(self.pause_s)

        missing = [s for s in symbols if s not in out]
        if missing:
            logger.info(f'  {len(missing)}/{len(symbols)} symbols returned no {interval} data: '
                        f'{missing[:12]}{" …" if len(missing) > 12 else ""}')
        return out

    @staticmethod
    def _split(raw, tickers):
        """yfinance returns (ticker, field) MultiIndex columns for group_by='ticker';
        older builds and single tickers can return flat columns. Handle both."""
        if raw is None or len(raw) == 0:
            return {}
        frames = {}
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = set(raw.columns.get_level_values(0))
            for t in tickers:
                if t in level0:
                    frames[from_yahoo(t)] = raw[t]
                elif t in set(raw.columns.get_level_values(-1)):
                    frames[from_yahoo(t)] = raw.xs(t, axis=1, level=-1)
        elif len(tickers) == 1:
            frames[from_yahoo(tickers[0])] = raw
        return frames

    def _with_retry(self, fn, what):
        for attempt in range(self.attempts):
            try:
                return fn()
            except Exception as e:                       # network, JSON, throttling
                if attempt == self.attempts - 1:
                    logger.error(f'  ✗ {what} failed after {self.attempts} attempts: {e}')
                    return None
                delay = self.base_delay_s * (2 ** attempt) + np.random.uniform(0, 0.5)
                logger.warning(f'  {what} attempt {attempt + 1} failed ({e}); retry in {delay:.1f}s')
                _time.sleep(delay)
        return None
