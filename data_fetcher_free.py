# data_fetcher_free.py  ── FIXED v2
# ─────────────────────────────────────────────────────────────────────────────
# Fixes vs original:
#   1. INDEX_SYMBOLS set — these get passed to yfinance as-is (no .NS suffix)
#   2. get_ltp() and new get_ltp_bulk() use yf.download for a batch fetch,
#      much faster than one Ticker() call per symbol
#   3. Minimum-bar validation split into two thresholds:
#         historical fetch  → requires 50 bars (unchanged)
#         price-only fetch  → no minimum (1 bar is fine)
#   4. MultiIndex flattening is more robust
# ─────────────────────────────────────────────────────────────────────────────

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These symbols must NOT have .NS appended
INDEX_SYMBOLS = {'^NSEI', '^NSEBANK', '^CNXIT', '^CNXPHARMA', '^CNXAUTO'}


def _yf_symbol(symbol: str) -> str:
    """Convert NSE ticker to yfinance symbol."""
    if symbol in INDEX_SYMBOLS or symbol.startswith('^'):
        return symbol          # index — use as-is
    return f"{symbol}.NS"     # equity — add .NS


class DataFetcherFree:
    """
    Fetch historical OHLCV data from FREE sources.
    No API key needed.
    """

    def __init__(self):
        logger.info("✓ DataFetcherFree v2 initialized (no API needed)")

    # ── Core OHLCV fetch ─────────────────────────────────────────────────────

    def get_historical_data(self, symbol: str, days: int = 200):
        """
        Fetch historical daily OHLCV data.

        Args:
            symbol : NSE ticker (e.g. 'RELIANCE') or index (e.g. '^NSEI')
            days   : calendar days of history requested

        Returns:
            DataFrame[datetime, open, high, low, close, volume]  or None
        """
        try:
            yf_sym    = _yf_symbol(symbol)
            end_date  = datetime.now()
            start_date = end_date - timedelta(days=days + 50)   # buffer for weekends/holidays

            logger.info(f"📥 Fetching {symbol} ({yf_sym}) "
                        f"from {start_date.date()} to {end_date.date()}")

            df = yf.download(
                yf_sym,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False,
            )

            if df is None or len(df) == 0:
                logger.error(f"✗ No data returned for {symbol}")
                return None

            df = self._normalise(df)
            if df is None:
                return None

            # Trim to requested window
            if len(df) > days:
                df = df.tail(days).copy()
            df.reset_index(drop=True, inplace=True)

            if len(df) < 50:
                logger.warning(f"⚠️ Only {len(df)} candles for {symbol} (need ≥ 50)")
                return None

            logger.info(f"✓ {symbol}: {len(df)} candles | "
                        f"₹{df['close'].min():.2f}–₹{df['close'].max():.2f}")
            return df

        except Exception as e:
            logger.error(f"✗ Error fetching {symbol}: {e}")
            return None

    # ── LTP helpers ───────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str):
        """
        Get last traded price for a single symbol.

        Returns: float or None
        """
        try:
            yf_sym = _yf_symbol(symbol)
            data   = yf.download(yf_sym, period='5d', progress=False, auto_adjust=False)
            if data is None or len(data) == 0:
                logger.error(f"✗ No LTP data for {symbol}")
                return None
            df = self._normalise(data)
            if df is None or len(df) == 0:
                return None
            ltp = float(df['close'].iloc[-1])
            logger.info(f"✓ {symbol} LTP: ₹{ltp:.2f}")
            return ltp
        except Exception as e:
            logger.error(f"✗ LTP error for {symbol}: {e}")
            return None

    def get_ltp_bulk(self, symbols: list, period: str = '5d') -> dict:
        """
        Batch-fetch last close prices for a list of symbols.
        Much faster than one call per symbol.

        Returns: {symbol: price, ...}  (only symbols with valid data)
        """
        if not symbols:
            return {}

        yf_syms   = [_yf_symbol(s) for s in symbols]
        sym_map   = {_yf_symbol(s): s for s in symbols}   # yf_sym → original

        try:
            raw = yf.download(
                yf_syms,
                period=period,
                progress=False,
                auto_adjust=False,
                group_by='ticker',
            )
        except Exception as e:
            logger.error(f"✗ Bulk download error: {e}")
            return {}

        prices = {}

        if isinstance(raw.columns, pd.MultiIndex):
            # group_by='ticker' → top level is ticker, second level is OHLCV
            for yf_sym in yf_syms:
                orig = sym_map.get(yf_sym, yf_sym)
                try:
                    sub = raw[yf_sym] if yf_sym in raw.columns.get_level_values(0) else None
                    if sub is None or len(sub) == 0:
                        continue
                    # close column (case-insensitive)
                    close_col = next((c for c in sub.columns if c.lower() == 'close'), None)
                    if close_col is None:
                        continue
                    val = sub[close_col].dropna()
                    if len(val) == 0:
                        continue
                    prices[orig] = float(val.iloc[-1])
                except Exception:
                    pass
        else:
            # Single-symbol or flat columns
            df = self._normalise(raw)
            if df is not None and len(df) > 0 and len(symbols) == 1:
                prices[symbols[0]] = float(df['close'].iloc[-1])

        logger.info(f"✓ Bulk LTP: got {len(prices)}/{len(symbols)} prices")
        return prices

    # ── Fundamentals ──────────────────────────────────────────────────────────

    def get_fundamentals(self, symbol: str) -> dict:
        """
        Scrape Screener.in for basic fundamentals.
        Returns defaults if scraping fails (bot still works).
        """
        try:
            url = f"https://www.screener.in/company/{symbol}/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return self._default_fundamentals()

            soup = BeautifulSoup(resp.content, 'html.parser')
            fund = {}

            def _num(text: str) -> float:
                try:
                    return float(text.strip().replace(',', '').replace('%', ''))
                except Exception:
                    return 0.0

            for label, key in [
                ('P/E',             'pe_ratio'),
                ('Debt to Equity',  'debt_to_equity'),
                ('ROE',             'roe_5yr'),
            ]:
                elem = soup.find('td', string=label)
                if elem:
                    val = elem.find_next('td')
                    if val:
                        fund[key] = _num(val.text)

            # ROE from Screener is a percentage — normalise to fraction
            if 'roe_5yr' in fund and fund['roe_5yr'] > 1:
                fund['roe_5yr'] /= 100

            defaults = self._default_fundamentals()
            for k, v in defaults.items():
                fund.setdefault(k, v)

            logger.info(f"✓ Fundamentals for {symbol}: PE={fund.get('pe_ratio')}, "
                        f"D/E={fund.get('debt_to_equity')}, ROE={fund.get('roe_5yr'):.1%}")
            return fund

        except Exception as e:
            logger.error(f"⚠️ Fundamentals scrape failed for {symbol}: {e}")
            return self._default_fundamentals()

    def _default_fundamentals(self) -> dict:
        return {
            'pe_ratio':        25.0,
            'sector_avg_pe':   25.0,
            'debt_to_equity':   0.70,
            'roe_5yr':          0.18,
            'revenue_cagr':     0.12,
            'current_ratio':    1.30,
            'market_cap':    500000,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise(df: pd.DataFrame):
        """
        Flatten MultiIndex columns, rename Date → datetime,
        lowercase all column names, select required columns.
        Returns clean DataFrame or None.
        """
        if df is None or len(df) == 0:
            return None

        # Flatten MultiIndex (single-ticker downloads sometimes produce these)
        if isinstance(df.columns, pd.MultiIndex):
            # For a single ticker: (Price, Ticker) → keep Price level
            # Detect: if second level has only one unique non-empty value
            lvl1 = [c[1] for c in df.columns if isinstance(c, tuple)]
            unique_tickers = {t for t in lvl1 if t}
            if len(unique_tickers) <= 1:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            # Multi-ticker downloads should be handled by get_ltp_bulk separately

        df = df.reset_index()

        # Rename index column
        for old in ('Date', 'Datetime', 'date', 'datetime', 'index'):
            if old in df.columns:
                df = df.rename(columns={old: 'datetime'})
                break

        df.columns = [str(c).lower() for c in df.columns]

        required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"✗ Missing columns after normalise: {missing} | Got: {list(df.columns)}")
            return None

        df = df[required].dropna().copy()
        return df
