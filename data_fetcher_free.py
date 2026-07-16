# data_fetcher_free.py
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import warnings
import time
import random

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Symbols that must NOT have .NS appended
_INDEX_SYMBOLS = {'^NSEI', '^BSESN', '^NSEBANK', '^NSMIDCP'}


def _retry(fn, attempts=3, base_delay=1.5, what=""):
    """
    Run fn() with exponential backoff + jitter. Returns fn()'s result, or None
    if every attempt fails. This exists because yfinance calls from shared/
    datacenter IPs (e.g. GitHub Actions runners) get transiently rate-limited
    fairly often — a single failed request should NOT mean "no data forever".
    """
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                delay = base_delay * (2 ** i) + random.uniform(0, 0.5)
                logger.warning(f"⚠️ {what} attempt {i+1}/{attempts} failed ({e}) — retrying in {delay:.1f}s")
                time.sleep(delay)
    logger.error(f"✗ {what} failed after {attempts} attempts: {last_err}")
    return None


class DataFetcherFree:
    def __init__(self):
        logger.info("✓ DataFetcherFree initialized (no API needed)")

    def _to_yf_symbol(self, symbol: str) -> str:
        """Return the correct yfinance ticker string for a given NSE symbol."""
        if symbol in _INDEX_SYMBOLS or symbol.startswith('^'):
            return symbol          # index — use as-is
        return f"{symbol}.NS"      # equity — append .NS

    def get_historical_data(self, symbol, days=200, min_bars=50):
        """
        Fetch historical daily OHLCV data.

        Args:
            symbol  : NSE symbol (e.g. 'RELIANCE') or index (e.g. '^NSEI')
            days    : Number of trading days wanted
            min_bars: Minimum acceptable rows (default 50); pass 1 to skip check
        """
        try:
            yf_symbol  = self._to_yf_symbol(symbol)
            end_date   = datetime.now()
            start_date = end_date - timedelta(days=days + 60)   # buffer for weekends/holidays

            logger.info(f"📥 Fetching {yf_symbol} from {start_date.date()} to {end_date.date()}")

            df = None
            for attempt in range(3):
                try:
                    candidate = yf.download(
                        yf_symbol,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        progress=False,
                        auto_adjust=False,
                    )
                    if candidate is not None and len(candidate) > 0:
                        df = candidate
                        break
                    # Empty result is often silent rate-limiting, not "no data" —
                    # worth a retry rather than trusting it immediately.
                    raise ValueError("empty dataframe returned")
                except Exception as e:
                    if attempt < 2:
                        delay = 1.5 * (2 ** attempt) + random.uniform(0, 0.5)
                        logger.warning(f"⚠️ {symbol} history fetch attempt {attempt+1}/3 failed ({e}) — retrying in {delay:.1f}s")
                        time.sleep(delay)

            if df is None or len(df) == 0:
                logger.error(f"✗ No data returned for {symbol} after retries")
                return None

            # Flatten MultiIndex columns (yfinance quirk with single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            df.reset_index(inplace=True)

            # Normalise column names
            df.columns = [str(c).lower().strip() for c in df.columns]

            # Accept 'date' or 'datetime'
            if 'date' in df.columns and 'datetime' not in df.columns:
                df = df.rename(columns={'date': 'datetime'})

            required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing  = [c for c in required if c not in df.columns]
            if missing:
                logger.error(f"✗ Missing columns {missing} for {symbol}. Got: {list(df.columns)}")
                return None

            df = df[required].dropna().copy()

            if len(df) > days:
                df = df.tail(days).copy()

            df.reset_index(drop=True, inplace=True)

            if len(df) < min_bars:
                logger.warning(f"⚠️ Only {len(df)} bars for {symbol} (min={min_bars})")
                return None

            logger.info(
                f"✓ {symbol}: {len(df)} bars | "
                f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()} | "
                f"close ₹{df['close'].iloc[-1]:.2f}"
            )
            return df

        except Exception as e:
            logger.error(f"✗ Error fetching {symbol}: {e}")
            return None

    def get_ltp(self, symbol, attempts=3):
        """
        Return the last traded / most-recent close price.
        Does NOT apply the min_bars validation — always returns a number or None.
        Retries with backoff — a single failed request should not mean "no price".
        """
        yf_symbol = self._to_yf_symbol(symbol)

        def _fetch():
            ticker = yf.Ticker(yf_symbol)
            data   = ticker.history(period='5d')   # 5d gives at least 1 trading session
            if data is None or len(data) == 0:
                raise ValueError("empty history")
            return float(data['Close'].iloc[-1])

        ltp = _retry(_fetch, attempts=attempts, what=f"get_ltp({symbol})")
        if ltp is not None:
            logger.info(f"✓ {symbol} LTP: ₹{ltp:.2f}")
        else:
            logger.error(f"✗ No LTP data for {symbol}")
        return ltp

    def get_ltp_bulk(self, symbols, attempts=3, chunk_size=40):
        """
        Fetch last-traded prices for MANY symbols in as few HTTP requests as
        possible, using yf.download's multi-ticker support instead of one
        yf.Ticker(...).history() call per symbol.

        Why this matters: looping get_ltp() over N symbols makes N separate
        requests, which is exactly the pattern that gets rate-limited/blocked
        by Yahoo Finance when run from a shared IP (e.g. GitHub Actions). A
        single batched request is far more reliable and much faster.

        Returns: dict {symbol: price} — only symbols that resolved successfully
        are included. Any symbols missing from the result should be treated by
        the caller as "price unknown", not "price is zero".
        """
        symbols = list(dict.fromkeys(symbols))  # de-dupe, preserve order
        results = {}

        for i in range(0, len(symbols), chunk_size):
            chunk        = symbols[i:i + chunk_size]
            yf_to_orig   = {self._to_yf_symbol(s): s for s in chunk}
            yf_symbols   = list(yf_to_orig.keys())

            data = None
            for attempt in range(attempts):
                try:
                    candidate = yf.download(
                        yf_symbols, period='5d', group_by='ticker',
                        progress=False, auto_adjust=False, threads=True,
                    )
                    if candidate is not None and len(candidate) > 0:
                        data = candidate
                        break
                    raise ValueError("empty bulk dataframe")
                except Exception as e:
                    if attempt < attempts - 1:
                        delay = 1.5 * (2 ** attempt) + random.uniform(0, 0.5)
                        logger.warning(
                            f"⚠️ bulk LTP fetch attempt {attempt+1}/{attempts} "
                            f"failed for chunk of {len(chunk)} ({e}) — retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)

            if data is None:
                logger.error(f"✗ Bulk LTP fetch failed entirely for {len(chunk)} symbols — will retry individually")
                continue

            for yf_sym, orig_sym in yf_to_orig.items():
                try:
                    if len(yf_symbols) == 1:
                        # yf.download with a single ticker doesn't use a MultiIndex
                        closes = data['Close'].dropna()
                    else:
                        closes = data[yf_sym]['Close'].dropna()
                    if len(closes) > 0:
                        results[orig_sym] = float(closes.iloc[-1])
                except Exception:
                    continue  # this symbol just wasn't in the bulk result — handled below

        # Anything the bulk call didn't resolve gets a small number of
        # individual retries — worth the extra requests since it's normally
        # just a handful of symbols (e.g. your open positions) at this point.
        missing = [s for s in symbols if s not in results]
        if missing:
            logger.warning(f"  {len(missing)} symbols missing from bulk fetch — retrying individually: {missing}")
            for s in missing:
                price = self.get_ltp(s, attempts=2)
                if price is not None:
                    results[s] = price

        logger.info(f"✓ Bulk LTP: {len(results)}/{len(symbols)} symbols resolved")
        return results

    def get_fundamentals(self, symbol):
        try:
            url     = f"https://www.screener.in/company/{symbol}/"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            resp    = requests.get(url, headers=headers, timeout=10)

            if resp.status_code != 200:
                return self._default_fundamentals()

            soup         = BeautifulSoup(resp.content, 'html.parser')
            fundamentals = {}

            def _extract(label):
                td = soup.find('td', string=label)
                return td.find_next('td').text.strip() if td else None

            try:
                v = _extract('P/E')
                if v:
                    fundamentals['pe_ratio'] = float(v)
            except Exception:
                pass

            try:
                v = _extract('Debt to Equity')
                if v:
                    fundamentals['debt_to_equity'] = float(v)
            except Exception:
                pass

            try:
                v = _extract('ROE')
                if v:
                    fundamentals['roe_5yr'] = float(v.replace('%', '')) / 100
            except Exception:
                pass

            for k, val in self._default_fundamentals().items():
                fundamentals.setdefault(k, val)

            logger.info(f"✓ Fundamentals fetched for {symbol}")
            return fundamentals

        except Exception as e:
            logger.error(f"⚠️ Fundamentals error for {symbol}: {e}")
            return self._default_fundamentals()

    def _default_fundamentals(self):
        return {
            'pe_ratio':        25.0,
            'sector_avg_pe':   25.0,
            'debt_to_equity':   0.70,
            'roe_5yr':          0.18,
            'revenue_cagr':     0.12,
            'current_ratio':    1.3,
            'market_cap':     500000,
        }
