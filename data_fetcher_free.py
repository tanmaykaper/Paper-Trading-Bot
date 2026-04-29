# data_fetcher_free.py
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

# Symbols that must NOT have .NS appended
_INDEX_SYMBOLS = {'^NSEI', '^BSESN', '^NSEBANK', '^NSMIDCP'}


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

            df = yf.download(
                yf_symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False,
            )

            if df is None or len(df) == 0:
                logger.error(f"✗ No data returned for {symbol}")
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

    def get_ltp(self, symbol):
        """
        Return the last traded / most-recent close price.
        Does NOT apply the min_bars validation — always returns a number or None.
        """
        try:
            yf_symbol = self._to_yf_symbol(symbol)
            ticker    = yf.Ticker(yf_symbol)
            data      = ticker.history(period='5d')   # 5d gives at least 1 trading session

            if len(data) == 0:
                logger.error(f"✗ No LTP data for {symbol}")
                return None

            ltp = float(data['Close'].iloc[-1])
            logger.info(f"✓ {symbol} LTP: ₹{ltp:.2f}")
            return ltp

        except Exception as e:
            logger.error(f"✗ LTP error for {symbol}: {e}")
            return None

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
