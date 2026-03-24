# data_fetcher_free.py
# Module 1: Data Fetcher using FREE sources (yfinance + screener.in)
# FIXED: Handle MultiIndex columns from yfinance

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcherFree:
    """
    Fetch historical OHLCV data from FREE sources
    No API key needed, completely free
    """
    
    def __init__(self):
        """Initialize - no authentication needed"""
        logger.info("✓ DataFetcherFree initialized (no API needed)")
    
    def get_historical_data(self, symbol, days=200):
        """
        Fetch historical daily OHLCV data for NSE stock
        
        Args:
            symbol: NSE stock symbol (e.g., 'RELIANCE', 'TCS')
            days: Number of days of historical data (default 200)
        
        Returns:
            pandas DataFrame with columns: datetime, open, high, low, close, volume
        """
        
        try:
            # NSE symbol format for yfinance: add ".NS"
            yf_symbol = f"{symbol}.NS"
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+50)
            
            logger.info(f"📥 Fetching {symbol} from {start_date.date()} to {end_date.date()}")
            
            # Download from Yahoo Finance
            df = yf.download(
                yf_symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False
            )
            
            # Handle empty data
            if df is None or len(df) == 0:
                logger.error(f"✗ No data returned for {symbol}")
                return None
            
            # CRITICAL FIX: Handle MultiIndex columns
            # yfinance sometimes returns MultiIndex columns, flatten them
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex - take the first level
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Rename 'Date' column to 'datetime'
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'datetime'})
            
            # Make column names lowercase for consistency
            df.columns = [col.lower() for col in df.columns]
            
            # Select only the columns we need
            # Handle different column name variations
            cols_to_use = []
            for col in df.columns:
                if col == 'datetime':
                    cols_to_use.append(col)
                elif col == 'open':
                    cols_to_use.append(col)
                elif col == 'high':
                    cols_to_use.append(col)
                elif col == 'low':
                    cols_to_use.append(col)
                elif col == 'close':
                    cols_to_use.append(col)
                elif col == 'volume':
                    cols_to_use.append(col)
            
            # Ensure we have all required columns
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in cols_to_use for col in required_cols):
                logger.error(f"✗ Missing required columns. Got: {cols_to_use}, Need: {required_cols}")
                return None
            
            df = df[required_cols].copy()
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            # Filter to exact number of days
            if len(df) > days:
                df = df.tail(days).copy()
            
            # Reset index after filtering
            df.reset_index(drop=True, inplace=True)
            
            # Validation
            if len(df) < 50:
                logger.warning(f"⚠️ Only {len(df)} candles fetched (expected {days})")
                return None
            
            logger.info(f"✓ Fetched {len(df)} candles for {symbol}")
            logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            logger.info(f"  Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
            
            return df
        
        except Exception as e:
            logger.error(f"✗ Error fetching {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_ltp(self, symbol):
        """
        Get Last Traded Price (current price)
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
        
        Returns:
            float: Current price, or None if error
        """
        
        try:
            yf_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(yf_symbol)
            
            # Get latest day's data
            data = ticker.history(period='1d')
            
            if len(data) > 0:
                ltp = data['Close'].iloc[-1]
                logger.info(f"✓ {symbol} LTP: ₹{ltp:.2f}")
                return ltp
            else:
                logger.error(f"✗ No data for {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"✗ Error getting LTP for {symbol}: {str(e)}")
            return None
    
    def get_fundamentals(self, symbol):
        """
        Fetch fundamental data by scraping Screener.in
        
        Returns default values if scraping fails (bot still works)
        """
        
        try:
            url = f"https://www.screener.in/company/{symbol}/"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"⚠️ Could not fetch fundamentals for {symbol}")
                return self._default_fundamentals()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            fundamentals = {}
            
            # Try to extract P/E
            try:
                pe_elem = soup.find('td', string='P/E')
                if pe_elem:
                    pe_value = pe_elem.find_next('td').text.strip()
                    fundamentals['pe_ratio'] = float(pe_value)
            except:
                pass
            
            # Try to extract D/E
            try:
                de_elem = soup.find('td', string='Debt to Equity')
                if de_elem:
                    de_value = de_elem.find_next('td').text.strip()
                    fundamentals['debt_to_equity'] = float(de_value)
            except:
                pass
            
            # Try to extract ROE
            try:
                roe_elem = soup.find('td', string='ROE')
                if roe_elem:
                    roe_value = roe_elem.find_next('td').text.strip().replace('%', '')
                    fundamentals['roe_5yr'] = float(roe_value) / 100
            except:
                pass
            
            # Fill in defaults for missing values
            defaults = self._default_fundamentals()
            for key, value in defaults.items():
                fundamentals.setdefault(key, value)
            
            logger.info(f"✓ Fetched fundamentals for {symbol}")
            return fundamentals
        
        except Exception as e:
            logger.error(f"⚠️ Error scraping fundamentals: {str(e)}")
            return self._default_fundamentals()
    
    def _default_fundamentals(self):
        """Return default fundamental values"""
        return {
            'pe_ratio': 25.0,
            'sector_avg_pe': 25.0,
            'debt_to_equity': 0.70,
            'roe_5yr': 0.18,
            'revenue_cagr': 0.12,
            'current_ratio': 1.3,
            'market_cap': 500000,
        }
