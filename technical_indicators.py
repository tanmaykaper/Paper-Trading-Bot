# technical_indicators.py
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(data, period):
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(data, period):
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}

    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {'upper': upper, 'middle': sma, 'lower': lower}

    @staticmethod
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / 
                        TechnicalIndicators.calculate_atr(high, low, close, period))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / 
                         TechnicalIndicators.calculate_atr(high, low, close, period))
        
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * (di_diff / di_sum)
        adx = dx.rolling(window=period).mean()
        return adx

    @staticmethod
    def calculate_volume_sma(volume, period=10):
        return volume.rolling(window=period).mean()

    @staticmethod
    def calculate_stochastic(high, low, close, period=14):
        """Stochastic oscillator: identifies overbought/oversold"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return {'k': k_percent, 'd': d_percent}

    @staticmethod
    def calculate_williams_r(high, low, close, period=14):
        """Williams %R: identifies support/resistance zones"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr

    @staticmethod
    def calculate_obv(close, volume):
        """On-Balance Volume: validates trend with volume"""
        obv = volume.copy()
        obv[close < close.shift()] = -obv
        return obv.cumsum()

    @staticmethod
    def calculate_cmf(high, low, close, volume, period=20):
        """Chaikin Money Flow: accumulation/distribution indicator"""
        mfv = ((close - low) - (high - close)) / (high - low) * volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf

    @staticmethod
    def detect_divergence(prices, rsi, lookback=5):
        """Detect bullish/bearish divergences for entry signals"""
        try:
            recent_price_low = prices.iloc[-lookback:].min()
            recent_rsi_low = rsi.iloc[-lookback:].min()
            
            prev_price_low = prices.iloc[-2*lookback:-lookback].min()
            prev_rsi_low = rsi.iloc[-2*lookback:-lookback].min()
            
            # Bullish divergence: lower lows in price, higher lows in RSI
            bullish = (recent_price_low < prev_price_low) and (recent_rsi_low > prev_rsi_low)
            
            return bullish
        except:
            return False

    @staticmethod
    def detect_support_resistance(high, low, lookback=10):
        """Identify key support and resistance levels"""
        resistance = high.tail(lookback).max()
        support = low.tail(lookback).min()
        return {'resistance': resistance, 'support': support}

    @staticmethod
    def calculate_momentum(close, period=10):
        """Momentum: rate of price change"""
        return close.diff(period)
