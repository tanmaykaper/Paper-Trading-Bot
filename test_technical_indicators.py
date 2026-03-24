# test_technical_indicators.py
# Test Module 2: Technical Indicators

import pandas as pd
import numpy as np
from data_fetcher_free import DataFetcherFree
from technical_indicators import TechnicalIndicators

print("\n" + "="*70)
print("TESTING MODULE 2: TECHNICAL INDICATORS")
print("="*70)

# Get sample data
print("\n1️⃣ Fetching sample data...")
fetcher = DataFetcherFree()
df = fetcher.get_historical_data('RELIANCE', days=200)

if df is None:
    print("✗ Failed to fetch data")
    exit()

print(f"   ✓ Got {len(df)} candles")

# Test individual indicators
print("\n2️⃣ Testing Individual Indicators")

tech = TechnicalIndicators()

# EMA-20
ema_20 = tech.calculate_ema(df['close'], 20)
print(f"   ✓ EMA-20 last value: {ema_20.iloc[-1]:.2f}")
assert ema_20.shape[0] == len(df), "EMA should have same length"

# SMA-200
sma_200 = tech.calculate_sma(df['close'], 200)
print(f"   ✓ SMA-200 last value: {sma_200.iloc[-1]:.2f}")

# RSI
rsi = tech.calculate_rsi(df['close'], 14)
print(f"   ✓ RSI last value: {rsi.iloc[-1]:.2f}")
assert (rsi.iloc[20:] >= 0).all() and (rsi.iloc[20:] <= 100).all(), "RSI should be 0-100"

# MACD
macd = tech.calculate_macd(df['close'])
print(f"   ✓ MACD last value: {macd['macd'].iloc[-1]:.4f}")
print(f"   ✓ MACD Histogram last value: {macd['histogram'].iloc[-1]:.4f}")

# Bollinger Bands
bb = tech.calculate_bollinger_bands(df['close'], 20, 2)
print(f"   ✓ BB Upper: {bb['upper'].iloc[-1]:.2f}")
print(f"   ✓ BB Middle: {bb['middle'].iloc[-1]:.2f}")
print(f"   ✓ BB Lower: {bb['lower'].iloc[-1]:.2f}")

# ATR
atr = tech.calculate_atr(df['high'], df['low'], df['close'], 14)
print(f"   ✓ ATR last value: {atr.iloc[-1]:.2f}")
assert (atr.iloc[20:] > 0).all(), "ATR should be positive"

# ADX
adx = tech.calculate_adx(df['high'], df['low'], df['close'], 14)
print(f"   ✓ ADX last value: {adx.iloc[-1]:.2f}")

print("\n3️⃣ Testing Calculate All (Batch)")
df_with_indicators = tech.calculate_all(df)
print(f"   ✓ Original columns: {len(df.columns)}")
print(f"   ✓ With indicators: {len(df_with_indicators.columns)}")
print(f"   ✓ New columns added: {df_with_indicators.columns.tolist()}")

print("\n4️⃣ Sample Data with All Indicators (Last Row)")
print(df_with_indicators[['datetime', 'close', 'ema_20', 'sma_200', 'rsi', 'macd', 'atr', 'adx']].tail(1).to_string())

print("\n5️⃣ Verification Against Expected Values")
latest = df_with_indicators.iloc[-1]
prev = df_with_indicators.iloc[-2]

# Trend check
in_uptrend = latest['close'] > latest['ema_20'] > latest['ema_50'] > latest['sma_200']
print(f"   Close > EMA-20 > EMA-50 > SMA-200: {in_uptrend}")

# RSI check
rsi_in_range = 30 < latest['rsi'] < 70
print(f"   RSI in 30-70 range: {rsi_in_range} (RSI: {latest['rsi']:.2f})")

# MACD check
macd_bullish = latest['macd_histogram'] > 0
print(f"   MACD Histogram > 0 (bullish): {macd_bullish}")

# Trend strength check
strong_trend = latest['adx'] > 25
print(f"   ADX > 25 (strong trend): {strong_trend} (ADX: {latest['adx']:.2f})")

print("\n6️⃣ Volume Analysis")
volume_sma = tech.calculate_volume_sma(df['volume'], 10)
latest_vol_ratio = latest['volume'] / volume_sma.iloc[-1]
print(f"   Current Volume: {int(latest['volume']):,}")
print(f"   10-day Avg Volume: {int(volume_sma.iloc[-1]):,}")
print(f"   Volume Ratio: {latest_vol_ratio:.2f}x")

print("\n" + "="*70)
print("✅ MODULE 2 TESTS COMPLETE")
print("="*70 + "\n")
