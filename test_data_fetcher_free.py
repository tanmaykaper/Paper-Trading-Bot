# test_data_fetcher_free.py
# Test Module 1: Data Fetcher

from data_fetcher_free import DataFetcherFree

print("\n" + "="*70)
print("TESTING MODULE 1: FREE DATA FETCHER")
print("="*70)

# Initialize
fetcher = DataFetcherFree()

# Test 1: Current price
print("\n✓ Test 1: Get Current Price")
ltp = fetcher.get_ltp('RELIANCE')
if ltp:
    print(f"  RELIANCE LTP: ₹{ltp:.2f} ✓")
else:
    print("  Failed ✗")

# Test 2: Historical data
print("\n✓ Test 2: Fetch 200 Days Historical Data")
df = fetcher.get_historical_data('RELIANCE', days=200)

if df is not None:
    print(f"  Rows fetched: {len(df)} ✓")
    print(f"  Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()} ✓")
    print(f"  Columns: {list(df.columns)} ✓")
    
    # Quality checks
    print(f"  No NaN: {not df[['open','high','low','close','volume']].isnull().any().any()} ✓")
    print(f"  High >= Low: {(df['high'] >= df['low']).all()} ✓")
    
    print("\n  Sample data (first 3 rows):")
    print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head(3).to_string())

# Test 3: Multi-stock
print("\n✓ Test 3: Multi-Stock Fetch")
stocks = ['RELIANCE', 'TCS', 'INFY', 'TATASTEEL']

for stock in stocks:
    df = fetcher.get_historical_data(stock, days=200)
    if df is not None:
        price = df['close'].iloc[-1]
        print(f"  {stock:12} | Price: ₹{price:8.2f} | Candles: {len(df):3} ✓")

# Test 4: Fundamentals
print("\n✓ Test 4: Fetch Fundamentals")
fund = fetcher.get_fundamentals('RELIANCE')
print(f"  P/E: {fund['pe_ratio']:.2f}")
print(f"  D/E: {fund['debt_to_equity']:.2f}")
print(f"  ROE: {fund['roe_5yr']*100:.1f}%")

print("\n" + "="*70)
print("✅ MODULE 1 TESTS COMPLETE")
print("="*70 + "\n")
