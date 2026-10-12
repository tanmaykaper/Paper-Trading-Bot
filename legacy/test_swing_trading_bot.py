# test_swing_trading_bot.py
# Test Module 6: Complete Bot

from swing_trading_bot import SwingTradingBot
import os
from dotenv import load_dotenv

print("\n" + "="*70)
print("TESTING MODULE 6: COMPLETE SWING TRADING BOT")
print("="*70 + "\n")

load_dotenv()

# Check if email is configured
email_configured = bool(os.getenv('EMAIL_SENDER') and os.getenv('EMAIL_PASSWORD'))

# Initialize bot
bot = SwingTradingBot(send_emails=email_configured)

# Test 1: Screen single stock
print("1️⃣ Test 1: Screen Single Stock")
print("-" * 70)
signal_type, signal_details = bot.screen_stock('RELIANCE')
print(f"Signal: {signal_type}")
if signal_type == 'BUY':
    print(f"  Entry: ₹{signal_details['entry_price']:.2f}")
    print(f"  Target: ₹{signal_details['target_price']:.2f}")

# Test 2: Screen multiple stocks (without sending emails to avoid spam)
print("\n2️⃣ Test 2: Screen Multiple Stocks (without email alerts)")
print("-" * 70)
stocks = ['RELIANCE', 'TCS', 'INFY', 'TATASTEEL', 'MARUTI']
buy_signals = bot.run_screening(stocks, send_alerts=False)

# Test 3: Backtest single stock
print("\n3️⃣ Test 3: Backtest Single Stock")
print("-" * 70)
backtest_results = bot.backtest('RELIANCE', days=200, print_trades=False)

if backtest_results is not None and len(backtest_results) > 0:
    print(f"✓ Backtest completed")
    print(f"✓ Total trades: {len(backtest_results)}")
else:
    print(f"ℹ️ No trades generated (normal for real data with limited history)")

# Test 4: Multi-stock backtest summary
print("\n4️⃣ Test 4: Multi-Stock Backtest")
print("-" * 70)

backtest_stocks = ['RELIANCE', 'TCS']

for stock in backtest_stocks:
    results = bot.backtest(stock, days=200, print_trades=False)
    if results is not None and len(results) > 0:
        print(f"\n✓ {stock}: {len(results)} trades, P&L: ₹{results['pnl'].sum():.0f}")
    else:
        print(f"\nℹ️ {stock}: No trades generated")

print("\n" + "="*70)
print("✅ MODULE 6 TESTS COMPLETE")
print("="*70 + "\n")

print("📧 Email Configuration Status:")
if email_configured:
    print(f"   ✓ Email notifications ENABLED")
    print(f"   ✓ Recipient: {os.getenv('EMAIL_RECIPIENT')}")
    print(f"\n   Next: Run live screening to receive email alerts")
else:
    print(f"   ⚠️ Email notifications DISABLED")
    print(f"\n   To enable: Add credentials to .env file:")
    print(f"   EMAIL_SENDER=your_email@gmail.com")
    print(f"   EMAIL_PASSWORD=your_app_password")
    print(f"   EMAIL_RECIPIENT=your_email@gmail.com")

print()
