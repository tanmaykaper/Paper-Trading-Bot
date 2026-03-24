# run_live_screening.py  ── HIGH-RISK / HIGH-FREQUENCY VERSION v3
# Screens 40+ NSE stocks and fires email alerts for every BUY signal found.

from swing_trading_bot import SwingTradingBot
import os
from dotenv import load_dotenv

load_dotenv()

email_configured = bool(os.getenv('EMAIL_SENDER') and os.getenv('EMAIL_PASSWORD'))

if not email_configured:
    print("\n⚠️  Email not configured — signals will print to console only.")
    print("   Add EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECIPIENT to .env to enable alerts.\n")

print("\n" + "="*70)
print("NSE SWING TRADING BOT — LIVE SCREENING v3 (HIGH-RISK)")
print("="*70 + "\n")

bot = SwingTradingBot(send_emails=email_configured)

# 40-stock universe — broad enough to always find setups
NSE_STOCKS = [
    # IT
    'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'MPHASIS',
    # Banks
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'FEDERALBNK',
    # FMCG / Consumer
    'HINDUNILVR', 'ITC', 'NESTLEIND', 'ASIANPAINT', 'TITAN', 'MARICO',
    # Energy
    'RELIANCE', 'ONGC', 'BPCL', 'GAIL',
    # Auto
    'MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'M&M', 'EICHERMOT',
    # Metals
    'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'SAIL',
    # Pharma
    'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB',
    # Infra / Cap Goods
    'LT', 'SIEMENS', 'ABB',
    # NBFC
    'BAJFINANCE', 'CHOLAFIN', 'MUTHOOTFIN',
    # Realty
    'DLF', 'GODREJPROP',
    # Others
    'BHARTIARTL', 'ULTRACEMCO', 'ADANIPORTS',
]

print(f"📡 Scanning {len(NSE_STOCKS)} stocks...\n")

buy_signals = bot.run_screening(NSE_STOCKS, send_alerts=email_configured)

print("\n" + "="*70)
print(f"RESULTS — {len(buy_signals)} BUY signal(s) found")
print("="*70)

if buy_signals:
    # Sort by confidence (most patterns triggered = highest conviction first)
    buy_signals.sort(key=lambda x: x[1].get('confidence', 1), reverse=True)
    for symbol, d in buy_signals:
        patterns = ', '.join(d.get('patterns_triggered', [d['entry_type']]))
        print(f"\n  🎯 {symbol}")
        print(f"     Entry      : ₹{d['entry_price']:.2f}")
        print(f"     Stop-Loss  : ₹{d['stop_loss']:.2f}")
        print(f"     Target     : ₹{d['target_price']:.2f}")
        print(f"     R:R        : 1:{d['risk_reward_ratio']:.1f}")
        print(f"     Patterns   : {patterns}")
        print(f"     Confidence : {d.get('confidence', 1)}/9")
        print(f"     RSI        : {d['indicators']['rsi']:.1f}")
        print(f"     ADX        : {d['indicators']['adx']:.1f}")
        if email_configured:
            print(f"     📧 Email alert sent")
else:
    print("\n  No BUY signals today. Run again tomorrow.")

print("\n" + "="*70 + "\n")
