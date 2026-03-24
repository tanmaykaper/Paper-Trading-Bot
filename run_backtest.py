# run_backtest.py  ── HIGH-RISK / HIGH-FREQUENCY VERSION v3
# 40-stock universe for more trade volume in backtest

from swing_trading_bot import SwingTradingBot
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("\n" + "="*70)
print("NSE SWING TRADING BOT — PORTFOLIO BACKTEST v3 (HIGH-RISK)")
print("="*70 + "\n")

INITIAL_EQUITY  = 50_000
BACKTEST_DAYS   = 600   # ~2.5 years of daily data

# 40-stock universe across diverse sectors for maximum trade opportunities
BACKTEST_STOCKS = [
    # IT (high beta, fast movers)
    'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'MPHASIS',
    # Banks
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
    # FMCG / Consumer
    'HINDUNILVR', 'ITC', 'NESTLEIND', 'ASIANPAINT', 'TITAN',
    # Energy
    'RELIANCE', 'ONGC', 'BPCL',
    # Auto
    'MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'M&M',
    # Metals
    'TATASTEEL', 'JSWSTEEL', 'HINDALCO',
    # Pharma
    'SUNPHARMA', 'DRREDDY', 'CIPLA',
    # Infra / Capital Goods
    'LT', 'SIEMENS', 'ABB',
    # NBFC
    'BAJFINANCE', 'CHOLAFIN',
    # Cement
    'ULTRACEMCO', 'AMBUJACEM',
    # Telecom / Others
    'BHARTIARTL', 'DLF',
]

bot = SwingTradingBot(
    send_emails=False,
    initial_equity=INITIAL_EQUITY,
    max_open_trades=7,
    max_hold_days=15,
)

results_df = bot.backtest_portfolio(BACKTEST_STOCKS, days=BACKTEST_DAYS)

if results_df is not None and len(results_df) > 0:
    print("\n" + "="*70)
    print("PER-STOCK BREAKDOWN")
    print("="*70)
    by_stock = results_df.groupby('symbol').agg(
        trades    = ('net_pnl', 'count'),
        wins      = ('result', lambda x: (x == 'WIN').sum()),
        total_pnl = ('net_pnl', 'sum'),
        avg_pnl   = ('net_pnl', 'mean'),
        avg_hold  = ('hold_days', 'mean'),
    )
    by_stock['win_rate%'] = (by_stock['wins'] / by_stock['trades'] * 100).round(1)
    by_stock = by_stock.sort_values('total_pnl', ascending=False)
    print(by_stock[['trades','wins','win_rate%','total_pnl','avg_pnl','avg_hold']].to_string())

    print("\n" + "="*70)
    print("EXIT REASON BREAKDOWN")
    print("="*70)
    print(results_df.groupby('exit_reason')['net_pnl'].agg(['count','sum','mean']).to_string())

    if 'entry_type' in results_df.columns:
        print("\n" + "="*70)
        print("PATTERN BREAKDOWN (which patterns are actually making money)")
        print("="*70)
        print(results_df.groupby('entry_type').agg(
            trades    = ('net_pnl','count'),
            total_pnl = ('net_pnl','sum'),
            avg_pnl   = ('net_pnl','mean'),
            wins      = ('result', lambda x: (x=='WIN').sum()),
        ).assign(win_rate=lambda d: (d['wins']/d['trades']*100).round(1)
        ).drop('wins',axis=1).sort_values('total_pnl',ascending=False).to_string())

    results_df.to_csv('backtest_results_v3.csv', index=False)
    print("\n✓ Full results saved to backtest_results_v3.csv")
else:
    print("⚠️ No trades generated")
