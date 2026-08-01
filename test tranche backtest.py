# test_tranche_backtest.py
#
# Validates that swing_trading_bot.py's backtest_portfolio() correctly
# simulates the SAME scaled-exit (tranche) system run_paper_trading.py
# actually trades live — the gap this was built to close (see
# tranche_manager.py's header for the full story: the backtester previously
# had no concept of tranches at all, silently testing a different, older
# exit policy than the one actually deployed).
#
# Uses fully controlled, deliberately-constructed synthetic price paths
# (not real market data — this sandbox has no live NSE/yfinance access,
# same constraint as the rest of this project's test suite) specifically so
# the expected outcome at each step is known in advance and can be asserted
# exactly, rather than eyeballed. Three things are checked:
#
#   1. RUNNER_WIN — quick/core/runner each close independently at their OWN
#      target, and the runner (no fixed target) captures MORE than a fixed
#      target would have — the entire point of tranching.
#   2. STOPPED_OUT — a loss closes EVERY tranche together, at the same
#      shared stop price, on the same bar — a losing trade can't "partially
#      survive" by construction.
#   3. use_scaled_exits=False reproduces the OLD untranched behaviour
#      exactly (one row, full size, capped at the original fixed target) —
#      backward compatibility for anyone who wants the pre-tranching policy.
#
# Run: python3 test_tranche_backtest.py

import sys
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from swing_trading_bot import SwingTradingBot
import data_fetcher_free

PASS, FAIL = 0, 0
def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  \u2713 {label}")
    else:
        FAIL += 1; print(f"  \u2717 {label}  {detail}")

N = 200
dates = pd.date_range('2024-01-01', periods=N, freq='B')

def flat(price, n=N):
    return np.full(n, price, dtype=float)

# ── RUNNER_WIN: entry@bar60=100, stop=95 (risk=5), target=115 (3R) ─────────
# Price: flat @100 pre-entry, then climbs steadily past quick(107.5)/core(115)
# up to 150 by bar 90, then holds near 148-150 until time exit (15 bars from
# entry -> exits at bar 75... wait max_hold_days=15 -> bar 60+15=75).
# Redesign: climb needs to reach quick+core WELL before bar 75, then hold
# high through the time-exit bar so the runner exits near its peak, not on
# the way up (which would understate what tranching captures).
close_runner = flat(100, N)
# bars 0-59: flat pre-entry at 100 (also serves as the lookback window
# generate_signal would see, irrelevant since generate_signal is mocked)
for j, bar in enumerate(range(60, 70)):   # 60->69: ramp 100 -> 120 (past quick@107.5 and core@115)
    close_runner[bar] = 100 + (20 * (j+1) / 10)
for bar in range(70, 76):                 # 70->75: hold near 122-124 through the time-exit bar (75)
    close_runner[bar] = 122 + (bar - 70) * 0.4
for bar in range(76, N):
    close_runner[bar] = 124  # irrelevant, position fully closed by bar 75

# ── STOPPED_OUT: entry@bar60=100, stop=95 ─────────────────────────────────
# Price drops straight to 94 right after entry, before ANY tranche target —
# every tranche should close TOGETHER at the shared stop price (95), same bar.
close_stopped = flat(100, N)
for bar in range(60, 63):
    close_stopped[bar] = [98, 96, 94][bar - 60]
for bar in range(63, N):
    close_stopped[bar] = 94

def make_df(close_arr):
    close = pd.Series(close_arr, index=dates)
    df = pd.DataFrame({
        'open': close.values, 'high': close.values * 1.002, 'low': close.values * 0.998,
        'close': close.values, 'volume': np.full(N, 300000.0), 'datetime': dates,
    })
    return df

df_runner  = make_df(close_runner)
df_stopped = make_df(close_stopped)
nifty_df   = make_df(flat(20000, N))

def fake_hist(symbol, days=200, min_bars=50):
    return {'RUNNER_WIN': df_runner, 'STOPPED_OUT': df_stopped, '^NSEI': nifty_df}.get(symbol)

def fake_signal(self, df, symbol, fund, current_equity, market_regime='NEUTRAL', last_exit_bar=None):
    # Fire exactly one BUY, for either symbol, only at bar 60 (df length 61
    # means iloc[:i+1] has just reached index 60 in the walk-forward loop).
    if len(df) - 1 != 60:
        return 'HOLD', {}
    return 'BUY', {
        'entry_price': 100.0, 'stop_loss': 95.0, 'target_price': 115.0,
        'position_size': 20, 'entry_type': 'breakout', 'confidence': 0.8,
        'risk_reward_ratio': 3.0,
    }

def fake_fund(self, symbol, retry=2):
    return {'pe_ratio': 25, 'roe': 18, 'debt_to_equity': 0.5, 'revenue_growth': 15}

print("="*70)
print("TEST 1: RUNNER_WIN — tranches should close independently")
print("="*70)
bot = SwingTradingBot(send_emails=False, initial_equity=100000, max_open_trades=5, max_hold_days=15)
with patch.object(data_fetcher_free.DataFetcherFree, 'get_historical_data', side_effect=fake_hist), \
     patch.object(SwingTradingBot, 'get_fundamentals_safe', fake_fund), \
     patch('signal_generator.SignalGenerator.generate_signal', fake_signal):
    trades_df = bot.backtest_portfolio(['RUNNER_WIN'], days=140, use_alpha_engine=False, use_scaled_exits=True)

print(trades_df[['symbol','tranche','exit_reason','entry_price','exit_price','position_size','net_pnl','trade_group_id']].to_string() if trades_df is not None else "NO TRADES")

if trades_df is not None:
    check("exactly 3 tranche rows produced", len(trades_df) == 3, len(trades_df))
    check("labels are quick/core/runner", set(trades_df['tranche']) == {'quick','core','runner'}, set(trades_df['tranche']))
    check("all 3 share one trade_group_id", trades_df['trade_group_id'].nunique() == 1, trades_df['trade_group_id'].unique())
    check("sizes sum to original 20", trades_df['position_size'].sum() == 20, trades_df['position_size'].sum())
    quick = trades_df[trades_df['tranche']=='quick'].iloc[0]
    core  = trades_df[trades_df['tranche']=='core'].iloc[0]
    runner= trades_df[trades_df['tranche']=='runner'].iloc[0]
    check("quick exits at its own 1.5R target (107.5), not the full target", abs(quick['exit_price'] - 107.5) < 0.01, quick['exit_price'])
    check("quick exit_reason is Target Hit", quick['exit_reason'] == 'Target Hit', quick['exit_reason'])
    check("core exits at signal's original target (115)", abs(core['exit_price'] - 115.0) < 0.01, core['exit_price'])
    check("runner does NOT exit at 115 (its target is ~unreachable)", runner['exit_price'] > 115.0, runner['exit_price'])
    check("runner captured MORE than the fixed 115 target", runner['exit_price'] > core['exit_price'], (runner['exit_price'], core['exit_price']))
else:
    check("trades were produced at all", False)

print("\n" + "="*70)
print("TEST 2: STOPPED_OUT — all tranches must close TOGETHER at shared stop")
print("="*70)
bot2 = SwingTradingBot(send_emails=False, initial_equity=100000, max_open_trades=5, max_hold_days=15)
with patch.object(data_fetcher_free.DataFetcherFree, 'get_historical_data', side_effect=fake_hist), \
     patch.object(SwingTradingBot, 'get_fundamentals_safe', fake_fund), \
     patch('signal_generator.SignalGenerator.generate_signal', fake_signal):
    trades_df2 = bot2.backtest_portfolio(['STOPPED_OUT'], days=140, use_alpha_engine=False, use_scaled_exits=True)

print(trades_df2[['symbol','tranche','exit_reason','exit_price','position_size','exit_index']].to_string() if trades_df2 is not None else "NO TRADES")
if trades_df2 is not None:
    check("exactly 3 tranche rows (all closed on the loss)", len(trades_df2) == 3, len(trades_df2))
    check("all exit at the SAME stop price (95)", (trades_df2['exit_price'] == 95.0).all(), trades_df2['exit_price'].tolist())
    check("all exit on the SAME bar (closed together, not staggered)", trades_df2['exit_index'].nunique() == 1, trades_df2['exit_index'].tolist())
    check("all reason 'SL Hit'", (trades_df2['exit_reason'] == 'SL Hit').all(), trades_df2['exit_reason'].tolist())

print("\n" + "="*70)
print("TEST 3: use_scaled_exits=False — backward compatibility (old single-exit behavior)")
print("="*70)
bot3 = SwingTradingBot(send_emails=False, initial_equity=100000, max_open_trades=5, max_hold_days=15)
with patch.object(data_fetcher_free.DataFetcherFree, 'get_historical_data', side_effect=fake_hist), \
     patch.object(SwingTradingBot, 'get_fundamentals_safe', fake_fund), \
     patch('signal_generator.SignalGenerator.generate_signal', fake_signal):
    trades_df3 = bot3.backtest_portfolio(['RUNNER_WIN'], days=140, use_alpha_engine=False, use_scaled_exits=False)

print(trades_df3[['symbol','tranche','exit_reason','exit_price','position_size']].to_string() if trades_df3 is not None else "NO TRADES")
if trades_df3 is not None:
    check("exactly 1 row when untranched", len(trades_df3) == 1, len(trades_df3))
    check("full position size (20) in that one row", trades_df3.iloc[0]['position_size'] == 20, trades_df3.iloc[0]['position_size'])
    check("exits at the ORIGINAL fixed target (115), capped — this IS the comparison point vs tranched", abs(trades_df3.iloc[0]['exit_price'] - 115.0) < 0.01, trades_df3.iloc[0]['exit_price'])

print(f"\n{'='*70}\n{PASS} passed, {FAIL} failed\n{'='*70}")
sys.exit(1 if FAIL else 0)
