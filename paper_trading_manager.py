# paper_trading_manager.py  ── FIXED + ENHANCED v2
# ─────────────────────────────────────────────────────────────────────────────
# Fixes vs original:
#   1. entry_date column: handles both 'entry_datr' typo AND mixed date formats
#      (DD-MM-YYYY vs YYYY-MM-DD) that existed in the legacy CSV
#   2. Capital tracking: deployed capital is subtracted; new trades only open
#      if free_cash >= capital_needed
#   3. Unrealised P&L: get_summary() and print_open_positions() both compute
#      mark-to-market for every open trade using latest prices
#   4. Max open trade count enforced in open_trade() (not just in bot)
#   5. Hold-days calculation fixed: uses robust date parsing
#   6. Detailed EOD console output: capital used, free cash, each open position
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _parse_date(val):
    """Parse a date string that may be DD-MM-YYYY, YYYY-MM-DD, or a timestamp."""
    if pd.isna(val) or str(val).strip() in ('', 'nan', 'NaT'):
        return None
    s = str(val).strip()
    for fmt in ('%d-%m-%Y', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        return None


class PaperTradingManager:
    """
    Paper trading ledger backed by CSV files.

    Capital concepts:
      initial_equity   — starting cash
      realised_pnl     — sum of net_pnl on all CLOSED trades
      deployed_capital — sum of (entry_price * position_size) for OPEN trades
      free_cash        — initial_equity + realised_pnl - deployed_capital
      unrealised_pnl   — mark-to-market on OPEN trades (requires live prices)
      total_value      — free_cash + deployed_capital + unrealised_pnl
    """

    def __init__(self, initial_equity=10000, csv_path='paper_trades.csv',
                 equity_csv_path='daily_equity.csv', commission_per_share=0.005,
                 max_open_trades=5):
        self.initial_equity       = initial_equity
        self.commission_per_share = commission_per_share
        self.max_open_trades      = max_open_trades
        self.csv_path             = csv_path
        self.equity_csv_path      = equity_csv_path
        self.trade_counter        = 0

        # In-memory accumulators (recomputed from CSV on load)
        self.realised_pnl     = 0.0
        self.deployed_capital = 0.0

        self._ensure_csv_exists()
        self._reload_state()
        logger.info(f"✓ PaperTradingManager | Capital: ₹{self.initial_equity:,} | "
                    f"Max open: {self.max_open_trades}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_csv_exists(self):
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=[
                'trade_id', 'symbol', 'side', 'entry_date', 'entry_price',
                'stop_loss', 'target_price', 'position_size', 'status',
                'exit_date', 'exit_price', 'exit_reason',
                'gross_pnl', 'commission', 'net_pnl', 'hold_days', 'entry_type'
            ]).to_csv(self.csv_path, index=False)
            logger.info(f"✓ Created {self.csv_path}")

        if not os.path.exists(self.equity_csv_path):
            pd.DataFrame(columns=[
                'date', 'free_cash', 'deployed_capital', 'unrealised_pnl',
                'total_portfolio_value', 'trades_open', 'trades_closed',
                'realised_pnl', 'daily_realised_pnl'
            ]).to_csv(self.equity_csv_path, index=False)

    def _load_csv(self):
        """Load and normalise the trades CSV, fixing legacy column issues."""
        df = pd.read_csv(self.csv_path)

        # Fix legacy 'entry_datr' typo
        if 'entry_datr' in df.columns and 'entry_date' not in df.columns:
            df = df.rename(columns={'entry_datr': 'entry_date'})
        elif 'entry_datr' in df.columns and 'entry_date' in df.columns:
            mask = df['entry_date'].isna() | (df['entry_date'].astype(str).str.strip() == '')
            df.loc[mask, 'entry_date'] = df.loc[mask, 'entry_datr']
            df = df.drop(columns=['entry_datr'])

        for col in ['exit_date', 'exit_price', 'exit_reason',
                    'gross_pnl', 'commission', 'net_pnl', 'hold_days']:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def _save_csv(self, df):
        if 'entry_datr' in df.columns:
            df = df.drop(columns=['entry_datr'])
        df.to_csv(self.csv_path, index=False)

    def _reload_state(self):
        """Recompute realised_pnl, deployed_capital, trade_counter from CSV."""
        try:
            df = self._load_csv()
            if len(df) == 0:
                return

            nums = df['trade_id'].str.extract(r'(\d+)$', expand=False).dropna()
            self.trade_counter = int(nums.astype(int).max()) if len(nums) else 0

            closed = df[df['status'] == 'CLOSED'].copy()
            closed['net_pnl'] = pd.to_numeric(closed['net_pnl'], errors='coerce').fillna(0)
            self.realised_pnl = closed['net_pnl'].sum()

            open_t = df[df['status'] == 'OPEN'].copy()
            open_t['entry_price']   = pd.to_numeric(open_t['entry_price'],   errors='coerce').fillna(0)
            open_t['position_size'] = pd.to_numeric(open_t['position_size'], errors='coerce').fillna(0)
            self.deployed_capital   = (open_t['entry_price'] * open_t['position_size']).sum()

            logger.info(
                f"✓ State loaded | Open: {len(open_t)} | Closed: {len(closed)} | "
                f"Realised P&L: ₹{self.realised_pnl:,.2f} | "
                f"Deployed: ₹{self.deployed_capital:,.2f} | "
                f"Free cash: ₹{self.free_cash:,.2f}"
            )
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    # ── Capital properties ────────────────────────────────────────────────────

    @property
    def free_cash(self):
        """Uninvested cash available for new trades."""
        return self.initial_equity + self.realised_pnl - self.deployed_capital

    @property
    def current_equity(self):
        return self.free_cash   # backward compat alias

    def get_capital_snapshot(self, latest_prices=None):
        df     = self._load_csv()
        open_t = df[df['status'] == 'OPEN'].copy()
        closed = df[df['status'] == 'CLOSED'].copy()

        open_t['entry_price']   = pd.to_numeric(open_t['entry_price'],   errors='coerce').fillna(0)
        open_t['position_size'] = pd.to_numeric(open_t['position_size'], errors='coerce').fillna(0)
        closed['net_pnl']       = pd.to_numeric(closed['net_pnl'],       errors='coerce').fillna(0)

        deployed   = (open_t['entry_price'] * open_t['position_size']).sum()
        realised   = closed['net_pnl'].sum()
        free_cash  = self.initial_equity + realised - deployed

        unrealised = 0.0
        if latest_prices:
            for _, row in open_t.iterrows():
                sym = row['symbol']
                if sym in latest_prices:
                    unrealised += (float(latest_prices[sym]) - row['entry_price']) * row['position_size']

        total_value = free_cash + deployed + unrealised

        return {
            'initial_equity':        self.initial_equity,
            'realised_pnl':          round(realised,    2),
            'deployed_capital':      round(deployed,    2),
            'free_cash':             round(free_cash,   2),
            'unrealised_pnl':        round(unrealised,  2),
            'total_portfolio_value': round(total_value, 2),
            'open_trades':           len(open_t),
            'closed_trades':         len(closed),
        }

    # ── Trade operations ──────────────────────────────────────────────────────

    def generate_trade_id(self, symbol):
        self.trade_counter += 1
        return f"{symbol}_{datetime.now().strftime('%Y%m%d')}_{self.trade_counter}"

    def open_trade(self, symbol, entry_price, stop_loss, target_price,
                   position_size, entry_type):
        """
        Open a new paper trade with capital and slot checks.
        Automatically reduces position size to fit available free cash.
        """
        try:
            df          = self._load_csv()
            open_trades = df[df['status'] == 'OPEN']

            if symbol in open_trades['symbol'].values:
                logger.warning(f"⚠️ {symbol} already has open trade — skipping")
                return False

            if len(open_trades) >= self.max_open_trades:
                logger.warning(f"⚠️ Max {self.max_open_trades} open trades reached — skipping {symbol}")
                return False

            capital_needed = entry_price * position_size
            if capital_needed > self.free_cash:
                affordable = int(self.free_cash / entry_price)
                if affordable < 1:
                    logger.warning(
                        f"⚠️ Insufficient cash for {symbol} "
                        f"(need ₹{capital_needed:,.0f}, have ₹{self.free_cash:,.0f})"
                    )
                    return False
                logger.info(f"ℹ️ {symbol}: adjusted position {position_size}→{affordable} to fit cash")
                position_size  = affordable
                capital_needed = entry_price * position_size

            trade_id  = self.generate_trade_id(symbol)
            new_trade = pd.DataFrame([{
                'trade_id':      trade_id,
                'symbol':        symbol,
                'side':          'LONG',
                'entry_date':    datetime.now().strftime('%Y-%m-%d'),
                'entry_price':   round(entry_price,  2),
                'stop_loss':     round(stop_loss,    2),
                'target_price':  round(target_price, 2),
                'position_size': int(position_size),
                'status':        'OPEN',
                'exit_date':     '',
                'exit_price':    '',
                'exit_reason':   '',
                'gross_pnl':     '',
                'commission':    '',
                'net_pnl':       '',
                'hold_days':     '',
                'entry_type':    entry_type,
            }])

            df = pd.concat([df, new_trade], ignore_index=True)
            self._save_csv(df)
            self.deployed_capital += capital_needed

            logger.info(
                f"✅ Trade OPENED: {symbol} @ ₹{entry_price:.2f} | "
                f"Size: {position_size} | Deployed: ₹{capital_needed:,.0f} | "
                f"Free cash left: ₹{self.free_cash:,.0f}"
            )
            return True

        except Exception as e:
            logger.error(f"Error opening trade for {symbol}: {e}")
            return False

    def update_trades(self, symbol_prices, max_hold_days=15):
        """
        Mark-to-market check: close any trade that hit SL, target, or time limit.

        Args:
            symbol_prices: dict {symbol: latest_close_price}
            max_hold_days: time-exit threshold in calendar days

        Returns:
            Number of trades closed this run
        """
        try:
            df          = self._load_csv()
            open_trades = df[df['status'] == 'OPEN'].copy()

            if len(open_trades) == 0:
                logger.info("  No open trades to check")
                return 0

            trades_closed = 0
            today         = datetime.now()

            for _, trade in open_trades.iterrows():
                symbol = trade['symbol']
                if symbol not in symbol_prices:
                    logger.debug(f"  No price for {symbol} — skipping exit check")
                    continue

                current_price = float(symbol_prices[symbol])
                entry_price   = float(trade['entry_price'])
                stop_loss     = float(trade['stop_loss'])
                target        = float(trade['target_price'])
                position_size = int(trade['position_size'])

                entry_dt  = _parse_date(trade['entry_date'])
                hold_days = (today - entry_dt).days if entry_dt else 999

                exit_triggered = False
                exit_reason    = ''
                exit_price     = 0.0

                if current_price <= stop_loss:
                    exit_triggered, exit_reason, exit_price = True, 'SL Hit',     stop_loss
                elif current_price >= target:
                    exit_triggered, exit_reason, exit_price = True, 'Target Hit', target
                elif hold_days > max_hold_days:
                    exit_triggered, exit_reason, exit_price = True, 'Time Exit',  current_price

                if exit_triggered:
                    commission = position_size * self.commission_per_share * 2
                    gross_pnl  = (exit_price - entry_price) * position_size
                    net_pnl    = gross_pnl - commission

                    mask = df['trade_id'] == trade['trade_id']
                    df.loc[mask, 'status']     = 'CLOSED'
                    df.loc[mask, 'exit_date']  = today.strftime('%Y-%m-%d')
                    df.loc[mask, 'exit_price'] = round(exit_price,  2)
                    df.loc[mask, 'exit_reason']= exit_reason
                    df.loc[mask, 'gross_pnl']  = round(gross_pnl,   2)
                    df.loc[mask, 'commission'] = round(commission,   2)
                    df.loc[mask, 'net_pnl']    = round(net_pnl,      2)
                    df.loc[mask, 'hold_days']  = hold_days

                    self.realised_pnl     += net_pnl
                    self.deployed_capital  = max(0.0, self.deployed_capital
                                                 - entry_price * position_size)
                    trades_closed         += 1

                    icon = '🟢' if net_pnl >= 0 else '🔴'
                    logger.info(
                        f"  {icon} CLOSED {symbol} | {exit_reason} | "
                        f"₹{entry_price:.2f}→₹{exit_price:.2f} | "
                        f"P&L: ₹{net_pnl:+,.2f} | Hold: {hold_days}d"
                    )

            self._save_csv(df)
            return trades_closed

        except Exception as e:
            logger.error(f"Error updating trades: {e}")
            return 0

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_open_positions(self, latest_prices=None):
        """
        Print a formatted table of all open positions with live P&L.
        Call with latest_prices dict for unrealised figures.
        """
        df          = self._load_csv()
        open_trades = df[df['status'] == 'OPEN'].copy()

        if len(open_trades) == 0:
            logger.info("  No open positions")
            return

        for col in ['entry_price', 'position_size', 'stop_loss', 'target_price']:
            open_trades[col] = pd.to_numeric(open_trades[col], errors='coerce')

        today = datetime.now()

        header = (f"\n  {'Symbol':<13} {'Entry':>8} {'CMP':>8} {'Qty':>5} "
                  f"{'Capital':>9} {'Unreal P&L':>12} {'%':>7} "
                  f"{'Hold':>5} {'SL':>8} {'Target':>8}  Status")
        logger.info(header)
        logger.info("  " + "─" * 110)

        total_unrealised = 0.0
        total_deployed   = 0.0

        for _, t in open_trades.iterrows():
            ep  = float(t['entry_price'])
            ps  = int(t['position_size'])
            sl  = float(t['stop_loss'])
            tp  = float(t['target_price'])
            sym = t['symbol']
            deployed = ep * ps
            total_deployed += deployed

            entry_dt  = _parse_date(t['entry_date'])
            hold_days = (today - entry_dt).days if entry_dt else '?'

            cmp = float(latest_prices[sym]) if (latest_prices and sym in latest_prices) else None

            if cmp is not None:
                unreal         = (cmp - ep) * ps
                pct            = (cmp - ep) / ep * 100
                total_unrealised += unreal
                pnl_str        = f"₹{unreal:+,.0f}"
                pct_str        = f"{pct:+.1f}%"
                cmp_str        = f"₹{cmp:.2f}"
                sl_dist_pct    = (cmp - sl) / ep * 100
                tp_dist_pct    = (tp  - cmp) / ep * 100
                if sl_dist_pct < 1.5:
                    status = '⚠️ NEAR SL'
                elif tp_dist_pct < 2.0:
                    status = '🎯 NEAR TP'
                elif unreal > 0:
                    status = '🟢 PROFIT'
                else:
                    status = '🔴 LOSS'
            else:
                cmp_str  = '---'
                pnl_str  = '---'
                pct_str  = '---'
                status   = '❓'

            logger.info(
                f"  {sym:<13} ₹{ep:>7.2f} {cmp_str:>8} {ps:>5} "
                f"₹{deployed:>8,.0f} {pnl_str:>12} {pct_str:>7} "
                f"{str(hold_days):>4}d ₹{sl:>7.2f} ₹{tp:>7.2f}  {status}"
            )

        logger.info("  " + "─" * 110)
        logger.info(f"  {'TOTAL DEPLOYED':>45}  ₹{total_deployed:>10,.0f}")
        if latest_prices:
            logger.info(f"  {'TOTAL UNREALISED P&L':>45}  ₹{total_unrealised:>+10,.0f}")

    def get_summary(self, latest_prices=None):
        """
        Full performance summary — realised + unrealised + capital breakdown.
        """
        try:
            df     = self._load_csv()
            closed = df[df['status'] == 'CLOSED'].copy()
            open_t = df[df['status'] == 'OPEN'].copy()

            closed['net_pnl']       = pd.to_numeric(closed['net_pnl'],       errors='coerce').fillna(0)
            open_t['entry_price']   = pd.to_numeric(open_t['entry_price'],   errors='coerce').fillna(0)
            open_t['position_size'] = pd.to_numeric(open_t['position_size'], errors='coerce').fillna(0)

            n_closed  = len(closed)
            wins      = int((closed['net_pnl'] > 0).sum())
            losses    = n_closed - wins
            realised  = closed['net_pnl'].sum()
            win_rate  = round(wins / n_closed * 100, 1) if n_closed else 0.0
            avg_win   = closed.loc[closed['net_pnl'] > 0, 'net_pnl'].mean() if wins   else 0.0
            avg_loss  = closed.loc[closed['net_pnl'] < 0, 'net_pnl'].mean() if losses else 0.0

            deployed   = (open_t['entry_price'] * open_t['position_size']).sum()
            unrealised = 0.0
            if latest_prices:
                for _, row in open_t.iterrows():
                    sym = row['symbol']
                    if sym in latest_prices:
                        unrealised += (float(latest_prices[sym]) - row['entry_price']) * row['position_size']

            free_cash   = self.initial_equity + realised - deployed
            total_pnl   = realised + unrealised
            total_value = free_cash + deployed + unrealised

            return {
                'initial_equity':        self.initial_equity,
                'free_cash':             round(free_cash,   2),
                'deployed_capital':      round(deployed,    2),
                'unrealised_pnl':        round(unrealised,  2),
                'realised_pnl':          round(realised,    2),
                'total_pnl':             round(total_pnl,   2),
                'total_portfolio_value': round(total_value, 2),
                'open_trades':           len(open_t),
                'closed_trades':         n_closed,
                'wins':                  wins,
                'losses':                losses,
                'win_rate':              win_rate,
                'avg_win':               round(avg_win,  2),
                'avg_loss':              round(avg_loss, 2),
            }
        except Exception as e:
            logger.error(f"Error in get_summary: {e}")
            return {}

    def log_daily_equity(self, latest_prices=None):
        """Append a portfolio snapshot row to the equity CSV."""
        try:
            snap      = self.get_capital_snapshot(latest_prices)
            df        = pd.read_csv(self.equity_csv_path) if os.path.exists(self.equity_csv_path) \
                        else pd.DataFrame()
            trades_df = self._load_csv()
            closed    = trades_df[trades_df['status'] == 'CLOSED'].copy()
            closed['net_pnl']  = pd.to_numeric(closed['net_pnl'], errors='coerce').fillna(0)
            today_str          = datetime.now().strftime('%Y-%m-%d')
            daily_realised     = closed[closed['exit_date'] == today_str]['net_pnl'].sum()

            new_row = pd.DataFrame([{
                'date':                  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'free_cash':             snap['free_cash'],
                'deployed_capital':      snap['deployed_capital'],
                'unrealised_pnl':        snap['unrealised_pnl'],
                'total_portfolio_value': snap['total_portfolio_value'],
                'trades_open':           snap['open_trades'],
                'trades_closed':         snap['closed_trades'],
                'realised_pnl':          snap['realised_pnl'],
                'daily_realised_pnl':    round(daily_realised, 2),
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.equity_csv_path, index=False)
        except Exception as e:
            logger.error(f"Error logging daily equity: {e}")

        
    def get_open_trades(self):
        df = self._load_csv()
        return df[df['status'] == 'OPEN'].to_dict('records')
