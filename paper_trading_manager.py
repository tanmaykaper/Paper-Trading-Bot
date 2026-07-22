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
from trailing_stop import compute_trailing_stop

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
                'gross_pnl', 'commission', 'net_pnl', 'hold_days', 'entry_type',
                'confidence', 'risk_reward_ratio', 'alpha_score', 'alpha_tier',
                'initial_stop_loss',
            ]).to_csv(self.csv_path, index=False)
            logger.info(f"✓ Created {self.csv_path}")

        _EQUITY_COLS = [
            'date', 'free_cash', 'deployed_capital', 'unrealised_pnl',
            'total_portfolio_value', 'trades_open', 'trades_closed',
            'realised_pnl', 'daily_realised_pnl'
        ]

        if not os.path.exists(self.equity_csv_path):
            pd.DataFrame(columns=_EQUITY_COLS).to_csv(self.equity_csv_path, index=False)
        else:
            # Migrate legacy schemas (e.g. an older version of this file wrote
            # different column names like 'equity'/'daily_pnl'/'cumulative_pnl').
            # Without this, concat-by-column-name silently leaves the current
            # schema's columns blank on every new row — exactly what happened
            # to unrealised_pnl/total_portfolio_value in this bot's history.
            try:
                existing = pd.read_csv(self.equity_csv_path)
                missing_cols = [c for c in _EQUITY_COLS if c not in existing.columns]
                if missing_cols:
                    for c in missing_cols:
                        existing[c] = np.nan
                    # keep legacy columns (don't destroy history) but ensure
                    # every current-schema column exists so writes never drift
                    existing.to_csv(self.equity_csv_path, index=False)
                    logger.info(f"✓ Migrated {self.equity_csv_path} schema — added columns: {missing_cols}")
            except Exception as e:
                logger.error(f"Could not migrate equity CSV schema: {e}")

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
                    'gross_pnl', 'commission', 'net_pnl', 'hold_days',
                    'confidence', 'risk_reward_ratio', 'alpha_score', 'alpha_tier',
                    'initial_stop_loss']:
            if col not in df.columns:
                df[col] = np.nan

        # ── Critical dtype fix ──────────────────────────────────────────────
        # When every OPEN row has an empty exit_date/exit_reason, pandas'
        # read_csv infers those columns as float64 (all-NaN). Newer pandas
        # (3.x) then REFUSES to assign a string into a float64 column via
        # .loc[...] and raises TypeError. That crash was happening on every
        # single attempt to close a trade — silently swallowed by the
        # try/except in update_trades() — which is why trades opened months
        # ago never closed even when prices were fetched successfully.
        # Forcing these to 'object' dtype makes string assignment always safe,
        # regardless of pandas version or how many rows are currently empty.
        for col in ['exit_date', 'exit_reason', 'status', 'symbol', 'side', 'entry_type', 'alpha_tier']:
            if col in df.columns:
                df[col] = df[col].astype(object)

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
                   position_size, entry_type, confidence=None, risk_reward_ratio=None,
                   alpha_score=None, alpha_tier=None):
        """
        Open a new paper trade with capital and slot checks.
        Automatically reduces position size to fit available free cash.

        confidence / risk_reward_ratio (optional): recorded from the signal
        that generated this trade, so open positions can later be scored
        fairly against new candidate signals (used by position-replacement
        logic in the runner) instead of guessing at their quality.

        alpha_score / alpha_tier (optional): recorded from alpha_engine's
        CompositeAlphaScore for this specific entry, if the runner has that
        layer wired in. When present, this is what replacement scoring
        actually uses (see run_paper_trading.py's _composite_score) —
        confidence/risk_reward_ratio remain as the fallback for trades
        opened before the alpha engine was integrated, or on any run where
        alpha scoring itself couldn't run (e.g. insufficient index data).
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
                'initial_stop_loss': round(stop_loss, 2),   # immutable — see trailing_stop.py for why
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
                'confidence':        confidence if confidence is not None else '',
                'risk_reward_ratio': round(risk_reward_ratio, 2) if risk_reward_ratio is not None else '',
                'alpha_score':       round(alpha_score, 1) if alpha_score is not None else '',
                'alpha_tier':        alpha_tier if alpha_tier is not None else '',
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

    def apply_trailing_stops(self, latest_prices):
        """
        Ratchets stop_loss upward for every OPEN position using
        trailing_stop.compute_trailing_stop() — the SAME function
        swing_trading_bot.py's backtest uses, so live trading and
        backtesting run identical trailing-stop economics rather than two
        implementations that could quietly drift apart (see trailing_stop.py
        for the bug that motivated sharing one implementation instead of two).

        Must be called BEFORE update_trades() in the daily run, so that if a
        position closes today, it closes against its current (possibly
        just-ratcheted) stop, not a stale one.

        Positions with no entry in latest_prices are left untouched for this
        run (same "missing price this run" handling as everywhere else in
        this codebase — skipped, not defaulted to some assumed price).

        Returns the number of positions whose stop actually moved, for
        logging.
        """
        try:
            df    = self._load_csv()
            open_mask = df['status'] == 'OPEN'
            if not open_mask.any():
                return 0

            moved = 0
            for idx in df[open_mask].index:
                symbol = df.at[idx, 'symbol']
                if symbol not in latest_prices:
                    continue

                entry_price = float(df.at[idx, 'entry_price'])
                current_sl  = float(df.at[idx, 'stop_loss'])
                initial_sl_raw = df.at[idx, 'initial_stop_loss']
                # Graceful migration: a position opened before this column
                # existed has no recorded initial_stop_loss. Bootstrap it
                # from whatever the current stop_loss is right now — not
                # dangerous, just means trailing starts fresh from today
                # for that one position instead of from its true original.
                initial_sl = float(initial_sl_raw) if pd.notna(initial_sl_raw) and initial_sl_raw != '' else current_sl

                new_sl = compute_trailing_stop(entry_price, initial_sl, current_sl, float(latest_prices[symbol]))
                if new_sl > current_sl:
                    df.at[idx, 'stop_loss'] = new_sl
                    if pd.isna(initial_sl_raw) or initial_sl_raw == '':
                        df.at[idx, 'initial_stop_loss'] = initial_sl   # backfill for next time
                    moved += 1
                    logger.info(f"  📈 {symbol}: trailing stop raised ₹{current_sl:.2f} → ₹{new_sl:.2f}")

            if moved:
                self._save_csv(df)
            return moved

        except Exception as e:
            logger.error(f"Error applying trailing stops: {e}")
            return 0

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

    def close_position(self, trade_id, exit_price, exit_reason):
        """
        Close one specific OPEN trade by id at a given price. Used for
        position replacement (swapping a weak/stale holding for a
        materially better new signal when all slots are full) rather than
        only closing on SL/target/time-exit.
        """
        try:
            df   = self._load_csv()
            mask = (df['trade_id'] == trade_id) & (df['status'] == 'OPEN')
            if not mask.any():
                logger.warning(f"⚠️ close_position: no OPEN trade with id {trade_id}")
                return False

            row           = df.loc[mask].iloc[0]
            entry_price   = float(row['entry_price'])
            position_size = int(row['position_size'])
            entry_dt      = _parse_date(row['entry_date'])
            hold_days     = (datetime.now() - entry_dt).days if entry_dt else 0

            commission = position_size * self.commission_per_share * 2
            gross_pnl  = (exit_price - entry_price) * position_size
            net_pnl    = gross_pnl - commission

            df.loc[mask, 'status']      = 'CLOSED'
            df.loc[mask, 'exit_date']   = datetime.now().strftime('%Y-%m-%d')
            df.loc[mask, 'exit_price']  = round(exit_price, 2)
            df.loc[mask, 'exit_reason'] = exit_reason
            df.loc[mask, 'gross_pnl']   = round(gross_pnl, 2)
            df.loc[mask, 'commission']  = round(commission, 2)
            df.loc[mask, 'net_pnl']     = round(net_pnl, 2)
            df.loc[mask, 'hold_days']   = hold_days

            self._save_csv(df)
            self.realised_pnl     += net_pnl
            self.deployed_capital  = max(0.0, self.deployed_capital - entry_price * position_size)

            icon = '🟢' if net_pnl >= 0 else '🔴'
            logger.info(
                f"  {icon} CLOSED {row['symbol']} (replaced) | {exit_reason} | "
                f"₹{entry_price:.2f}→₹{exit_price:.2f} | P&L: ₹{net_pnl:+,.2f}"
            )
            return True
        except Exception as e:
            logger.error(f"Error closing position {trade_id}: {e}")
            return False

    def get_aggregate_open_risk(self):
        """
        Total capital currently 'at stake' across all OPEN positions — i.e.
        the sum of (entry_price - stop_loss) * position_size for each open
        trade, which is what you'd actually lose if every stop-loss hit at
        once. This is the basis for portfolio-level risk budgeting: sizing
        each trade individually as 2.5% of equity doesn't bound the total
        book risk if 5+ trades are open simultaneously, especially when
        several of them may be correlated (same sector/regime).
        """
        try:
            df     = self._load_csv()
            open_t = df[df['status'] == 'OPEN'].copy()
            if len(open_t) == 0:
                return 0.0
            open_t['entry_price']   = pd.to_numeric(open_t['entry_price'],   errors='coerce').fillna(0)
            open_t['stop_loss']     = pd.to_numeric(open_t['stop_loss'],     errors='coerce').fillna(0)
            open_t['position_size'] = pd.to_numeric(open_t['position_size'], errors='coerce').fillna(0)
            risk = ((open_t['entry_price'] - open_t['stop_loss']) * open_t['position_size']).sum()
            return max(0.0, float(risk))
        except Exception as e:
            logger.error(f"Error in get_aggregate_open_risk: {e}")
            return 0.0

    def get_open_positions_by_sector(self, sector_map):
        """
        Returns {sector: [symbols]} for currently OPEN positions, using the
        same SECTOR_MAP the rest of the codebase already defines. Symbols
        not present in sector_map are grouped under their own symbol (so an
        unmapped name never silently counts against some other sector's cap).
        """
        open_trades = self.get_open_trades()
        by_sector = {}
        for t in open_trades:
            sector = sector_map.get(t['symbol'], t['symbol'])
            by_sector.setdefault(sector, []).append(t['symbol'])
        return by_sector

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

    def get_closed_trades(self):
        """
        Returns CLOSED trades as a DataFrame (not records) with net_pnl
        coerced numeric — the shape alpha_engine.AdaptiveWeightCalibrator
        expects. Separate from get_open_trades()'s dict-records return
        because the calibrator does pandas groupby/aggregation over this,
        not per-row iteration.
        """
        df = self._load_csv()
        closed = df[df['status'] == 'CLOSED'].copy()
        closed['net_pnl'] = pd.to_numeric(closed['net_pnl'], errors='coerce')
        return closed

    # ── Maintenance ───────────────────────────────────────────────────────────

    def force_close_stale(self, latest_prices, max_hold_days=15, reason_suffix=""):
        """
        Immediately close any OPEN trade older than max_hold_days, using the
        best price available (live price if we have it, else the entry price
        as a last-resort fallback so the position isn't left open forever).

        This exists as a manual unstick tool: if price fetching silently fails
        for weeks/months (as it did before the retry/bulk-fetch fix), positions
        can sit open far past their intended hold window with no error raised.
        Run this once after deploying the fetch fixes to clear the backlog,
        rather than waiting for the ordinary daily update_trades() cycle.

        Returns list of dicts describing what was closed.
        """
        try:
            df          = self._load_csv()
            open_trades = df[df['status'] == 'OPEN'].copy()
            today       = datetime.now()
            closed_info = []

            for _, trade in open_trades.iterrows():
                symbol       = trade['symbol']
                entry_dt     = _parse_date(trade['entry_date'])
                hold_days    = (today - entry_dt).days if entry_dt else 9999

                if hold_days <= max_hold_days:
                    continue

                entry_price   = float(trade['entry_price'])
                position_size = int(trade['position_size'])
                stop_loss     = float(trade['stop_loss'])
                target        = float(trade['target_price'])

                if symbol in latest_prices:
                    exit_price = float(latest_prices[symbol])
                    reason     = f'Time Exit (maintenance, {hold_days}d held){reason_suffix}'
                else:
                    # No live price available even now — fall back to entry
                    # price (net_pnl = -commission only) so the slot frees up
                    # rather than staying stuck indefinitely. Flag it clearly.
                    exit_price = entry_price
                    reason     = f'Time Exit (maintenance, NO PRICE — closed flat, {hold_days}d held){reason_suffix}'
                    logger.warning(f"⚠️ {symbol}: no live price available — closing flat at entry price")

                # Respect SL/target if the live price already breached them
                if exit_price <= stop_loss:
                    exit_price, reason = stop_loss, f'SL Hit (maintenance, {hold_days}d held){reason_suffix}'
                elif exit_price >= target:
                    exit_price, reason = target, f'Target Hit (maintenance, {hold_days}d held){reason_suffix}'

                commission = position_size * self.commission_per_share * 2
                gross_pnl  = (exit_price - entry_price) * position_size
                net_pnl    = gross_pnl - commission

                mask = df['trade_id'] == trade['trade_id']
                df.loc[mask, 'status']      = 'CLOSED'
                df.loc[mask, 'exit_date']   = today.strftime('%Y-%m-%d')
                df.loc[mask, 'exit_price']  = round(exit_price, 2)
                df.loc[mask, 'exit_reason'] = reason
                df.loc[mask, 'gross_pnl']   = round(gross_pnl, 2)
                df.loc[mask, 'commission']  = round(commission, 2)
                df.loc[mask, 'net_pnl']     = round(net_pnl, 2)
                df.loc[mask, 'hold_days']   = hold_days

                self.realised_pnl     += net_pnl
                self.deployed_capital  = max(0.0, self.deployed_capital - entry_price * position_size)

                closed_info.append({
                    'symbol': symbol, 'hold_days': hold_days,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'net_pnl': round(net_pnl, 2), 'reason': reason,
                })
                icon = '🟢' if net_pnl >= 0 else '🔴'
                logger.info(
                    f"  {icon} MAINTENANCE CLOSE {symbol} | {reason} | "
                    f"₹{entry_price:.2f}→₹{exit_price:.2f} | P&L: ₹{net_pnl:+,.2f}"
                )

            if closed_info:
                self._save_csv(df)

            return closed_info

        except Exception as e:
            logger.error(f"Error in force_close_stale: {e}")
            return []

    def price_fetch_health_check(self, latest_prices):
        """
        Loud, explicit report on whether exit-checks actually had the data
        they needed this run. Call this every run and treat a bad result as
        a real incident, not background noise — the original 3-month-silent
        failure happened precisely because nothing surfaced this.
        """
        open_trades   = self.get_open_trades()
        held_symbols  = [t['symbol'] for t in open_trades]
        missing       = [s for s in held_symbols if s not in latest_prices]

        report = {
            'held_positions':   len(held_symbols),
            'prices_resolved':  len(held_symbols) - len(missing),
            'missing_symbols':  missing,
            'healthy':          len(missing) == 0,
        }

        if not report['healthy']:
            logger.error(
                f"🚨 HEALTH CHECK FAILED: {len(missing)}/{len(held_symbols)} open positions "
                f"have NO live price this run: {missing}. Exit checks for these were SKIPPED. "
                f"If this repeats for multiple consecutive runs, price fetching is broken "
                f"and positions will silently stay open indefinitely."
            )
        else:
            logger.info(f"✓ HEALTH CHECK OK: all {len(held_symbols)} open positions priced successfully")

        return report
