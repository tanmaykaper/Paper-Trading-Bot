# paper_trading_manager.py
import pandas as pd
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperTradingManager:
    """
    Manages paper trading via CSV files.
    All trades are stored in paper_trades.csv.
    Equity tracking in daily_equity.csv.
    """
    
    def __init__(self, initial_equity=10000, csv_path='paper_trades.csv', equity_csv_path='daily_equity.csv', commission_per_share=0.005):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.csv_path = csv_path
        self.equity_csv_path = equity_csv_path
        self.commission_per_share = commission_per_share
        self.trade_counter = 0
        
        self._ensure_csv_exists()
        self._load_state()
        logger.info(f"✓ Paper Trading Manager initialized | Capital: ₹{self.initial_equity:,}")

    def _ensure_csv_exists(self):
        """Create CSV if doesn't exist"""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'trade_id', 'symbol', 'side', 'entry_date', 'entry_price', 'stop_loss',
                'target_price', 'position_size', 'status', 'exit_date', 'exit_price',
                'exit_reason', 'gross_pnl', 'commission', 'net_pnl', 'hold_days', 'entry_type'
            ])
            df.to_csv(self.csv_path, index=False)
            logger.info(f"✓ Created {self.csv_path}")
        
        if not os.path.exists(self.equity_csv_path):
            df = pd.DataFrame(columns=[
                'date', 'equity', 'trades_open', 'trades_closed', 'daily_pnl', 'cumulative_pnl'
            ])
            df.to_csv(self.equity_csv_path, index=False)
            logger.info(f"✓ Created {self.equity_csv_path}")

    def _load_state(self):
        """Load existing trades and rebuild state"""
        try:
            trades_df = pd.read_csv(self.csv_path)
            if len(trades_df) > 0:
                self.trade_counter = int(trades_df['trade_id'].str.extract('(\d+)', expand=False).max() or 0)
                
                # Recalculate current equity based on closed trades
                closed_trades = trades_df[trades_df['status'] == 'CLOSED']
                if len(closed_trades) > 0:
                    total_closed_pnl = closed_trades['net_pnl'].sum()
                    self.current_equity = self.initial_equity + total_closed_pnl
                
                logger.info(f"✓ Loaded state: {len(trades_df)} total trades, Current equity: ₹{self.current_equity:,.2f}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def generate_trade_id(self, symbol):
        """Generate unique trade ID"""
        self.trade_counter += 1
        return f"{symbol}_{datetime.now().strftime('%Y%m%d')}_{self.trade_counter}"

    def open_trade(self, symbol, entry_price, stop_loss, target_price, position_size, entry_type):
        """Record a new open trade in CSV"""
        try:
            trades_df = pd.read_csv(self.csv_path)
            
            # Check if symbol already has open trade
            open_same_symbol = trades_df[(trades_df['symbol'] == symbol) & (trades_df['status'] == 'OPEN')]
            if len(open_same_symbol) > 0:
                logger.warning(f"⚠️ {symbol} already has open trade. Skipping.")
                return False
            
            trade_id = self.generate_trade_id(symbol)
            new_trade = pd.DataFrame([{
                'trade_id': trade_id,
                'symbol': symbol,
                'side': 'LONG',
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'target_price': round(target_price, 2),
                'position_size': int(position_size),
                'status': 'OPEN',
                'exit_date': '',
                'exit_price': '',
                'exit_reason': '',
                'gross_pnl': '',
                'commission': '',
                'net_pnl': '',
                'hold_days': '',
                'entry_type': entry_type
            }])
            
            trades_df = pd.concat([trades_df, new_trade], ignore_index=True)
            trades_df.to_csv(self.csv_path, index=False)
            logger.info(f"✓ Trade opened: {trade_id} | {symbol} @ ₹{entry_price:.2f}")
            return True
        except Exception as e:
            logger.error(f"Error opening trade: {e}")
            return False

    def update_trades(self, symbol_prices):
        """
        Check open trades for exit conditions based on latest prices.
        symbol_prices: dict like {'RELIANCE': 1545.50, 'TCS': 4500.00, ...}
        """
        try:
            trades_df = pd.read_csv(self.csv_path)
            open_trades = trades_df[trades_df['status'] == 'OPEN'].copy()
            
            if len(open_trades) == 0:
                return 0  # No open trades
            
            trades_closed = 0
            today = datetime.now()
            
            for idx, trade in open_trades.iterrows():
                symbol = trade['symbol']
                if symbol not in symbol_prices:
                    continue
                
                current_price = symbol_prices[symbol]
                entry_price = trade['entry_price']
                stop_loss = trade['stop_loss']
                target = trade['target_price']
                position_size = trade['position_size']
                entry_date = pd.to_datetime(trade['entry_date'])
                hold_days = (today - entry_date).days
                
                exit_triggered = False
                exit_reason = ''
                exit_price = 0
                
                # Check exit conditions
                if current_price <= stop_loss:
                    exit_triggered = True
                    exit_reason = 'SL Hit'
                    exit_price = stop_loss
                elif current_price >= target:
                    exit_triggered = True
                    exit_reason = 'Target Hit'
                    exit_price = target
                elif hold_days > 30:  # Max hold 30 days
                    exit_triggered = True
                    exit_reason = 'Time Exit'
                    exit_price = current_price
                
                if exit_triggered:
                    commission = position_size * self.commission_per_share * 2
                    gross_pnl = (exit_price - entry_price) * position_size
                    net_pnl = gross_pnl - commission
                    
                    # Update this trade in DataFrame
                    trades_df.loc[trades_df['trade_id'] == trade['trade_id'], 'status'] = 'CLOSED'
                    trades_df.loc[trades_df['trade_id'] == trade['trade_id'], 'exit_date'] = today.strftime('%Y-%m-%d')
                    trades_df.loc[trades_df['trade_id'] == trade['trade_id'], 'exit_price'] = round(exit_price, 2)
                    trades_df.loc[trades_df['trade_id'] == trade['trade_id'], 'exit_reason'] = exit_reason
                    trades_df.loc[trades_df['trade_id'] == trade['trade_id'], 'gross_pnl'] = round(gross_pnl, 2)
                    trades_df.loc[trades_df['trade_id'] == trade['trade_id'], 'commission'] = round(commission, 2)
                    trades_df.loc[trades_df['trade_id'] == trade['trade_id'], 'net_pnl'] = round(net_pnl, 2)
                    trades_df.loc[trades_df['trade_id'] == trade['trade_id'], 'hold_days'] = hold_days
                    
                    self.current_equity += net_pnl
                    trades_closed += 1
                    
                    logger.info(f"✓ Trade closed: {symbol} | {exit_reason} | P&L: ₹{net_pnl:.2f}")
            
            trades_df.to_csv(self.csv_path, index=False)
            return trades_closed
        except Exception as e:
            logger.error(f"Error updating trades: {e}")
            return 0

    def log_daily_equity(self):
        """Record daily equity snapshot"""
        try:
            trades_df = pd.read_csv(self.csv_path)
            trades_open = len(trades_df[trades_df['status'] == 'OPEN'])
            trades_closed = len(trades_df[trades_df['status'] == 'CLOSED'])
            
            closed_trades = trades_df[trades_df['status'] == 'CLOSED']
            daily_pnl = closed_trades[closed_trades['exit_date'] == datetime.now().strftime('%Y-%m-%d')]['net_pnl'].sum()
            cumulative_pnl = closed_trades['net_pnl'].sum()
            
            equity_df = pd.read_csv(self.equity_csv_path)
            new_entry = pd.DataFrame([{
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'equity': round(self.current_equity, 2),
                'trades_open': trades_open,
                'trades_closed': trades_closed,
                'daily_pnl': round(daily_pnl, 2),
                'cumulative_pnl': round(cumulative_pnl, 2)
            }])
            
            equity_df = pd.concat([equity_df, new_entry], ignore_index=True)
            equity_df.to_csv(self.equity_csv_path, index=False)
            
            logger.info(f"Daily equity logged | Current: ₹{self.current_equity:,.2f} | Open trades: {trades_open}")
        except Exception as e:
            logger.error(f"Error logging daily equity: {e}")

    def get_open_trades(self):
        """Return list of currently open trades"""
        try:
            trades_df = pd.read_csv(self.csv_path)
            return trades_df[trades_df['status'] == 'OPEN'].to_dict('records')
        except:
            return []

    def get_summary(self):
        """Return performance summary"""
        try:
            trades_df = pd.read_csv(self.csv_path)
            closed = trades_df[trades_df['status'] == 'CLOSED']
            
            if len(closed) == 0:
                return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
            
            wins = len(closed[closed['net_pnl'] > 0])
            total_pnl = closed['net_pnl'].sum()
            
            return {
                'total_trades': len(closed),
                'wins': wins,
                'losses': len(closed) - wins,
                'win_rate': round((wins / len(closed) * 100), 1),
                'total_pnl': round(total_pnl, 2),
                'avg_win': round(closed[closed['net_pnl'] > 0]['net_pnl'].mean(), 2) if wins > 0 else 0,
                'avg_loss': round(closed[closed['net_pnl'] < 0]['net_pnl'].mean(), 2) if (len(closed) - wins) > 0 else 0,
                'current_equity': round(self.current_equity, 2)
            }
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {}
