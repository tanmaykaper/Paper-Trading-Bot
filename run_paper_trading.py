# run_paper_trading.py
import logging
import pandas as pd
from datetime import datetime, time as dt_time
import time
from swing_trading_bot import SwingTradingBot
from paper_trading_manager import PaperTradingManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyPaperTradingRunner:
    def __init__(self, initial_equity=10000):
        self.bot = SwingTradingBot(send_emails=False, initial_equity=initial_equity, max_open_trades=5)
        self.paper_mgr = PaperTradingManager(initial_equity=initial_equity)
        self.stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'ASIANPAINT',
            'MARUTI', 'TATASTEEL', 'BAJFINANCE', 'KOTAKBANK', 'LT',
            'AXISBANK', 'TITAN', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND'
        ]
        logger.info("✓ Daily Paper Trading Runner initialized")

    def is_market_hours(self):
        """Check if market is in trading hours (9:15-15:30 IST)"""
        now = datetime.now().time()
        return dt_time(9, 15) <= now <= dt_time(15, 30)

    def is_market_closed(self):
        """Check if market has closed (>16:00 IST)"""
        now = datetime.now().time()
        return now >= dt_time(16, 0)

    def run_eod_process(self):
        """Run end-of-day paper trading process"""
        logger.info("\n" + "="*70)
        logger.info("📊 STARTING EOD PAPER TRADING PROCESS")
        logger.info("="*70)
        
        # Step 1: Update open trades with latest prices
        logger.info("\n[Step 1] Checking open trades for exits...")
        symbol_prices = self._fetch_latest_prices()
        trades_closed = self.paper_mgr.update_trades(symbol_prices)
        logger.info(f"Trades closed today: {trades_closed}")
        
        # Step 2: Generate new signals
        logger.info("\n[Step 2] Scanning for new trading signals...")
        new_trades = 0
        for symbol in self.stocks:
            try:
                df = self.bot.fetcher.get_historical_data(symbol, days=180)
                if df is None or len(df) < 60:
                    continue
                
                fundamentals = self.bot.fetcher.get_fundamentals(symbol)
                signal_type, signal_details = self.bot.signal_gen.generate_signal(
                    df, symbol, fundamentals, self.paper_mgr.current_equity
                )
                
                if signal_type == 'BUY':
                    opened = self.paper_mgr.open_trade(
                        symbol=symbol,
                        entry_price=signal_details['entry_price'],
                        stop_loss=signal_details['stop_loss'],
                        target_price=signal_details['target_price'],
                        position_size=signal_details['position_size'],
                        entry_type=signal_details['entry_type']
                    )
                    if opened:
                        new_trades += 1
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info(f"New trades opened: {new_trades}")
        
        # Step 3: Log daily equity
        logger.info("\n[Step 3] Logging daily equity...")
        self.paper_mgr.log_daily_equity()
        
        # Step 4: Print summary
        logger.info("\n[Step 4] Daily Summary:")
        summary = self.paper_mgr.get_summary()
        logger.info(f"  Total trades: {summary.get('total_trades', 0)}")
        logger.info(f"  Win rate: {summary.get('win_rate', 0)}%")
        logger.info(f"  Total P&L: ₹{summary.get('total_pnl', 0):,.2f}")
        logger.info(f"  Current equity: ₹{summary.get('current_equity', 10000):,.2f}")
        
        open_trades = self.paper_mgr.get_open_trades()
        logger.info(f"  Open trades: {len(open_trades)}")
        
        logger.info("\n" + "="*70)
        logger.info("✓ EOD PROCESS COMPLETE")
        logger.info("="*70 + "\n")

    def _fetch_latest_prices(self):
        """Fetch latest close prices for all stocks"""
        prices = {}
        for symbol in self.stocks:
            try:
                df = self.bot.fetcher.get_historical_data(symbol, days=5)
                if df is not None and len(df) > 0:
                    prices[symbol] = df.iloc[-1]['close']
            except:
                pass
        return prices

    def run_continuous(self, hours=8):
        """Run continuously during market hours"""
        logger.info(f"🚀 Starting continuous paper trading for {hours} hours...")
        
        start = datetime.now()
        while (datetime.now() - start).total_seconds() < hours * 3600:
            try:
                # Run EOD process once after market closes (4:00 PM IST)
                if self.is_market_closed():
                    logger.info("Market closed. Running EOD process...")
                    self.run_eod_process()
                    
                    # Sleep until next market open
                    logger.info("Waiting for next market open (9:15 AM)...")
                    time.sleep(60)  # Check every minute
                else:
                    logger.info(f"Market still open. Checking again in 5 minutes...")
                    time.sleep(300)  # Check every 5 minutes
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in continuous run: {e}")
                time.sleep(60)

if __name__ == "__main__":
    runner = DailyPaperTradingRunner(initial_equity=10000)
    
    # Option 1: Run continuous (keeps checking market)
    runner.run_continuous(hours=24)
    
    # Option 2: Run manual EOD once (for testing)
    # runner.run_eod_process()
