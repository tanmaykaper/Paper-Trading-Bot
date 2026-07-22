# swing_trading_bot.py  ── HIGH-RISK / HIGH-FREQUENCY VERSION v3
# ─────────────────────────────────────────────────────────────────────────────
# Fixes vs previous:
#   - TATAMOTORS → TATAMOTOR in SECTOR_MAP (yfinance symbol change)
#   - _get_market_regime passes '^NSEI' which DataFetcherFree now handles
#     correctly (no .NS suffix for index symbols)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import logging
import numpy as np
from datetime import datetime
from data_fetcher_free import DataFetcherFree
from technical_indicators import TechnicalIndicators
from fundamental_screener import FundamentalScreener
from signal_generator import SignalGenerator, RISK_PROFILE
from notification_handler import NotificationHandler
from alpha_engine import CompositeAlphaScore
from trailing_stop import compute_trailing_stop
import os
from dotenv import load_dotenv
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

SECTOR_MAP = {
    # IT
    'TCS':'IT','INFY':'IT','WIPRO':'IT','HCLTECH':'IT','TECHM':'IT','LTIM':'IT',
    'MPHASIS':'IT','PERSISTENT':'IT','COFORGE':'IT',
    # Banks
    'HDFCBANK':'BANK','ICICIBANK':'BANK','SBIN':'BANK','KOTAKBANK':'BANK',
    'AXISBANK':'BANK','INDUSINDBK':'BANK','FEDERALBNK':'BANK','BANDHANBNK':'BANK',
    # Energy
    'RELIANCE':'ENERGY','ONGC':'ENERGY','BPCL':'ENERGY','IOC':'ENERGY','GAIL':'ENERGY',
    # FMCG
    'HINDUNILVR':'FMCG','ITC':'FMCG','NESTLEIND':'FMCG','DABUR':'FMCG',
    'MARICO':'FMCG','GODREJCP':'FMCG',
    # Auto
    'MARUTI':'AUTO','TATAMOTOR':'AUTO','BAJAJ-AUTO':'AUTO','EICHERMOT':'AUTO',
    'M&M':'AUTO','HEROMOTOCO':'AUTO',
    # Metals
    'TATASTEEL':'METAL','JSWSTEEL':'METAL','HINDALCO':'METAL','SAIL':'METAL',
    # Paint / Consumer
    'ASIANPAINT':'PAINT','BERGERPAINTS':'PAINT',
    'TITAN':'CONSUMER','PIDILITIND':'CONSUMER','VOLTAS':'CONSUMER',
    # Infra / Capital Goods
    'LT':'INFRA','ADANIPORTS':'INFRA','ABB':'CAPGOODS','SIEMENS':'CAPGOODS',
    # Cement
    'ULTRACEMCO':'CEMENT','SHREECEM':'CEMENT','AMBUJACEM':'CEMENT',
    # Telecom
    'BHARTIARTL':'TELECOM',
    # NBFC / Finance
    'BAJFINANCE':'NBFC','BAJAJFINSV':'NBFC','CHOLAFIN':'NBFC','MUTHOOTFIN':'NBFC',
    # Pharma
    'SUNPHARMA':'PHARMA','DRREDDY':'PHARMA','CIPLA':'PHARMA','DIVISLAB':'PHARMA',
    # Realty
    'DLF':'REALTY','GODREJPROP':'REALTY','OBEROIRLTY':'REALTY',

    # New-age tech / internet — added alongside HIGH_GROWTH_MOMENTUM_UNIVERSE
    # in run_paper_trading.py. Grouping these together matters: they tend to
    # move together on sentiment/risk-appetite shifts, so without this they'd
    # each silently count as their own separate "sector" and the concentration
    # cap would do nothing to prevent a fully correlated cluster of bets.
    'ZOMATO':'NEWAGE_TECH', 'NYKAA':'NEWAGE_TECH', 'PAYTM':'NEWAGE_TECH',
    'POLICYBZR':'NEWAGE_TECH', 'DELHIVERY':'NEWAGE_TECH', 'IRCTC':'NEWAGE_TECH',
    'NAUKRI':'NEWAGE_TECH', 'INDIAMART':'NEWAGE_TECH', 'CARTRADE':'NEWAGE_TECH',
    'MAPMYINDIA':'NEWAGE_TECH', 'EASEMYTRIP':'NEWAGE_TECH', 'NAZARA':'NEWAGE_TECH',

    # Defence — same reasoning: these move together hard on procurement
    # news/budget headlines (see run_paper_trading.py header note).
    'HAL':'DEFENCE', 'BEL':'DEFENCE', 'BDL':'DEFENCE', 'MAZDOCK':'DEFENCE',
    'COCHINSHIP':'DEFENCE', 'SOLARINDS':'DEFENCE', 'ASTRAMICRO':'DEFENCE',
    'MTARTECH':'DEFENCE', 'PARAS':'DEFENCE', 'ZENTEC':'DEFENCE',
    'DATAPATTNS':'DEFENCE', 'BEML':'DEFENCE', 'GRSE':'DEFENCE',

    # Renewable energy / EV
    'SUZLON':'RENEWABLE_EV', 'WAAREEENER':'RENEWABLE_EV', 'ADANIGREEN':'RENEWABLE_EV',
    'NTPCGREEN':'RENEWABLE_EV', 'ACMESOLAR':'RENEWABLE_EV', 'PREMIERENE':'RENEWABLE_EV',
    'JSWENERGY':'RENEWABLE_EV', 'TATAPOWER':'RENEWABLE_EV', 'INOXWIND':'RENEWABLE_EV',
}

MAX_SECTOR_EXPOSURE     = 3
MAX_OPEN_TRADES_DEFAULT = 7
MAX_HOLD_DAYS_DEFAULT   = 15
MAX_DRAWDOWN_DEFAULT    = 0.30


class SwingTradingBot:
    def __init__(self, send_emails=True, initial_equity=50000,
                 commission_per_share=0.005,
                 max_open_trades=MAX_OPEN_TRADES_DEFAULT,
                 max_hold_days=MAX_HOLD_DAYS_DEFAULT):

        self.fetcher    = DataFetcherFree()
        self.tech       = TechnicalIndicators()
        self.screener   = FundamentalScreener()
        self.signal_gen = SignalGenerator(self.tech, self.screener)
        self.notifier   = NotificationHandler(use_email=send_emails, use_sms=False)

        self.initial_equity       = initial_equity
        self.current_equity       = initial_equity
        self.commission_per_share = commission_per_share
        self.max_open_trades      = max_open_trades
        self.max_hold_days        = max_hold_days
        self.max_drawdown_pct     = MAX_DRAWDOWN_DEFAULT

        self.fundamentals_cache = {}
        self.peak_equity        = initial_equity

        logger.info(
            f"✓ Bot v3 HIGH-RISK | Equity: ₹{initial_equity:,} | "
            f"Max trades: {max_open_trades} | Max hold: {max_hold_days}d | "
            f"Circuit breaker: {MAX_DRAWDOWN_DEFAULT*100:.0f}%"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_fundamentals_safe(self, symbol, retry=2):
        if symbol in self.fundamentals_cache:
            return self.fundamentals_cache[symbol]
        for attempt in range(retry):
            try:
                fund = self.fetcher.get_fundamentals(symbol)
                self.fundamentals_cache[symbol] = fund
                return fund
            except Exception:
                if attempt < retry - 1:
                    time.sleep(0.4)
        return self.fetcher._default_fundamentals()

    def _get_market_regime(self, days=300):
        """
        Fetch Nifty 50 and classify regime.
        '^NSEI' is passed directly — DataFetcherFree._to_yf_symbol() leaves
        index symbols untouched (no .NS suffix).
        """
        try:
            nifty  = self.fetcher.get_historical_data('^NSEI', days=days, min_bars=200)
            if nifty is None:
                logger.warning("⚠️ Could not fetch Nifty data → NEUTRAL")
                return 'NEUTRAL'
            regime = SignalGenerator.classify_market_regime(nifty)
            logger.info(f"📊 Market regime: {regime}")
            return regime
        except Exception as e:
            logger.warning(f"⚠️ Regime check failed: {e} → NEUTRAL")
            return 'NEUTRAL'

    def _drawdown_ok(self):
        self.peak_equity = max(self.peak_equity, self.current_equity)
        dd = (self.peak_equity - self.current_equity) / self.peak_equity
        if dd >= self.max_drawdown_pct:
            logger.warning(f"🛑 Drawdown {dd*100:.1f}% ≥ {self.max_drawdown_pct*100:.0f}% — halting new entries")
            return False
        return True

    def _sector_counts(self, open_trades_dict):
        counts = {}
        for sym in open_trades_dict:
            s = SECTOR_MAP.get(sym, sym)
            counts[s] = counts.get(s, 0) + 1
        return counts

    def _apply_trailing_stop(self, trade):
        """
        Thin wrapper around trailing_stop.compute_trailing_stop() — the
        SAME function paper_trading_manager.py uses for live trading, so
        backtest and live run identical trailing-stop economics.

        Requires trade['initial_stop_loss'] to be set at trade-open time
        (immutable — the true 1R reference point). Without it, this used to
        recompute risk from entry_price - CURRENT stop_loss, which is wrong
        after the first ratchet (it can even go negative once the stop has
        passed entry_price) — traced through a concrete case where a
        position correctly ratcheted to its 2R tier, then incorrectly
        failed to progress to the 3R tier on a later bar despite reaching
        the original 3R price level, because "risk" was being measured from
        the wrong, already-moved reference point.
        """
        initial_sl = trade.get('initial_stop_loss', trade['stop_loss'])
        new_sl = compute_trailing_stop(
            trade['entry_price'], initial_sl, trade['stop_loss'], trade.get('_curr_price', trade['entry_price'])
        )
        trade['stop_loss'] = new_sl
        return trade

    # ── Live screening ────────────────────────────────────────────────────────

    def screen_stock(self, symbol):
        df = self.fetcher.get_historical_data(symbol, days=250, min_bars=50)
        if df is None:
            return 'HOLD', {'reason': 'Insufficient data'}
        fund   = self.get_fundamentals_safe(symbol)
        regime = self._get_market_regime()
        return self.signal_gen.generate_signal(
            df, symbol, fund, self.current_equity, market_regime=regime)

    def run_screening(self, stock_list, send_alerts=True):
        buy_signals = []
        regime      = self._get_market_regime()
        for symbol in stock_list:
            try:
                df = self.fetcher.get_historical_data(symbol, days=250, min_bars=50)
                if df is None:
                    continue
                fund = self.get_fundamentals_safe(symbol)
                sig, details = self.signal_gen.generate_signal(
                    df, symbol, fund, self.current_equity, market_regime=regime)
                if sig == 'BUY':
                    buy_signals.append((symbol, details))
                    if send_alerts:
                        self.notifier.send_signal('BUY', details, self.current_equity)
                    logger.info(
                        f"  🎯 BUY {symbol} | {details['entry_type']} | "
                        f"conf={details['confidence']} | R:R={details['risk_reward_ratio']}"
                    )
            except Exception as e:
                logger.warning(f"  ⚠️ {symbol}: {e}")
            time.sleep(0.25)
        logger.info(f"\n✅ Screening complete — {len(buy_signals)} BUY signals from {len(stock_list)} stocks")
        return buy_signals

    # ── Portfolio backtest ────────────────────────────────────────────────────

    def backtest_portfolio(self, stock_list, days=600, use_alpha_engine=True,
                            max_portfolio_risk_pct=0.16, min_alpha_score=None,
                            tier_size_multiplier=None):
        """
        use_alpha_engine: when True (default), every technical BUY signal
            also gets alpha_engine's cross-sectional conviction score —
            gating weak signals, scaling position size by tier, exactly as
            in live trading (run_paper_trading.py). When False, runs the
            pre-alpha-engine policy (signal_generator alone) — this is the
            A/B baseline: run the SAME stock_list/days with both settings
            and diff the performance reports to see the alpha engine's
            actual marginal contribution, rather than trusting a single
            absolute number.

        max_portfolio_risk_pct: caps total capital-at-stake across all
            open positions simultaneously, same mechanism as Step 2's live
            risk budget (run_paper_trading.py's MAX_PORTFOLIO_RISK_PCT).
            Defaults to 0.16 to match live's current setting — this
            matters for backtest fidelity: a backtest that doesn't share
            live's capital-allocation policy isn't actually predicting
            what live trading will do, just a different, untested policy.

        min_alpha_score / tier_size_multiplier: override the values on
            CompositeAlphaScore if you want to experiment with different
            thresholds; default to the same live-production values.

        Point-in-time correctness: every alpha_engine computation below
        uses only data available UP TO AND INCLUDING bar i — this is a
        walk-forward simulation, not a vectorized one, specifically so
        that's straightforward to get right and verify (see
        test_lookahead_bias in the accompanying test suite, which injects
        a deliberately unmistakable future price shock and confirms an
        earlier bar's decisions are completely unaffected by it).
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 PORTFOLIO BACKTEST v4 — {len(stock_list)} stocks, {days} days")
        logger.info(f"Initial Equity: ₹{self.initial_equity:,} | Max trades: {self.max_open_trades} | "
                    f"Alpha engine: {'ON' if use_alpha_engine else 'OFF (baseline)'}")
        logger.info(f"{'='*70}\n")

        alpha_scorer = CompositeAlphaScore() if use_alpha_engine else None
        min_alpha_score = min_alpha_score if min_alpha_score is not None else (
            alpha_scorer.MIN_ALPHA_SCORE_TO_TRADE if alpha_scorer else 0)
        tier_mult_map = tier_size_multiplier if tier_size_multiplier is not None else (
            alpha_scorer.TIER_SIZE_MULTIPLIER if alpha_scorer else {})
        pattern_weights = {}   # evolves across the run, mirroring cross-run persistence in live

        all_dfs = {}
        for i, sym in enumerate(stock_list):
            try:
                df = self.fetcher.get_historical_data(sym, days=days + 100, min_bars=60)
                if df is not None:
                    all_dfs[sym] = df
                    logger.info(f"  ✓ {sym} ({i+1}/{len(stock_list)}): {len(df)} candles")
                else:
                    logger.warning(f"  ⚠️ {sym}: insufficient data")
            except Exception as e:
                logger.warning(f"  ⚠️ {sym}: {e}")
            time.sleep(0.2)

        if not all_dfs:
            logger.error("❌ No valid data fetched")
            return None

        try:
            nifty_df = self.fetcher.get_historical_data('^NSEI', days=days + 100, min_bars=200)
        except Exception:
            nifty_df = None

        all_trades    = []
        open_trades   = {}
        last_exit_bar = {}
        self.current_equity = self.initial_equity
        self.peak_equity    = self.initial_equity
        equity_curve  = []

        max_candles = max(len(df) for df in all_dfs.values())

        for i in range(60, max_candles):
            if nifty_df is not None and i < len(nifty_df):
                regime = SignalGenerator.classify_market_regime(nifty_df.iloc[:i+1])
            else:
                regime = 'NEUTRAL'

            to_exit = []
            for sym, trade in list(open_trades.items()):
                if sym not in all_dfs or i >= len(all_dfs[sym]):
                    continue

                bar        = all_dfs[sym].iloc[i]
                curr_price = float(bar['close'])
                ep, ps     = trade['entry_price'], trade['position_size']
                hold_days  = i - trade['entry_index']

                trade['_curr_price'] = curr_price
                trade = self._apply_trailing_stop(trade)
                sl = trade['stop_loss']
                tp = trade['target']

                exit_triggered = False
                exit_reason    = ''
                exit_price     = 0.0

                if curr_price <= sl:
                    exit_triggered, exit_reason, exit_price = True, 'SL Hit',     sl
                elif curr_price >= tp:
                    exit_triggered, exit_reason, exit_price = True, 'Target Hit', tp
                elif hold_days >= self.max_hold_days:
                    exit_triggered, exit_reason, exit_price = True, 'Time Exit',  curr_price

                if exit_triggered:
                    commission = ps * self.commission_per_share * 2
                    net_pnl    = (exit_price - ep) * ps - commission

                    all_trades.append({
                        'symbol':        sym,
                        'entry_date':    trade['entry_date'],
                        'exit_date':     bar['datetime'],
                        'entry_price':   ep,
                        'exit_price':    exit_price,
                        'position_size': ps,
                        'exit_reason':   exit_reason,
                        'gross_pnl':     (exit_price - ep) * ps,
                        'commission':    commission,
                        'net_pnl':       net_pnl,
                        'result':        'WIN' if net_pnl > 0 else 'LOSS',
                        'hold_days':     hold_days,
                        'entry_type':    trade['entry_type'],
                        'market_regime': trade['market_regime'],
                        'confidence':    trade.get('confidence', 1),
                        'alpha_score':   trade.get('alpha_score'),
                        'alpha_tier':    trade.get('alpha_tier'),
                        'exit_index':    i,
                    })
                    self.current_equity += net_pnl
                    self.peak_equity     = max(self.peak_equity, self.current_equity)
                    last_exit_bar[sym]   = i
                    to_exit.append(sym)

            for sym in to_exit:
                del open_trades[sym]

            if not self._drawdown_ok():
                equity_curve.append({'bar': i, 'equity': self.current_equity})
                continue

            if len(open_trades) < self.max_open_trades and self.current_equity > 0:
                sector_counts = self._sector_counts(open_trades)

                # ── Point-in-time cross-sectional alpha prep ────────────────
                # Everything here uses ONLY data up to and including bar i —
                # see test_lookahead_bias in the test suite for the explicit
                # check that this holds (a future price shock injected at a
                # later bar has zero effect on an earlier bar's scores).
                regime_result, factor_ranks = {'tilts': {}}, {}
                if use_alpha_engine:
                    try:
                        if nifty_df is not None and i < len(nifty_df):
                            nifty_window = nifty_df.iloc[:i + 1]
                            if len(nifty_window) >= 30:
                                regime_result = alpha_scorer.regime_detector.classify(nifty_window)

                        candidates = {s: all_dfs[s].iloc[:i + 1] for s in all_dfs
                                      if s not in open_trades and i < len(all_dfs[s])}
                        if len(candidates) >= 2:
                            factor_values = {s: alpha_scorer.factor_engine.compute_all(df)
                                              for s, df in candidates.items()}
                            factor_ranks = alpha_scorer.ranker.rank_universe(factor_values, sector_map=SECTOR_MAP)

                        # Recalibrate pattern weights from trades CLOSED SO FAR
                        # ONLY. all_trades is safe to use directly here without
                        # an extra bar-index filter: exits for bar i are
                        # processed earlier in THIS SAME iteration (above),
                        # and every prior iteration already appended its own
                        # closes before this point ran — so by construction,
                        # every entry in all_trades at this point in the loop
                        # closed at or before bar i, never after it.
                        if all_trades:
                            pattern_weights = alpha_scorer.calibrator.calibrate_weights(
                                pd.DataFrame(all_trades), previous_weights=pattern_weights,
                                pattern_col='entry_type', pnl_col='net_pnl')
                    except Exception as e:
                        logger.debug(f"Alpha engine unavailable at bar {i}: {e}")
                        regime_result, factor_ranks = {'tilts': {}}, {}

                current_agg_risk = sum(
                    (t['entry_price'] - t['stop_loss']) * t['position_size']
                    for t in open_trades.values()
                )

                for sym in all_dfs:
                    if sym in open_trades or i >= len(all_dfs[sym]):
                        continue

                    sym_sector = SECTOR_MAP.get(sym, sym)
                    if sector_counts.get(sym_sector, 0) >= MAX_SECTOR_EXPOSURE:
                        continue

                    try:
                        df_win = all_dfs[sym].iloc[:i+1].copy()
                        fund   = self.get_fundamentals_safe(sym)
                        lex    = last_exit_bar.get(sym)

                        sig, det = self.signal_gen.generate_signal(
                            df_win, sym, fund, self.current_equity,
                            market_regime=regime, last_exit_bar=lex
                        )

                        if sig != 'BUY':
                            continue

                        alpha_score, alpha_tier = None, None
                        if use_alpha_engine:
                            pattern_weight = pattern_weights.get(det['entry_type'], 1.0)
                            alpha_result = alpha_scorer.score_symbol(
                                sym, factor_ranks.get(sym, {}), regime_result, pattern_weight=pattern_weight,
                            )
                            if alpha_result['composite_score'] is None or alpha_result['composite_score'] < min_alpha_score:
                                continue
                            alpha_score, alpha_tier = alpha_result['composite_score'], alpha_result['tier']
                            tier_mult = tier_mult_map.get(alpha_tier, 1.0)
                            if tier_mult != 1.0:
                                det['position_size'] = max(1, int(det['position_size'] * tier_mult))

                        # Portfolio risk budget — same mechanism as live
                        # (run_paper_trading.py Step 6), applied AFTER any
                        # tier-based resize above, so it remains the hard
                        # final constraint regardless of conviction tier.
                        risk_per_share = det['entry_price'] - det['stop_loss']
                        risk_budget_left = max_portfolio_risk_pct * self.current_equity - current_agg_risk
                        if risk_budget_left <= 0:
                            continue
                        trade_risk = det['position_size'] * risk_per_share
                        if trade_risk > risk_budget_left:
                            shrunk = max(0, int(risk_budget_left / risk_per_share)) if risk_per_share > 0 else 0
                            if shrunk < 1:
                                continue
                            det['position_size'] = shrunk

                        capital = det['entry_price'] * det['position_size']
                        # RISK_PROFILE['max_capital_pct'] referenced directly
                        # rather than a separately hardcoded number here —
                        # signal_generator.py already sizes to this cap
                        # internally; this is a basic affordability sanity
                        # check on top, not a second independent policy that
                        # could silently drift out of sync with it again (an
                        # older version of this check used a stale 0.30
                        # while signal_generator had already moved to 0.40).
                        if capital <= self.current_equity * RISK_PROFILE['max_capital_pct']:
                            open_trades[sym] = {
                                'entry_price':   det['entry_price'],
                                'stop_loss':     det['stop_loss'],
                                'initial_stop_loss': det['stop_loss'],   # immutable — see trailing_stop.py
                                'target':        det['target_price'],
                                'position_size': det['position_size'],
                                'entry_date':    all_dfs[sym].iloc[i]['datetime'],
                                'entry_index':   i,
                                'entry_type':    det['entry_type'],
                                'market_regime': regime,
                                'confidence':    det.get('confidence', 1),
                                'alpha_score':   alpha_score,
                                'alpha_tier':    alpha_tier,
                            }
                            sector_counts[sym_sector] = sector_counts.get(sym_sector, 0) + 1
                            current_agg_risk += (det['entry_price'] - det['stop_loss']) * det['position_size']
                            if len(open_trades) >= self.max_open_trades:
                                break
                    except Exception as e:
                        logger.debug(f"Signal error {sym}: {e}")

            equity_curve.append({'bar': i, 'equity': self.current_equity})

        # Stored as an instance attribute rather than added to the return
        # value — backtest_portfolio() has always returned just trades_df,
        # and existing callers (run_backtest.py, any ad-hoc scripts) may
        # already rely on that. Anyone who wants the equity curve too reads
        # bot.last_equity_curve right after calling backtest_portfolio().
        self.last_equity_curve = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame(columns=['bar', 'equity'])

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            eq_df     = self.last_equity_curve
            self._print_summary(trades_df, eq_df)
            return trades_df
        else:
            logger.warning("⚠️ No trades generated")
            return None

    def _print_summary(self, trades_df, eq_df=None):
        wins   = (trades_df['result'] == 'WIN').sum()
        losses = (trades_df['result'] == 'LOSS').sum()
        total  = len(trades_df)
        wr     = wins / total * 100

        avg_win  = trades_df.loc[trades_df['result']=='WIN',  'net_pnl'].mean() if wins   else 0
        avg_loss = trades_df.loc[trades_df['result']=='LOSS', 'net_pnl'].mean() if losses else 0
        gw       = trades_df.loc[trades_df['result']=='WIN',  'net_pnl'].sum()
        gl       = abs(trades_df.loc[trades_df['result']=='LOSS','net_pnl'].sum()) or 1e-6
        pf       = gw / gl
        ret_pct  = (self.current_equity - self.initial_equity) / self.initial_equity * 100

        max_dd = 0
        if eq_df is not None and len(eq_df) > 0:
            eq          = eq_df['equity']
            rolling_max = eq.cummax()
            max_dd      = ((rolling_max - eq) / rolling_max).max() * 100

        avg_hold         = trades_df['hold_days'].mean()
        trades_per_month = total / (600 / 21)

        logger.info(f"\n{'='*70}")
        logger.info("PORTFOLIO BACKTEST RESULTS — HIGH-RISK v3")
        logger.info(f"{'='*70}")
        logger.info(f"Total Trades        : {total}  (~{trades_per_month:.1f}/month)")
        logger.info(f"Avg Hold Days       : {avg_hold:.1f}")
        logger.info(f"Wins / Losses       : {wins} / {losses}")
        logger.info(f"Win Rate            : {wr:.1f}%")
        logger.info(f"Avg Win / Avg Loss  : ₹{avg_win:,.0f} / ₹{avg_loss:,.0f}")
        logger.info(f"Profit Factor       : {pf:.2f}x")
        logger.info(f"Max Drawdown        : {max_dd:.1f}%")
        logger.info(f"Total P&L           : ₹{trades_df['net_pnl'].sum():,.0f}")
        logger.info(f"Initial Equity      : ₹{self.initial_equity:,}")
        logger.info(f"Final Equity        : ₹{self.current_equity:,.0f}")
        logger.info(f"Return              : {ret_pct:.2f}%")
        logger.info(f"{'='*70}")

        if 'market_regime' in trades_df.columns:
            logger.info("By Market Regime:")
            logger.info(trades_df.groupby('market_regime')['net_pnl'].agg(['count','sum','mean']).to_string())

        if 'entry_type' in trades_df.columns:
            logger.info("\nBy Entry Pattern:")
            logger.info(trades_df.groupby('entry_type').agg(
                trades=('net_pnl','count'),
                win_rate=('result', lambda x: f"{(x=='WIN').sum()/len(x)*100:.0f}%"),
                total_pnl=('net_pnl','sum')
            ).to_string())

        logger.info(f"{'='*70}\n")
