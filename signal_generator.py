# signal_generator.py  ── HIGH-RISK / HIGH-FREQUENCY VERSION v3
# ─────────────────────────────────────────────────────────────────────────────
# Philosophy: capture more setups per week by widening pattern criteria,
# shortening the minimum data requirement, accepting NEUTRAL regime longs,
# lowering R:R floor to 1.5, and adding 4 new aggressive patterns:
#   • Momentum burst (RSI crosses 55 with ADX expansion + volume)
#   • BB squeeze breakout (volatility contraction → expansion)
#   • EMA-12/20 golden cross
#   • CMF accumulation above EMA-50
#
# Risk controls kept: lookahead-free, debounce 2 bars, drawdown circuit
# breaker (handled in bot), sector cap (handled in bot).
# ─────────────────────────────────────────────────────────────────────────────

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SECTOR_PE = {
    'RELIANCE': 25, 'TCS': 30, 'INFY': 28, 'WIPRO': 24, 'HCLTECH': 22,
    'HDFCBANK': 18, 'ICICIBANK': 17, 'SBIN': 10, 'KOTAKBANK': 20, 'AXISBANK': 14,
    'HINDUNILVR': 55, 'NESTLEIND': 70, 'ITC': 25, 'ASIANPAINT': 60, 'TITAN': 80,
    'MARUTI': 28, 'TATAMOTORS': 15, 'BAJAJ-AUTO': 25,
    'TATASTEEL': 12, 'JSWSTEEL': 10, 'HINDALCO': 10,
    'BHARTIARTL': 35, 'LT': 28, 'BAJFINANCE': 30, 'ULTRACEMCO': 35,
}

# ── Tunable risk profile ──────────────────────────────────────────────────────
RISK_PROFILE = {
    # position sizing
    'risk_pct_per_trade':  0.025,   # 2.5% of equity at risk per trade (vs 1.5% conservative)
    'max_capital_pct':     0.30,    # max 30% of equity in one name

    # entry filters — lower values = more trades
    'min_bars':            50,      # was 80
    'min_adx':             12,      # was 18
    'rsi_lo':              25,      # was 30
    'rsi_hi':              75,      # was 68 — allows buying momentum
    'debounce_bars':        2,      # was 5 — re-enter faster after an exit

    # pattern thresholds
    'breakout_vol_mult':   1.2,     # was 1.5 — easier volume breakout
    'breakout_pct':        1.002,   # price barely above resistance counts
    'pullback_band':       0.025,   # wider EMA-20 touch zone (was 0.015)
    'pullback_vol_mult':   0.7,     # was 0.8

    # exit parameters — tighter stops = larger size = more $ upside
    'sl_mult_breakout':    1.2,     # was 1.5
    'sl_mult_pullback':    1.5,     # was 2.0
    'sl_mult_default':     1.3,     # was 1.8
    'tgt_mult_breakout':   3.5,     # was 3.0 — aim higher on momentum
    'tgt_mult_pullback':   4.0,
    'tgt_mult_default':    3.0,
    'min_rr':              1.5,     # was 2.0 — accept more setups

    # bear market handling
    'allow_bear_longs':    True,    # was False — still trade but with tighter rules
    'bear_rsi_cap':        55,      # in BEAR, RSI must be below this (not extended)
}


class SignalGenerator:
    def __init__(self, tech_indicators, fundamental_screener):
        self.tech = tech_indicators
        self.fund = fundamental_screener
        self.R    = RISK_PROFILE
        logger.info("✓ SignalGenerator v3 — HIGH-RISK / HIGH-FREQUENCY")

    # ─────────────────────────────────────────────────────────────────────────
    def generate_signal(self, df, symbol, fundamentals, current_equity=50000,
                        market_regime='BULL', last_exit_bar=None):
        """
        Returns ('BUY', details) or ('HOLD', {'reason': ...}).

        Nine entry patterns — any single one is sufficient to trigger:
          1. Resistance breakout + volume
          2. Pullback to EMA-20
          3. Bullish engulfing candle
          4. RSI divergence
          5. Stochastic cross from oversold
          6. Momentum burst (RSI crosses 55)       ← NEW
          7. BB squeeze breakout                    ← NEW
          8. EMA 12/20 golden cross                 ← NEW
          9. CMF accumulation above EMA-50          ← NEW
        """
        R = self.R

        if len(df) < R['min_bars']:
            return 'HOLD', {'reason': f'Need {R["min_bars"]} bars, got {len(df)}'}

        if market_regime == 'BEAR' and not R['allow_bear_longs']:
            return 'HOLD', {'reason': 'BEAR regime, longs disabled'}

        current_bar = len(df) - 1
        if last_exit_bar is not None and (current_bar - last_exit_bar) < R['debounce_bars']:
            return 'HOLD', {'reason': f'Debounce ({current_bar - last_exit_bar} bars)'}

        # ── Compute indicators ────────────────────────────────────────────────
        d = df.copy()
        d['ema_9']   = self.tech.calculate_ema(d['close'], 9)
        d['ema_12']  = self.tech.calculate_ema(d['close'], 12)
        d['ema_20']  = self.tech.calculate_ema(d['close'], 20)
        d['ema_50']  = self.tech.calculate_ema(d['close'], 50)
        d['sma_200'] = self.tech.calculate_sma(d['close'], 200)
        d['rsi']     = self.tech.calculate_rsi(d['close'], 14)
        d['rsi_9']   = self.tech.calculate_rsi(d['close'], 9)

        macd = self.tech.calculate_macd(d['close'])
        d['macd']          = macd['macd']
        d['macd_signal']   = macd['signal']
        d['macd_histogram']= macd['histogram']

        bb = self.tech.calculate_bollinger_bands(d['close'], 20, 2)
        d['bb_upper'] = bb['upper']
        d['bb_lower'] = bb['lower']
        d['bb_middle']= bb['middle']
        d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / d['bb_middle']

        d['atr']        = self.tech.calculate_atr(d['high'], d['low'], d['close'], 14)
        d['adx']        = self.tech.calculate_adx(d['high'], d['low'], d['close'], 14)
        d['volume_sma'] = self.tech.calculate_volume_sma(d['volume'], 20)

        stoch = self.tech.calculate_stochastic(d['high'], d['low'], d['close'], 14)
        d['stoch_k'] = stoch['k']
        d['stoch_d'] = stoch['d']

        d['wr']  = self.tech.calculate_williams_r(d['high'], d['low'], d['close'], 14)
        d['cmf'] = self.tech.calculate_cmf(d['high'], d['low'], d['close'], d['volume'], 20)

        latest = d.iloc[-1]
        prev   = d.iloc[-2]
        prev2  = d.iloc[-3] if len(d) >= 3 else prev

        # ── Trend / structure filter (relaxed) ───────────────────────────────
        ema50_slope  = latest['ema_50'] > d['ema_50'].iloc[-8]
        above_ema20  = latest['close']  > latest['ema_20']
        above_ema50  = latest['close']  > latest['ema_50']

        if market_regime == 'BEAR':
            if not above_ema50:
                return 'HOLD', {'reason': 'BEAR: price below EMA-50'}
            if latest['rsi'] > R['bear_rsi_cap']:
                return 'HOLD', {'reason': f'BEAR: RSI {latest["rsi"]:.0f} extended'}
        else:
            if not (above_ema20 or (above_ema50 and ema50_slope)):
                return 'HOLD', {'reason': 'Below both EMA-20 and declining EMA-50'}

        # ── ADX ───────────────────────────────────────────────────────────────
        if latest['adx'] < R['min_adx']:
            return 'HOLD', {'reason': f'ADX {latest["adx"]:.1f} < {R["min_adx"]}'}

        # ── RSI bounds ────────────────────────────────────────────────────────
        if not (R['rsi_lo'] < latest['rsi'] < R['rsi_hi']):
            return 'HOLD', {'reason': f'RSI {latest["rsi"]:.1f} outside [{R["rsi_lo"]},{R["rsi_hi"]}]'}

        # ── MACD: bullish or histogram turning up ─────────────────────────────
        hist_up  = latest['macd_histogram'] > prev['macd_histogram'] > prev2['macd_histogram']
        macd_ok  = (latest['macd'] > latest['macd_signal']) or hist_up
        if not macd_ok:
            return 'HOLD', {'reason': 'MACD not constructive'}

        # ── Entry patterns ────────────────────────────────────────────────────
        resistance_20d = d['close'].iloc[-21:-1].max()

        # 1. Resistance breakout + volume surge
        breakout = (
            latest['close'] > resistance_20d * R['breakout_pct'] and
            latest['volume'] > latest['volume_sma'] * R['breakout_vol_mult']
        )

        # 2. Pullback to EMA-20 with recovery close
        near_ema20 = abs(latest['close'] - latest['ema_20']) / latest['ema_20'] < R['pullback_band']
        pullback   = (
            near_ema20 and
            latest['close'] > prev['close'] and
            latest['volume'] > latest['volume_sma'] * R['pullback_vol_mult']
        )

        # 3. Bullish engulfing above EMA-50
        body_now  = abs(latest['close'] - latest['open'])
        body_prev = abs(prev['close']   - prev['open'])
        engulfing = (
            latest['close'] > latest['open'] and
            prev['close']   < prev['open']   and
            latest['close'] > prev['open']   and
            latest['open']  < prev['close']  and
            body_now > body_prev * 0.7
        )

        # 4. RSI bullish divergence
        rsi_divergence = (
            latest['close'] < prev2['close'] and
            latest['rsi_9'] > prev2['rsi_9'] and
            latest['rsi_9'] < 60
        )

        # 5. Stochastic cross from oversold (widened to 35)
        stoch_cross = (
            prev['stoch_k']   < 35 and
            latest['stoch_k'] > latest['stoch_d'] and
            prev['stoch_k']   < prev['stoch_d']
        )

        # 6. Momentum burst — RSI just crossed 55, ADX expanding, volume
        momentum_burst = (
            latest['rsi']  > 55 and
            prev['rsi']    < 55 and
            latest['adx']  > prev['adx'] and
            latest['volume'] > latest['volume_sma'] * 1.1
        )

        # 7. BB squeeze breakout — bands contracted, now expanding bullishly
        bb_width_min = d['bb_width'].iloc[-6:-1].min()
        bb_squeeze   = (
            prev['bb_width']   <= bb_width_min * 1.05 and
            latest['bb_width']  > prev['bb_width'] and
            latest['close']     > latest['bb_middle'] and
            latest['volume']    > latest['volume_sma']
        )

        # 8. EMA 12/20 golden cross
        ema_cross = (
            prev['ema_12']   <= prev['ema_20'] and
            latest['ema_12']  > latest['ema_20'] and
            above_ema50
        )

        # 9. CMF accumulation (institutional buying) + holding EMA-50
        cmf_accum = (
            float(latest['cmf']) > 0.08 and
            above_ema50 and
            latest['volume'] > latest['volume_sma'] * 0.9
        )

        entry_signals = {
            'breakout':       breakout,
            'pullback':       pullback,
            'engulfing':      engulfing,
            'rsi_divergence': rsi_divergence,
            'stoch_cross':    stoch_cross,
            'momentum_burst': momentum_burst,
            'bb_squeeze':     bb_squeeze,
            'ema_cross':      ema_cross,
            'cmf_accum':      cmf_accum,
        }
        active_patterns = [k for k, v in entry_signals.items() if v]

        if not active_patterns:
            return 'HOLD', {'reason': 'No pattern triggered'}

        # ── Fundamental gate — only block hard fails ──────────────────────────
        try:
            fund_passed, fund_checks = self.fund.check_fundamental_gate(fundamentals)
            if isinstance(fund_checks, dict) and 'reason' in fund_checks:
                return 'HOLD', {'reason': f'Fundamental hard-fail: {fund_checks["reason"]}'}
        except Exception:
            pass  # data missing → proceed

        # ── Position sizing ───────────────────────────────────────────────────
        atr = float(latest['atr'])
        if atr < latest['close'] * 0.005:
            atr = latest['close'] * 0.012

        entry_price = float(latest['close'])
        primary     = active_patterns[0]

        if primary in ('breakout', 'momentum_burst'):
            sl_mult, tgt_mult = R['sl_mult_breakout'], R['tgt_mult_breakout']
        elif primary in ('pullback', 'ema_cross', 'cmf_accum'):
            sl_mult, tgt_mult = R['sl_mult_pullback'], R['tgt_mult_pullback']
        elif primary == 'bb_squeeze':
            sl_mult, tgt_mult = 1.0, 3.5   # very tight stop on squeeze
        else:
            sl_mult, tgt_mult = R['sl_mult_default'], R['tgt_mult_default']

        stop_loss      = round(entry_price - sl_mult * atr, 2)
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:
            return 'HOLD', {'reason': 'Degenerate stop-loss'}

        max_risk  = current_equity * R['risk_pct_per_trade']
        max_cap   = current_equity * R['max_capital_pct']

        position_size = int(max_risk / risk_per_share)
        position_size = max(1, min(position_size, int(max_cap / entry_price)))

        target_price = round(entry_price + tgt_mult * risk_per_share, 2)
        actual_rr    = (target_price - entry_price) / risk_per_share

        if actual_rr < R['min_rr']:
            return 'HOLD', {'reason': f'R:R {actual_rr:.2f} < {R["min_rr"]}'}

        confidence = min(9, len(active_patterns))

        signal_details = {
            'symbol':             symbol,
            'entry_price':        round(entry_price, 2),
            'stop_loss':          stop_loss,
            'target_price':       target_price,
            'position_size':      position_size,
            'risk':               round(position_size * risk_per_share, 2),
            'reward':             round(position_size * (target_price - entry_price), 2),
            'risk_reward_ratio':  round(actual_rr, 2),
            'entry_type':         primary,
            'patterns_triggered': active_patterns,
            'confidence':         confidence,
            'market_regime':      market_regime,
            'indicators': {
                'rsi':      round(float(latest['rsi']),  2),
                'rsi_9':    round(float(latest['rsi_9']), 2),
                'macd':     round(float(latest['macd']),  4),
                'macd_hist':round(float(latest['macd_histogram']), 4),
                'adx':      round(float(latest['adx']),  2),
                'atr':      round(atr,                    2),
                'bb_width': round(float(latest['bb_width']), 4),
                'cmf':      round(float(latest['cmf']),  3),
                'stoch_k':  round(float(latest['stoch_k']), 1),
            },
            'fundamentals': {
                'pe_ratio':       fundamentals.get('pe_ratio', 'N/A'),
                'debt_to_equity': fundamentals.get('debt_to_equity', 'N/A'),
                'roe':            f"{fundamentals.get('roe_5yr', 0)*100:.1f}%",
                'revenue_growth': f"{fundamentals.get('revenue_cagr', 0)*100:.1f}%",
            },
            'breakout':  'breakout'  in active_patterns,
            'pullback':  'pullback'  in active_patterns,
            'timestamp': latest.get('datetime', pd.Timestamp.now()),
        }

        return 'BUY', signal_details

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def classify_market_regime(nifty_df):
        """BULL / NEUTRAL / BEAR based on Nifty 50 EMA-50 vs SMA-200."""
        if nifty_df is None or len(nifty_df) < 200:
            return 'NEUTRAL'
        from technical_indicators import TechnicalIndicators
        tech  = TechnicalIndicators()
        ema50 = tech.calculate_ema(nifty_df['close'], 50)
        s200  = tech.calculate_sma(nifty_df['close'], 200)
        c, e, s = nifty_df['close'].iloc[-1], ema50.iloc[-1], s200.iloc[-1]
        if c > e and e > s:
            return 'BULL'
        elif c < e and e < s:
            return 'BEAR'
        return 'NEUTRAL'
