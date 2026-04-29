# signal_generator.py  ── v4  LARGECAP + MIDCAP DUAL-MODE
# ─────────────────────────────────────────────────────────────────────────────
# Each symbol is classified as LARGECAP or MIDCAP before signal generation.
# The two paths share indicator computation but diverge on:
#   • Entry patterns  — midcaps get 5 momentum/breakout patterns suited to
#                       less-liquid, higher-volatility names
#   • Thresholds      — wider ATR stops, stricter volume requirements for mids
#   • Fundamental gate — midcap hard limits differ (see fundamental_screener)
#   • Scoring         — midcap confidence penalised slightly for liquidity risk
# ─────────────────────────────────────────────────────────────────────────────

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Universe classification ───────────────────────────────────────────────────
LARGECAP_SYMBOLS = {
    'RELIANCE','TCS','INFY','WIPRO','HCLTECH','TECHM','LTIM','MPHASIS',
    'HDFCBANK','ICICIBANK','SBIN','KOTAKBANK','AXISBANK','INDUSINDBK',
    'FEDERALBNK','HINDUNILVR','ITC','NESTLEIND','ASIANPAINT','TITAN',
    'MARUTI','TATAMOTOR','BAJAJ-AUTO','TATASTEEL','JSWSTEEL','HINDALCO',
    'BHARTIARTL','LT','BAJFINANCE','ULTRACEMCO','ONGC','BPCL','GAIL',
    'SUNPHARMA','DRREDDY','CIPLA','SIEMENS','ABB','DLF','CHOLAFIN',
}
# Anything not in LARGECAP_SYMBOLS is treated as MIDCAP automatically.

SECTOR_PE = {
    'RELIANCE':25,'TCS':30,'INFY':28,'WIPRO':24,'HCLTECH':22,
    'HDFCBANK':18,'ICICIBANK':17,'SBIN':10,'KOTAKBANK':20,'AXISBANK':14,
    'HINDUNILVR':55,'NESTLEIND':70,'ITC':25,'ASIANPAINT':60,'TITAN':80,
    'MARUTI':28,'TATAMOTOR':15,'BAJAJ-AUTO':25,
    'TATASTEEL':12,'JSWSTEEL':10,'HINDALCO':10,
    'BHARTIARTL':35,'LT':28,'BAJFINANCE':30,'ULTRACEMCO':35,
    '_MC_IT':35,'_MC_PHARMA':40,'_MC_FMCG':50,'_MC_AUTO':25,
    '_MC_BANK':15,'_MC_INFRA':30,'_MC_NBFC':28,'_MC_REALTY':35,
    '_MC_DEFAULT':30,
}

# ── Risk profiles ─────────────────────────────────────────────────────────────
LARGECAP_PROFILE = {
    'risk_pct_per_trade':  0.025,
    'max_capital_pct':     0.30,
    'min_bars':            50,
    'min_adx':             12,
    'rsi_lo':              25,
    'rsi_hi':              75,
    'debounce_bars':        2,
    'breakout_vol_mult':   1.2,
    'breakout_pct':        1.002,
    'pullback_band':       0.025,
    'pullback_vol_mult':   0.7,
    'sl_mult_breakout':    1.2,
    'sl_mult_pullback':    1.5,
    'sl_mult_default':     1.3,
    'tgt_mult_breakout':   3.5,
    'tgt_mult_pullback':   4.0,
    'tgt_mult_default':    3.0,
    'min_rr':              1.5,
    'allow_bear_longs':    True,
    'bear_rsi_cap':        55,
}

MIDCAP_PROFILE = {
    'risk_pct_per_trade':  0.020,
    'max_capital_pct':     0.20,    # never >20% in a single midcap
    'min_bars':            60,      # need more history to confirm structure
    'min_adx':             18,      # require stronger trend — midcaps chop more
    'rsi_lo':              30,
    'rsi_hi':              72,
    'debounce_bars':        3,
    'breakout_vol_mult':   1.8,     # stricter volume confirmation
    'breakout_pct':        1.005,
    'pullback_band':       0.030,
    'pullback_vol_mult':   0.8,
    'sl_mult_breakout':    1.5,     # wider stops for volatility
    'sl_mult_pullback':    2.0,
    'sl_mult_default':     1.7,
    'tgt_mult_breakout':   4.0,     # bigger move potential in midcaps
    'tgt_mult_pullback':   4.5,
    'tgt_mult_default':    3.5,
    'min_rr':              2.0,     # stricter R:R for higher-risk names
    'allow_bear_longs':    False,   # never long midcaps in bear regime
    'bear_rsi_cap':        50,
}


class SignalGenerator:
    def __init__(self, tech_indicators, fundamental_screener):
        self.tech = tech_indicators
        self.fund = fundamental_screener
        logger.info("✓ SignalGenerator v4 — LARGECAP + MIDCAP dual-mode")

    # ─────────────────────────────────────────────────────────────────────────
    def generate_signal(self, df, symbol, fundamentals, current_equity=50000,
                        market_regime='BULL', last_exit_bar=None):
        """
        Returns ('BUY', details) or ('HOLD', {'reason': ...}).
        Automatically routes through largecap or midcap logic.
        """
        is_midcap = symbol not in LARGECAP_SYMBOLS
        R         = MIDCAP_PROFILE if is_midcap else LARGECAP_PROFILE
        cap_label = 'MIDCAP'     if is_midcap else 'LARGECAP'

        if len(df) < R['min_bars']:
            return 'HOLD', {'reason': f'Need {R["min_bars"]} bars, got {len(df)}'}

        if market_regime == 'BEAR' and not R['allow_bear_longs']:
            return 'HOLD', {'reason': f'BEAR regime — {cap_label} longs disabled'}

        current_bar = len(df) - 1
        if last_exit_bar is not None and (current_bar - last_exit_bar) < R['debounce_bars']:
            return 'HOLD', {'reason': f'Debounce ({current_bar - last_exit_bar} bars)'}

        # ── Indicators ────────────────────────────────────────────────────────
        d = df.copy()
        d['ema_9']   = self.tech.calculate_ema(d['close'], 9)
        d['ema_12']  = self.tech.calculate_ema(d['close'], 12)
        d['ema_20']  = self.tech.calculate_ema(d['close'], 20)
        d['ema_50']  = self.tech.calculate_ema(d['close'], 50)
        d['sma_200'] = self.tech.calculate_sma(d['close'], 200)
        d['rsi']     = self.tech.calculate_rsi(d['close'], 14)
        d['rsi_9']   = self.tech.calculate_rsi(d['close'], 9)

        macd = self.tech.calculate_macd(d['close'])
        d['macd']           = macd['macd']
        d['macd_signal']    = macd['signal']
        d['macd_histogram'] = macd['histogram']

        bb = self.tech.calculate_bollinger_bands(d['close'], 20, 2)
        d['bb_upper']  = bb['upper']
        d['bb_lower']  = bb['lower']
        d['bb_middle'] = bb['middle']
        d['bb_width']  = (d['bb_upper'] - d['bb_lower']) / d['bb_middle']

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

        # ── Shared pre-filters ────────────────────────────────────────────────
        above_ema20 = latest['close'] > latest['ema_20']
        above_ema50 = latest['close'] > latest['ema_50']
        ema50_slope = latest['ema_50'] > d['ema_50'].iloc[-8]

        if market_regime == 'BEAR':
            if not above_ema50:
                return 'HOLD', {'reason': 'BEAR: price below EMA-50'}
            if latest['rsi'] > R['bear_rsi_cap']:
                return 'HOLD', {'reason': f'BEAR: RSI {latest["rsi"]:.0f} extended'}
        else:
            if not (above_ema20 or (above_ema50 and ema50_slope)):
                return 'HOLD', {'reason': 'Below EMA-20 and declining EMA-50'}

        if latest['adx'] < R['min_adx']:
            return 'HOLD', {'reason': f'ADX {latest["adx"]:.1f} < {R["min_adx"]}'}

        if not (R['rsi_lo'] < latest['rsi'] < R['rsi_hi']):
            return 'HOLD', {'reason': f'RSI {latest["rsi"]:.1f} outside [{R["rsi_lo"]},{R["rsi_hi"]}]'}

        hist_up = latest['macd_histogram'] > prev['macd_histogram'] > prev2['macd_histogram']
        macd_ok = (latest['macd'] > latest['macd_signal']) or hist_up
        if not macd_ok:
            return 'HOLD', {'reason': 'MACD not constructive'}

        # ── Route to pattern set ──────────────────────────────────────────────
        if is_midcap:
            active_patterns, primary = self._midcap_patterns(d, latest, prev, prev2, R)
        else:
            active_patterns, primary = self._largecap_patterns(d, latest, prev, prev2, R)

        if not active_patterns:
            return 'HOLD', {'reason': 'No pattern triggered'}

        # ── Fundamental gate ──────────────────────────────────────────────────
        try:
            fund_passed, fund_checks = self.fund.check_fundamental_gate(
                fundamentals, is_midcap=is_midcap
            )
            if not fund_passed:
                reason = fund_checks.get('reason', 'Fundamental gate failed')
                return 'HOLD', {'reason': f'Fund fail: {reason}'}
        except Exception:
            pass

        # ── Position sizing ───────────────────────────────────────────────────
        atr = float(latest['atr'])
        if atr < latest['close'] * 0.005:
            atr = latest['close'] * 0.012

        entry_price = float(latest['close'])

        if primary in ('breakout', 'momentum_burst', 'mc_volume_breakout',
                       'mc_52w_breakout', 'mc_gap_continuation'):
            sl_mult  = R['sl_mult_breakout']
            tgt_mult = R['tgt_mult_breakout']
        elif primary in ('pullback', 'ema_cross', 'cmf_accum', 'mc_ema_cluster'):
            sl_mult  = R['sl_mult_pullback']
            tgt_mult = R['tgt_mult_pullback']
        elif primary == 'bb_squeeze':
            sl_mult, tgt_mult = 1.0, 3.5
        else:
            sl_mult  = R['sl_mult_default']
            tgt_mult = R['tgt_mult_default']

        stop_loss      = round(entry_price - sl_mult * atr, 2)
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:
            return 'HOLD', {'reason': 'Degenerate stop-loss'}

        max_risk = current_equity * R['risk_pct_per_trade']
        max_cap  = current_equity * R['max_capital_pct']

        position_size = int(max_risk / risk_per_share)
        position_size = max(1, min(position_size, int(max_cap / entry_price)))

        target_price = round(entry_price + tgt_mult * risk_per_share, 2)
        actual_rr    = (target_price - entry_price) / risk_per_share

        if actual_rr < R['min_rr']:
            return 'HOLD', {'reason': f'R:R {actual_rr:.2f} < {R["min_rr"]}'}

        confidence = min(9, len(active_patterns))
        if is_midcap:
            confidence = max(1, confidence - 1)   # liquidity haircut

        signal_details = {
            'symbol':             symbol,
            'cap_type':           cap_label,
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
                'rsi':       round(float(latest['rsi']),            2),
                'rsi_9':     round(float(latest['rsi_9']),          2),
                'macd':      round(float(latest['macd']),           4),
                'macd_hist': round(float(latest['macd_histogram']), 4),
                'adx':       round(float(latest['adx']),            2),
                'atr':       round(atr,                             2),
                'bb_width':  round(float(latest['bb_width']),       4),
                'cmf':       round(float(latest['cmf']),            3),
                'stoch_k':   round(float(latest['stoch_k']),        1),
            },
            'fundamentals': {
                'pe_ratio':       fundamentals.get('pe_ratio', 'N/A'),
                'debt_to_equity': fundamentals.get('debt_to_equity', 'N/A'),
                'roe':            f"{fundamentals.get('roe_5yr', 0)*100:.1f}%",
                'revenue_growth': f"{fundamentals.get('revenue_cagr', 0)*100:.1f}%",
            },
            'breakout': any(p in active_patterns for p in
                            ('breakout','mc_volume_breakout','mc_52w_breakout')),
            'pullback': 'pullback' in active_patterns,
            'timestamp': latest.get('datetime', pd.Timestamp.now()),
        }

        return 'BUY', signal_details

    # ── Largecap patterns (9 proven patterns for liquid names) ───────────────
    def _largecap_patterns(self, d, latest, prev, prev2, R):
        resistance_20d = d['close'].iloc[-21:-1].max()

        breakout = (
            latest['close'] > resistance_20d * R['breakout_pct'] and
            latest['volume'] > latest['volume_sma'] * R['breakout_vol_mult']
        )
        near_ema20 = abs(latest['close'] - latest['ema_20']) / latest['ema_20'] < R['pullback_band']
        pullback = (
            near_ema20 and
            latest['close'] > prev['close'] and
            latest['volume'] > latest['volume_sma'] * R['pullback_vol_mult']
        )
        body_now  = abs(latest['close'] - latest['open'])
        body_prev = abs(prev['close']   - prev['open'])
        engulfing = (
            latest['close'] > latest['open'] and
            prev['close']   < prev['open']   and
            latest['close'] > prev['open']   and
            latest['open']  < prev['close']  and
            body_now > body_prev * 0.7
        )
        rsi_divergence = (
            latest['close'] < prev2['close'] and
            latest['rsi_9'] > prev2['rsi_9'] and
            latest['rsi_9'] < 60
        )
        stoch_cross = (
            prev['stoch_k']   < 35 and
            latest['stoch_k'] > latest['stoch_d'] and
            prev['stoch_k']   < prev['stoch_d']
        )
        momentum_burst = (
            latest['rsi']    > 55 and
            prev['rsi']      < 55 and
            latest['adx']    > prev['adx'] and
            latest['volume'] > latest['volume_sma'] * 1.1
        )
        bb_width_min = d['bb_width'].iloc[-6:-1].min()
        bb_squeeze = (
            prev['bb_width']   <= bb_width_min * 1.05 and
            latest['bb_width']  > prev['bb_width'] and
            latest['close']     > latest['bb_middle'] and
            latest['volume']    > latest['volume_sma']
        )
        ema_cross = (
            prev['ema_12']   <= prev['ema_20'] and
            latest['ema_12']  > latest['ema_20'] and
            latest['close']   > latest['ema_50']
        )
        cmf_accum = (
            float(latest['cmf']) > 0.08 and
            latest['close']       > latest['ema_50'] and
            latest['volume']      > latest['volume_sma'] * 0.9
        )

        signals = {
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
        active = [k for k, v in signals.items() if v]
        return active, (active[0] if active else None)

    # ── Midcap patterns (5 patterns tuned for smaller, more volatile names) ──
    def _midcap_patterns(self, d, latest, prev, prev2, R):
        """
        Philosophy:
        - Require stronger volume confirmation (midcaps fake breakouts more)
        - 50-bar resistance breakout > 20-bar (midcaps need wider structure)
        - 52-week high breakout = very high conviction institutional signal
        - Gap-up continuation captures earnings/news-driven moves in midcaps
        - HH/HL structure + CMF confirms clean uptrend with accumulation
        """
        # 1. Volume-confirmed breakout of 50-bar (10-week) resistance
        resistance_50d = d['close'].iloc[-51:-1].max()
        mc_volume_breakout = (
            latest['close']  > resistance_50d * 1.008 and
            latest['volume'] > latest['volume_sma'] * R['breakout_vol_mult'] and
            latest['adx']    > 20
        )

        # 2. 52-week high breakout with institutional-grade volume (2x avg)
        high_52w = d['close'].iloc[-252:-1].max() if len(d) >= 252 else d['close'].iloc[:-1].max()
        mc_52w_breakout = (
            latest['close']  > high_52w * 1.01 and
            latest['volume'] > latest['volume_sma'] * 2.0 and
            latest['rsi']    > 55
        )

        # 3. EMA-9/20 cluster compression then expansion (coiled spring)
        ema_spread      = abs(latest['ema_9'] - latest['ema_20']) / latest['ema_20']
        prev_ema_spread = abs(prev['ema_9']   - prev['ema_20'])   / prev['ema_20']
        mc_ema_cluster = (
            prev_ema_spread  < 0.008 and
            ema_spread        > prev_ema_spread and
            latest['close']   > latest['ema_20'] and
            latest['volume']  > latest['volume_sma'] * 1.3
        )

        # 4. Gap-up continuation (news/earnings-driven; midcaps gap more decisively)
        gap_pct = (float(latest['open']) - float(prev['close'])) / float(prev['close'])
        mc_gap_continuation = (
            gap_pct          > 0.015 and
            latest['close']  > latest['open'] * 0.995 and
            latest['volume'] > latest['volume_sma'] * 1.5 and
            latest['rsi']    < 75
        )

        # 5. Three-bar HH/HL structure with CMF accumulation
        mc_hh_hl_accumulation = (
            latest['high']  > prev['high'] and
            latest['low']   > prev['low'] and
            prev['high']    > prev2['high'] and
            prev['low']     > prev2['low'] and
            float(latest['cmf']) > 0.05 and
            latest['volume'] > latest['volume_sma'] * 0.9
        )

        signals = {
            'mc_volume_breakout':    mc_volume_breakout,
            'mc_52w_breakout':       mc_52w_breakout,
            'mc_ema_cluster':        mc_ema_cluster,
            'mc_gap_continuation':   mc_gap_continuation,
            'mc_hh_hl_accumulation': mc_hh_hl_accumulation,
        }
        active = [k for k, v in signals.items() if v]
        return active, (active[0] if active else None)

    # ── Signal scoring (used by replacement logic) ────────────────────────────
    @staticmethod
    def score_signal(details: dict) -> float:
        """
        Composite score for comparing signals head-to-head.
        score = R:R × (confidence / 9) × cap_bonus
        cap_bonus: largecap=1.0, midcap=0.85 (liquidity haircut)
        """
        rr        = details.get('risk_reward_ratio', 1.5)
        conf      = details.get('confidence', 1) / 9
        cap_bonus = 0.85 if details.get('cap_type') == 'MIDCAP' else 1.0
        return rr * conf * cap_bonus

    # ── Market regime ─────────────────────────────────────────────────────────
    @staticmethod
    def classify_market_regime(nifty_df):
        if nifty_df is None or len(nifty_df) < 200:
            return 'NEUTRAL'
        from technical_indicators import TechnicalIndicators
        tech  = TechnicalIndicators()
        ema50 = tech.calculate_ema(nifty_df['close'], 50)
        s200  = tech.calculate_sma(nifty_df['close'], 200)
        c, e, s = nifty_df['close'].iloc[-1], ema50.iloc[-1], s200.iloc[-1]
        if c > e and e > s:   return 'BULL'
        elif c < e and e < s: return 'BEAR'
        return 'NEUTRAL'
