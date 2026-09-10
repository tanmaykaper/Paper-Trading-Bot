# technical_indicators.py  ── v2  (MEASUREMENT LAYER)
# ─────────────────────────────────────────────────────────────────────────────
# Every public method from v1 keeps its exact name and signature, so
# swing_trading_bot.py, alpha_engine.py and signal_generator.py import this
# unchanged. What changed is *correctness* and *precision* of the numbers
# those callers already trust:
#
#   • RSI / ATR / ADX now use Wilder's recursive smoothing (alpha = 1/period),
#     which is the definition every published threshold ("ADX > 25 = trending",
#     "RSI 30/70") is calibrated against. v1 used simple rolling means, which
#     produce a materially different, faster-reacting series — so a filter
#     written as `adx < 12` was rejecting/accepting a different population of
#     bars than intended.
#   • ADX now applies Wilder's directional-movement exclusivity rule: on an
#     outside bar, ONLY the larger of (up-move, down-move) counts. v1 credited
#     both simultaneously, which inflates +DI and -DI together on exactly the
#     high-range days that matter most, blurring trend-vs-chop discrimination.
#   • OBV now leaves volume unsigned on unchanged closes (Granville's rule)
#     instead of counting a flat day as accumulation.
#   • CMF and Stochastic are now division-safe on zero-range bars (a limit-up
#     NSE circuit day has high == low, which produced inf/NaN and silently
#     poisoned every downstream rolling window).
#
# New estimators added for the adaptive barrier geometry in
# signal_generator.py v4 — each one exists to answer a specific question the
# v1 indicator set structurally could not:
#
#   yang_zhang_volatility   How far can this name plausibly travel in H days?
#   efficiency_ratio        Is this trend clean, or is it noise a tight stop
#                           will be shaken out by?
#   recent_swing_low/high   Where is the price level that actually invalidates
#                           the setup (as opposed to an arbitrary ATR multiple)?
#   relative_strength       Is this name leading or lagging the index?
#   gap_statistics          How much overnight gap risk does an EOD-only stop
#                           actually carry on this scrip?
#   median_traded_value     Is there enough real liquidity for the ATR/stop
#                           estimates to mean anything?
#
# All estimators are vectorised, strictly backward-looking (bar t uses only
# bars <= t), and NaN-tolerant.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Smallest meaningful daily volatility, as a fraction of price. Guards every
# "divide by volatility" path against a near-zero denominator produced by a
# suspended/illiquid scrip printing an identical close for days.
MIN_DAILY_VOL = 0.004     # 0.4% — below this, a daily-bar swing system has no edge to size against
MAX_DAILY_VOL = 0.150     # 15% — above this the series is almost certainly corrupt, not volatile


def _rma(series, period):
    """
    Wilder's recursive moving average: RMA_t = RMA_{t-1} + (x_t - RMA_{t-1})/n.
    Equivalent to an EWMA with alpha = 1/n, adjust=False. This is the smoothing
    RSI, ATR and ADX are *defined* with — a plain rolling mean gives a
    different series with different threshold semantics.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _safe_div(numer, denom, fill=np.nan):
    """Elementwise division that returns `fill` where the denominator is ~0."""
    denom = pd.Series(denom).astype(float)
    numer = pd.Series(numer).astype(float)
    out = numer / denom.where(denom.abs() > 1e-12)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill) if fill is not np.nan else \
        out.replace([np.inf, -np.inf], np.nan)


class TechnicalIndicators:

    # ═════════════════════════════════════════════════════════════════════════
    # MOVING AVERAGES
    # ═════════════════════════════════════════════════════════════════════════
    @staticmethod
    def calculate_ema(data, period):
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(data, period):
        return data.rolling(window=period).mean()

    # ═════════════════════════════════════════════════════════════════════════
    # OSCILLATORS
    # ═════════════════════════════════════════════════════════════════════════
    @staticmethod
    def calculate_rsi(data, period=14):
        """
        Wilder RSI. A flat/rising-only window (loss == 0) correctly pins to
        100 instead of producing inf/NaN — that case is real on a stock that
        has closed up every session of the window, and v1 returned NaN there,
        which silently failed the `rsi_lo < rsi < rsi_hi` band check and
        discarded the strongest momentum bars in the universe.
        """
        delta = data.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = _rma(gain, period)
        avg_loss = _rma(loss, period)

        rs = avg_gain / avg_loss.where(avg_loss > 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # loss == 0 with gain > 0 -> pure uptrend -> 100 ; both zero -> 50 (no information)
        rsi = rsi.where(avg_loss > 1e-12, np.where(avg_gain > 1e-12, 100.0, 50.0))
        return rsi

    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}

    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std(ddof=1)
        return {'upper': sma + std * std_dev, 'middle': sma, 'lower': sma - std * std_dev}

    @staticmethod
    def calculate_stochastic(high, low, close, period=14):
        """Zero-range windows (circuit-locked scrip) yield 50 — 'no information'
        — rather than inf, which would propagate through %D and every filter
        downstream of it."""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        rng = (highest_high - lowest_low)
        k_percent = 100.0 * (close - lowest_low) / rng.where(rng > 1e-12)
        k_percent = k_percent.where(rng > 1e-12, 50.0)
        return {'k': k_percent, 'd': k_percent.rolling(window=3).mean()}

    @staticmethod
    def calculate_williams_r(high, low, close, period=14):
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        rng = (highest_high - lowest_low)
        wr = -100.0 * (highest_high - close) / rng.where(rng > 1e-12)
        return wr.where(rng > 1e-12, -50.0)

    # ═════════════════════════════════════════════════════════════════════════
    # VOLATILITY / RANGE
    # ═════════════════════════════════════════════════════════════════════════
    @staticmethod
    def calculate_true_range(high, low, close):
        """max(H-L, |H-C_prev|, |L-C_prev|); first bar falls back to H-L."""
        prev_close = close.shift()
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        tr.iloc[0] = float(high.iloc[0] - low.iloc[0]) if len(high) else np.nan
        return tr

    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Wilder ATR (RMA of True Range), not an SMA of True Range."""
        tr = TechnicalIndicators.calculate_true_range(high, low, close)
        return _rma(tr, period)

    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Wilder ADX. See calculate_dmi for the full +DI/-DI/ADX triplet."""
        return TechnicalIndicators.calculate_dmi(high, low, close, period)['adx']

    @staticmethod
    def calculate_dmi(high, low, close, period=14):
        """
        Wilder's Directional Movement Index, implemented to spec.

        The rule v1 omitted: directional movement is EXCLUSIVE. On a bar where
        both the high extends up and the low extends down (an outside/expansion
        bar), only the larger move is credited; the other is zero. Crediting
        both — as v1 did — lifts +DI and -DI in lockstep on precisely the
        expansion bars where directional information is richest, which drags
        DX toward zero and makes ADX read "no trend" on genuine breakouts.
        """
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move.fillna(0.0), 0.0),
            index=high.index, dtype=float)
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move.fillna(0.0), 0.0),
            index=high.index, dtype=float)

        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        atr_safe = atr.where(atr > 1e-12)

        plus_di = 100.0 * _rma(plus_dm, period) / atr_safe
        minus_di = 100.0 * _rma(minus_dm, period) / atr_safe

        di_sum = (plus_di + minus_di)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum > 1e-12)
        adx = _rma(dx.fillna(0.0), period)

        return {'plus_di': plus_di, 'minus_di': minus_di, 'adx': adx, 'dx': dx}

    @staticmethod
    def parkinson_volatility(high, low, period=20):
        """
        High-low range estimator. ~5x more statistically efficient than a
        close-to-close estimator at the same sample size, because it uses the
        whole day's path rather than one point of it — which is exactly what a
        20-bar window on daily data is short of.
        Returns DAILY volatility as a fraction (log units).
        """
        rng = np.log(high / low.where(low > 1e-12)) ** 2
        return np.sqrt(rng.rolling(period).mean() / (4.0 * np.log(2.0)))

    @staticmethod
    def garman_klass_volatility(open_, high, low, close, period=20):
        """Adds the open-close body to the Parkinson range term."""
        hl = 0.5 * np.log(high / low.where(low > 1e-12)) ** 2
        co = (2.0 * np.log(2.0) - 1.0) * np.log(close / open_.where(open_ > 1e-12)) ** 2
        return np.sqrt((hl - co).rolling(period).mean().clip(lower=0.0))

    @staticmethod
    def yang_zhang_volatility(open_, high, low, close, period=20):
        """
        Yang-Zhang (2000): the only common OHLC estimator that is
        simultaneously drift-independent AND handles the overnight jump.

        That second property is the reason it is here rather than Parkinson
        alone. NSE cash equities do not trade overnight; a material share of a
        swing position's total variance arrives as a gap at 09:15, which a
        range-only estimator cannot see. Under-measuring that component is what
        makes an ATR-derived stop look "2 ATR wide" while actually sitting
        inside one overnight move.

        Returns DAILY volatility as a fraction (log units).
        """
        n = int(period)
        log_oc = np.log(open_ / close.shift())          # overnight jump
        log_co = np.log(close / open_)                   # intraday drift
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)

        sigma_o2 = log_oc.rolling(n).var(ddof=1)
        sigma_c2 = log_co.rolling(n).var(ddof=1)
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)   # Rogers-Satchell
        sigma_rs2 = rs.rolling(n).mean()

        k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
        var_yz = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
        return np.sqrt(var_yz.clip(lower=0.0))

    @staticmethod
    def daily_volatility_fraction(df, period=20, atr_period=14):
        """
        The single volatility number the barrier geometry is built on: a
        conservative blend of the two independent estimators, expressed as a
        fraction of price per day.

        max(ATR/price, Yang-Zhang) is deliberate rather than an average. These
        two disagree in informative ways — ATR misses nothing intraday but
        smooths gaps into the average, YZ prices the gap explicitly. Taking the
        larger means the stop is sized against whichever risk the data is
        currently showing, and never against the flattering one.

        Replaces v1 signal_generator's `if atr < 0.5% of price: atr = 1.2% of
        price` heuristic, which silently multiplied the risk estimate by up to
        2.4x on quiet names and therefore mis-sized every position on them.
        """
        close = df['close']
        atr = TechnicalIndicators.calculate_atr(df['high'], df['low'], close, atr_period)
        atr_frac = (atr / close.where(close > 1e-12))
        yz = TechnicalIndicators.yang_zhang_volatility(df['open'], df['high'], df['low'], close, period)
        blended = pd.concat([atr_frac, yz], axis=1).max(axis=1)
        return blended.clip(lower=MIN_DAILY_VOL, upper=MAX_DAILY_VOL)

    # ═════════════════════════════════════════════════════════════════════════
    # TREND QUALITY / STRUCTURE
    # ═════════════════════════════════════════════════════════════════════════
    @staticmethod
    def efficiency_ratio(close, period=20):
        """
        Kaufman Efficiency Ratio = |net displacement| / total path length,
        over `period` bars. 1.0 = a straight line; ~0 = pure round-tripping.

        This is the single most directly actionable number for stop placement.
        A given ATR buys far more protection on a high-ER name (price spends
        its range going somewhere) than on a low-ER name (price spends its
        range oscillating through the stop). signal_generator v4 widens the
        stop as ER falls instead of applying one fixed ATR multiple to both.
        """
        net = close.diff(period).abs()
        path = close.diff().abs().rolling(period).sum()
        return (net / path.where(path > 1e-12)).clip(0.0, 1.0)

    @staticmethod
    def recent_swing_low(low, lookback=30, confirm=2):
        """
        Most recent confirmed swing low: a bar whose low is the minimum of the
        window [i-confirm, i+confirm]. Strictly backward-looking — only bars
        already closed at evaluation time are inspected (the newest candidate
        is `confirm` bars old, because that is when confirmation actually
        exists), so this is safe inside a walk-forward backtest.

        Returns (price, bars_ago) or (None, None).
        """
        n = len(low)
        if n < (2 * confirm + 2):
            return None, None
        vals = low.to_numpy(dtype=float)
        start = max(confirm, n - 1 - lookback)
        for i in range(n - 1 - confirm, start - 1, -1):
            window = vals[i - confirm: i + confirm + 1]
            if np.isnan(window).any():
                continue
            if vals[i] <= window.min() + 1e-12:
                return float(vals[i]), int(n - 1 - i)
        return None, None

    @staticmethod
    def recent_swing_high(high, lookback=30, confirm=2):
        """Mirror of recent_swing_low."""
        n = len(high)
        if n < (2 * confirm + 2):
            return None, None
        vals = high.to_numpy(dtype=float)
        start = max(confirm, n - 1 - lookback)
        for i in range(n - 1 - confirm, start - 1, -1):
            window = vals[i - confirm: i + confirm + 1]
            if np.isnan(window).any():
                continue
            if vals[i] >= window.max() - 1e-12:
                return float(vals[i]), int(n - 1 - i)
        return None, None

    @staticmethod
    def detect_support_resistance(high, low, lookback=10):
        return {'resistance': high.tail(lookback).max(), 'support': low.tail(lookback).min()}

    @staticmethod
    def relative_strength(close, benchmark_close, period=20):
        """
        Excess log return over the benchmark across `period` bars, as a
        fraction. Positive = outperforming Nifty.

        Indices are aligned on position (both series are daily NSE bars ending
        on the same session), and the shorter history governs — a benchmark
        with fewer bars degrades the measurement window rather than raising.
        """
        n = min(len(close), len(benchmark_close))
        if n <= period:
            return np.nan
        a = float(close.iloc[-1]) / float(close.iloc[-1 - period])
        b = float(benchmark_close.iloc[-1]) / float(benchmark_close.iloc[-1 - period])
        if a <= 0 or b <= 0:
            return np.nan
        return float(np.log(a) - np.log(b))

    @staticmethod
    def calculate_momentum(close, period=10):
        return close.diff(period)

    # ═════════════════════════════════════════════════════════════════════════
    # VOLUME / FLOW
    # ═════════════════════════════════════════════════════════════════════════
    @staticmethod
    def calculate_volume_sma(volume, period=10):
        return volume.rolling(window=period).mean()

    @staticmethod
    def calculate_obv(close, volume):
        """Granville OBV: unchanged closes contribute zero, not +volume."""
        direction = np.sign(close.diff().fillna(0.0))
        return (volume * direction).cumsum()

    @staticmethod
    def calculate_cmf(high, low, close, volume, period=20):
        """Chaikin Money Flow, safe on zero-range (circuit-locked) bars."""
        rng = (high - low)
        mfm = ((close - low) - (high - close)) / rng.where(rng > 1e-12)
        mfv = mfm.fillna(0.0) * volume
        vol_sum = volume.rolling(period).sum()
        return mfv.rolling(period).sum() / vol_sum.where(vol_sum > 1e-12)

    @staticmethod
    def median_traded_value(close, volume, period=20):
        """
        Rolling median of daily traded value in rupees. Median rather than mean
        because a single block/bulk-deal print routinely triples the mean on a
        midcap and would wave an otherwise-illiquid name through a liquidity
        gate.
        """
        return (close * volume).rolling(period).median()

    # ═════════════════════════════════════════════════════════════════════════
    # RISK / EXECUTION CHARACTER
    # ═════════════════════════════════════════════════════════════════════════
    @staticmethod
    def gap_statistics(open_, close, lookback=60):
        """
        Overnight gap profile: {mean_abs, p90_abs, down_gap_p90, n}, all as
        fractions of the prior close.

        Why this belongs in the risk model rather than the indicator trivia
        pile: this bot places EOD stop levels on a market that is closed for
        ~17.5 hours a day. A stop 3% below entry does not fill at -3% when the
        scrip opens -6%; it fills at -6%. Expected loss per stop-out is
        therefore stop_distance + E[adverse gap beyond it], and a name with a
        fat down-gap tail needs either a wider stop or a smaller position for
        the *realised* loss to match the *intended* loss.
        """
        prev_close = close.shift()
        gaps = (open_ / prev_close.where(prev_close > 1e-12) - 1.0).tail(lookback).dropna()
        if len(gaps) < 10:
            return {'mean_abs': 0.004, 'p90_abs': 0.010, 'down_gap_p90': 0.010, 'n': int(len(gaps))}
        down = (-gaps.clip(upper=0.0))
        return {
            'mean_abs': float(gaps.abs().mean()),
            'p90_abs': float(gaps.abs().quantile(0.90)),
            'down_gap_p90': float(down.quantile(0.90)),
            'n': int(len(gaps)),
        }

    @staticmethod
    def keltner_channels(df, ema_period=20, atr_period=14, mult=2.0):
        mid = TechnicalIndicators.calculate_ema(df['close'], ema_period)
        atr = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'], atr_period)
        return {'middle': mid, 'upper': mid + mult * atr, 'lower': mid - mult * atr}

    @staticmethod
    def detect_divergence(prices, rsi, lookback=5):
        """
        Bullish divergence: price makes a lower low over the recent window
        while RSI makes a higher low. Narrow try/except (v1 used a bare
        `except:`, which also swallowed KeyboardInterrupt and genuine
        programming errors as "no divergence").
        """
        try:
            if len(prices) < 2 * lookback or len(rsi) < 2 * lookback:
                return False
            recent_price_low = float(prices.iloc[-lookback:].min())
            recent_rsi_low = float(rsi.iloc[-lookback:].min())
            prev_price_low = float(prices.iloc[-2 * lookback:-lookback].min())
            prev_rsi_low = float(rsi.iloc[-2 * lookback:-lookback].min())
            if any(np.isnan(v) for v in (recent_price_low, recent_rsi_low, prev_price_low, prev_rsi_low)):
                return False
            return (recent_price_low < prev_price_low) and (recent_rsi_low > prev_rsi_low)
        except (KeyError, IndexError, ValueError, TypeError):
            return False
