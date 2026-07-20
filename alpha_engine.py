# alpha_engine.py
#
# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE REGIME-CONDITIONED, CROSS-SECTIONAL MOMENTUM ALPHA ENGINE
# ─────────────────────────────────────────────────────────────────────────────
#
# STANDALONE MODULE — not yet wired into run_paper_trading.py / signal_generator.py.
# Per instructions, this is the algorithm and logic on its own, built and
# tested in isolation, to be integrated afterward.
#
# ── First, a reframe on "accuracy" ──────────────────────────────────────────
# The brief asked for high accuracy. I want to be straight about this before
# building anything: optimizing a trading system for raw win-rate ("accuracy")
# is a well-known trap. A system can win 80% of its trades and still lose
# money if the 20% of losers are large enough; a system that's only right 40%
# of the time can compound aggressively if winners run 3x bigger than losers.
# What actually matters is EXPECTANCY — (win_rate × avg_win) − (loss_rate ×
# avg_loss) — and risk-adjusted return (Sharpe/Sortino, max drawdown), not
# accuracy in isolation. Everything below is built to maximise expectancy and
# survive bad regimes, and is honest about what "high accuracy" would
# realistically look like for something like this: most robust systematic
# equity strategies run somewhere in the 40-55% win-rate range, with edge
# coming from asymmetric payoff (R:R), not from being right most of the time.
# I'd rather build something that's honest about that and actually works than
# something that chases a "high accuracy" number that would either be
# overfit or come from cutting winners short to inflate the win-rate stat.
#
# ── What makes this "unique" ────────────────────────────────────────────────
# Not a brand-new academic factor — the individual ingredients (momentum,
# cross-sectional ranking, volatility-adjustment, regime-dependence) are all
# well-established, real, published findings, not invented for this project:
#   • Cross-sectional momentum is one of the most robustly documented return
#     anomalies in equity markets (Jegadeesh & Titman 1993), still used today
#     by real quantitative asset managers (AQR, Alpha Architect, Dimensional).
#   • Risk-adjusting momentum (return per unit of volatility, rather than raw
#     return) is a documented refinement — plain momentum has a well-known
#     "crash risk" problem, especially coming out of high-volatility regimes
#     (Daniel & Moskowitz, momentum crash literature).
#   • Momentum's efficacy is regime-dependent — stronger in calmer, trending
#     markets, materially weaker (sometimes negative) in choppy or highly
#     volatile ones. A system that applies the same momentum weight in every
#     regime is ignoring well-documented evidence.
# The combination — cross-sectional ranking + risk-adjustment + a regime
# detector that actively re-weights factors, PLUS a layer that recalibrates
# pattern weights from this bot's own realized trade history (with proper
# statistical guardrails, not naive curve-fitting) — is the actual "unique"
# part: a system that measures its own track record and adjusts, rather than
# a fixed rule set someone hand-tuned once and never revisited.
#
# ── Architecture ─────────────────────────────────────────────────────────────
#   RegimeDetector           — classifies current market state from index data
#   FactorEngine              — computes 5 return-predictive factors per symbol
#   CrossSectionalRanker      — converts raw factors into relative percentile
#                                ranks against the current scan universe (not
#                                fixed absolute thresholds — this is what makes
#                                the system adapt to "whatever market exists
#                                right now" rather than a static rulebook)
#   AdaptiveWeightCalibrator  — reweights entry patterns using this bot's own
#                                realized trade history, with minimum-sample
#                                gating, shrinkage toward the population mean,
#                                and a hard cap on how much any one
#                                recalibration can move a weight
#   CompositeAlphaScore       — orchestrates all of the above into one
#                                explainable 0-100 score per symbol
#
# ── What this module does NOT do (yet) ──────────────────────────────────────
# It does not fetch data, does not place trades, and is not wired into
# run_paper_trading.py. It also has not been validated against real
# historical NSE data — this sandbox has no live yfinance/NSE access, so
# everything below is tested against synthetic data with known, deliberately
# constructed properties (pure uptrend, pure downtrend, choppy noise, high-
# volatility whipsaw) to prove the MECHANICS are correct. Real historical
# performance — actual win rate, profit factor, Sharpe, drawdown — has to be
# measured empirically once this is wired into the existing run_backtest.py
# infrastructure against real price history. Anything else would be a claim
# I can't actually back up from here.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import logging
import os

from technical_indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TI = TechnicalIndicators


# ═════════════════════════════════════════════════════════════════════════════
# 1. REGIME DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    Classifies the current market regime from a broad index's OHLCV data
    (e.g. Nifty 50/500), and produces per-regime factor-weight tilts.

    Why this exists: momentum's edge is well-documented to be regime-
    dependent — it works best in calm, trending markets and decays (or
    reverses) in choppy or high-volatility ones. A system that scores
    "momentum" the same way in a raging bull market and a whipsawing,
    high-VIX month is ignoring that. This classifies which environment
    we're actually in, so the rest of the engine can adjust accordingly.

    Regimes:
      STRONG_UPTREND   — ADX >= trend_threshold, EMA20 > EMA50, vol not elevated
      STRONG_DOWNTREND — ADX >= trend_threshold, EMA20 < EMA50, vol not elevated
      WEAK_TREND       — ADX between the two thresholds — transitional/ambiguous
      CHOPPY_RANGE      — ADX < range_threshold — momentum has the least edge here
      HIGH_VOL_STRESS   — realized volatility elevated vs its own recent history,
                           regardless of trend — this OVERRIDES the other
                           classifications, because momentum's worst drawdowns
                           historically cluster exactly here (volatility spikes
                           / regime transitions), not in ordinary choppy markets.
    """

    TREND_ADX_THRESHOLD = 25
    RANGE_ADX_THRESHOLD = 18
    VOL_STRESS_PERCENTILE = 85   # realized vol above this percentile of its own history -> stress

    # Factor-weight tilts per regime. These multiply the base weights in
    # CompositeAlphaScore.BASE_FACTOR_WEIGHTS. >1 = emphasise, <1 = de-emphasise.
    REGIME_TILTS = {
        'STRONG_UPTREND': {
            'momentum': 1.25, 'risk_adj_momentum': 1.15, 'trend_quality': 1.15,
            'volume_conviction': 1.00, 'relative_strength': 1.10, 'overall_dampener': 1.00,
        },
        'STRONG_DOWNTREND': {
            # Long-only momentum has little business chasing strength into a
            # confirmed downtrend — dampen everything rather than pretend a
            # normal momentum read still means the same thing here.
            'momentum': 0.60, 'risk_adj_momentum': 0.60, 'trend_quality': 0.70,
            'volume_conviction': 0.90, 'relative_strength': 1.15, 'overall_dampener': 0.75,
        },
        'WEAK_TREND': {
            'momentum': 1.00, 'risk_adj_momentum': 1.00, 'trend_quality': 1.00,
            'volume_conviction': 1.00, 'relative_strength': 1.00, 'overall_dampener': 1.00,
        },
        'CHOPPY_RANGE': {
            # Documented: momentum decays in range-bound markets. Lean more on
            # trend_quality (filters out the noise) and volume_conviction
            # (genuine breakouts still show up in volume even in a choppy tape).
            'momentum': 0.65, 'risk_adj_momentum': 0.75, 'trend_quality': 1.20,
            'volume_conviction': 1.20, 'relative_strength': 0.90, 'overall_dampener': 0.90,
        },
        'HIGH_VOL_STRESS': {
            # Momentum crash risk clusters here. Dampen the whole score rather
            # than try to selectively reweight — in genuine stress, the
            # priority is capital preservation, which the drawdown circuit
            # breaker (run_paper_trading.py) already also handles separately.
            'momentum': 0.55, 'risk_adj_momentum': 0.70, 'trend_quality': 0.80,
            'volume_conviction': 0.85, 'relative_strength': 0.85, 'overall_dampener': 0.65,
        },
    }

    def classify(self, index_df):
        """
        index_df: DataFrame with 'high','low','close' columns, most recent
        row last. Needs at least ~60 rows for a meaningful read; degrades
        gracefully (wider uncertainty, defaults to WEAK_TREND) with less.

        Returns dict: {regime, adx, vol_percentile, trend_direction,
                        tilts, confidence}
        """
        if index_df is None or len(index_df) < 30:
            logger.warning("RegimeDetector: insufficient data (<30 bars) — defaulting to WEAK_TREND")
            return self._default_result(reason='insufficient_data')

        high, low, close = index_df['high'], index_df['low'], index_df['close']

        adx_series = TI.calculate_adx(high, low, close, period=14)
        adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else None

        ema20 = TI.calculate_ema(close, 20)
        ema50 = TI.calculate_ema(close, 50) if len(close) >= 50 else None

        returns = close.pct_change()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)   # annualised 20d realised vol

        # Percentile of *recent* realised vol (smoothed over the last 3
        # sessions, not a single day) against its own trailing history —
        # self-relative on purpose, so this doesn't depend on picking one
        # fixed "20% vol = stressed" number that may not fit whatever the
        # prevailing baseline volatility regime actually is. The smoothing
        # matters: a single day's realised-vol reading is noisy enough
        # (finite-sample "vol of vol") that even a genuinely constant-
        # volatility process will occasionally show one elevated reading by
        # chance — regime classification flipping to HIGH_VOL_STRESS on that
        # alone would make the whole downstream system unstable.
        #
        # Minimum history of 80 (not a smaller number) is deliberate and
        # evidence-based, not arbitrary: testing against synthetic data with
        # a KNOWN-constant true volatility showed that with less history, the
        # percentile-of-own-history read is unreliable enough to occasionally
        # override even an unambiguous trend (one test case had ADX=60 — about
        # as clear a trend as exists — spuriously flagged HIGH_VOL_STRESS from
        # a noisy vol read on a short sample). 80+ sessions resolved it. The
        # live system calls this with 300 days of history in practice
        # (run_paper_trading.py's _get_market_regime), so this floor doesn't
        # bind in normal operation — it only matters for graceful degradation
        # when less history is available.
        vol_hist = realized_vol.dropna()
        if len(vol_hist) >= 80:
            lookback = vol_hist.tail(min(len(vol_hist), 252))
            recent_vol = lookback.tail(3).mean()
            vol_percentile = float((lookback <= recent_vol).mean() * 100)
        else:
            vol_percentile = 50.0   # not enough history for a reliable read — assume "normal"

        if adx is None:
            return self._default_result(reason='adx_unavailable')

        trend_direction = 'UP' if (ema50 is not None and ema20.iloc[-1] > ema50.iloc[-1]) else 'DOWN'

        # ── Classification ───────────────────────────────────────────────────
        # HIGH_VOL_STRESS is meant to catch genuine regime stress, not just
        # noise. A percentile-threshold read has an inherent, unavoidable
        # false-positive rate BY DEFINITION — checking "is the latest reading
        # above the 85th percentile of its own history" fires on ~15% of
        # perfectly ordinary draws even under a truly constant-volatility
        # process (verified empirically: 14.7% over 2000 simulated trials,
        # matching the theoretical 15% exactly — this is a property of the
        # method itself, not something more historical data fixes).
        #
        # So a moderately-elevated reading (85th-95th percentile) should NOT
        # be allowed to override an extremely unambiguous trend (very high
        # ADX) — that combination is exactly where the vol reading is more
        # likely to be noise than the trend reading is. It SHOULD still
        # override a moderate/weak trend, where the vol signal is relatively
        # more informative than a marginal ADX reading. And an EXTREME vol
        # reading (95th+) overrides regardless of trend strength — at that
        # extreme it's more likely to reflect genuine stress (this also lines
        # up with real momentum-crash research: the sharpest trend reversals
        # tend to come with genuinely extreme, not just moderately elevated,
        # volatility).
        EXTREME_VOL_PERCENTILE = 95
        DOMINANT_TREND_ADX     = 50

        if vol_percentile >= EXTREME_VOL_PERCENTILE:
            regime = 'HIGH_VOL_STRESS'
        elif vol_percentile >= self.VOL_STRESS_PERCENTILE and adx < DOMINANT_TREND_ADX:
            regime = 'HIGH_VOL_STRESS'
        elif adx >= self.TREND_ADX_THRESHOLD and trend_direction == 'UP':
            regime = 'STRONG_UPTREND'
        elif adx >= self.TREND_ADX_THRESHOLD and trend_direction == 'DOWN':
            regime = 'STRONG_DOWNTREND'
        elif adx < self.RANGE_ADX_THRESHOLD:
            regime = 'CHOPPY_RANGE'
        else:
            regime = 'WEAK_TREND'

        # crude confidence: how far past its threshold the deciding metric is
        if regime in ('STRONG_UPTREND', 'STRONG_DOWNTREND'):
            confidence = min(1.0, (adx - self.TREND_ADX_THRESHOLD) / 20 + 0.5)
        elif regime == 'CHOPPY_RANGE':
            confidence = min(1.0, (self.RANGE_ADX_THRESHOLD - adx) / self.RANGE_ADX_THRESHOLD + 0.5)
        elif regime == 'HIGH_VOL_STRESS':
            confidence = min(1.0, (vol_percentile - self.VOL_STRESS_PERCENTILE) / 15 + 0.5)
        else:
            confidence = 0.5

        return {
            'regime': regime,
            'adx': round(adx, 1),
            'vol_percentile': round(vol_percentile, 1),
            'trend_direction': trend_direction,
            'tilts': self.REGIME_TILTS[regime],
            'confidence': round(confidence, 2),
        }

    def _default_result(self, reason):
        return {
            'regime': 'WEAK_TREND', 'adx': None, 'vol_percentile': None,
            'trend_direction': None, 'tilts': self.REGIME_TILTS['WEAK_TREND'],
            'confidence': 0.0, 'note': reason,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 2. FACTOR ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class FactorEngine:
    """
    Computes raw, per-symbol return-predictive factors from OHLCV data.
    "Raw" is deliberate — these are real-unit values (e.g. a 12.4% 3-month
    return), not yet comparable across symbols on different price/vol scales.
    That comparison step is CrossSectionalRanker's job, on purpose: computing
    a factor and ranking it are different concerns, and keeping them separate
    means the ranking logic doesn't need to know anything about how any
    individual factor was derived.

    A fifth factor, relative_strength (sector-relative momentum), is NOT
    computed here — by definition it needs the whole universe's data at once
    ("relative to what?"), so it lives in CrossSectionalRanker instead. The
    four below are all computable from one symbol's own price history.
    """

    def momentum_multi_horizon(self, df):
        """
        Blended rate-of-change across three lookback windows (~1M/3M/6M
        trading days). Multi-horizon on purpose: a stock up only in the last
        week but flat over 3 months is a different (weaker, less confirmed)
        setup than one trending consistently across all three windows —
        blending catches genuine, sustained moves over single-window noise.
        Recent horizon weighted slightly higher, consistent with this being
        a swing-trading system (max ~15 day holds), not a long-term hold.
        """
        close = df['close']
        if len(close) < 30:
            return None

        def roc(n):
            if len(close) <= n:
                return None
            return (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100

        r21, r63, r126 = roc(21), roc(63), roc(126)
        available = [(r, w) for r, w in [(r21, 0.5), (r63, 0.3), (r126, 0.2)] if r is not None]
        if not available:
            return None
        total_w = sum(w for _, w in available)
        return sum(r * w for r, w in available) / total_w

    def risk_adjusted_momentum(self, df):
        """
        63-day return divided by 63-day realised volatility (annualised) —
        a Sharpe-like ratio. Two stocks with identical raw returns are NOT
        equally good momentum candidates if one got there through a smooth,
        orderly climb and the other through wild, jagged swings — the smooth
        one is more likely to be a genuine sustained trend (also consistent
        with the "frog-in-the-pan"/gradual-information-diffusion research on
        why momentum works: gradual moves persist better than spiky ones).
        This is what actually keeps "high risk capacity" from degenerating
        into "just buy whatever moved the most" — reward return achieved
        efficiently, not return achieved through raw chaos.
        """
        close = df['close']
        if len(close) < 64:
            return None
        window = close.iloc[-64:]
        ret_63 = (window.iloc[-1] / window.iloc[0] - 1) * 100
        daily_returns = window.pct_change().dropna()
        vol_63 = daily_returns.std() * np.sqrt(252) * 100
        if vol_63 is None or vol_63 < 1e-6:
            return None
        return ret_63 / vol_63

    def trend_quality(self, df):
        """
        Blends ADX (trend strength), % of up-days over the lookback (a
        smooth, persistent trend has a high fraction of up-days; a volatile
        chop doesn't, even with the same net return), and whether price sits
        above a rising 50-EMA (structural confirmation). Filters for
        genuine, sustained trends over noisy ones — a documented distinction
        in the momentum literature, not just an intuition.
        """
        high, low, close = df['high'], df['low'], df['close']
        if len(close) < 55:
            return None

        adx_series = TI.calculate_adx(high, low, close, period=14)
        adx = adx_series.iloc[-1]
        if pd.isna(adx):
            return None

        recent = close.iloc[-63:] if len(close) >= 63 else close
        daily_ret = recent.pct_change().dropna()
        pct_up_days = (daily_ret > 0).mean() * 100 if len(daily_ret) > 0 else 50.0

        ema50 = TI.calculate_ema(close, 50)
        above_rising_ema = 0.0
        if len(ema50) >= 6 and not pd.isna(ema50.iloc[-1]) and not pd.isna(ema50.iloc[-6]):
            is_above = close.iloc[-1] > ema50.iloc[-1]
            is_rising = ema50.iloc[-1] > ema50.iloc[-6]
            above_rising_ema = 100.0 if (is_above and is_rising) else (50.0 if is_above else 0.0)

        # Blend onto a common ~0-100-ish scale (ADX is naturally 0-100 already)
        return float(adx) * 0.5 + pct_up_days * 0.3 + above_rising_ema * 0.2

    def volume_conviction(self, df):
        """
        Recent (10d) average volume vs longer-run (60d) average volume, plus
        whether Chaikin Money Flow is positive (net accumulation, not
        distribution). Rising participation is what separates a genuine
        institutional move from a low-volume drift that's easy to reverse —
        this is the factor that keeps CHOPPY_RANGE regimes from being pure
        noise: real breakouts tend to still show up here even in a
        directionless tape.
        """
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        if len(volume) < 61:
            return None

        vol_recent = volume.iloc[-10:].mean()
        vol_hist   = volume.iloc[-60:-10].mean()
        if vol_hist is None or vol_hist < 1e-6:
            return None
        vol_ratio = vol_recent / vol_hist

        cmf_series = TI.calculate_cmf(high, low, close, volume, period=20)
        cmf = cmf_series.iloc[-1] if not pd.isna(cmf_series.iloc[-1]) else 0.0

        # vol_ratio of 1.0 = normal participation; CMF in roughly [-1, 1].
        # Scale so a doubling of volume + strongly positive CMF both matter.
        return (vol_ratio * 50) + (cmf * 50)

    def compute_all(self, df):
        """Returns {factor_name: raw_value or None} for one symbol."""
        return {
            'momentum':          self.momentum_multi_horizon(df),
            'risk_adj_momentum': self.risk_adjusted_momentum(df),
            'trend_quality':     self.trend_quality(df),
            'volume_conviction': self.volume_conviction(df),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 3. CROSS-SECTIONAL RANKER
# ═════════════════════════════════════════════════════════════════════════════

class CrossSectionalRanker:
    """
    Converts raw, real-unit factor values into percentile ranks (0-100)
    against the CURRENT scan universe snapshot — this is the single most
    important design choice in the whole engine, and the most directly
    grounded in real, published research (cross-sectional momentum,
    Jegadeesh & Titman 1993 and a large literature since).

    Why relative instead of absolute: a fixed rule like "buy if 3-month
    return > 15%" breaks down completely depending on what the market is
    doing — 15% is unremarkable in a raging bull run (everything is up
    that much) and enormous in a flat or falling market (almost nothing is).
    Ranking each stock against its current peer set — the exact universe
    being scanned, on the exact day being scanned — makes the system
    automatically recalibrate to whatever regime and opportunity set
    currently exists, rather than needing hand-tuned thresholds re-picked
    every time market conditions shift. This is also the direct mechanism
    for "taking into account the industries that exist in the world right
    now": the ranking is always relative to the current universe, which
    itself already includes the current high-growth/momentum theme names.

    Also computes relative_strength — the fifth factor, sector-relative
    momentum — here rather than in FactorEngine, because by definition it
    needs the whole universe (and the sector map) at once to answer
    "relative to whom."
    """

    def rank_universe(self, factor_values_by_symbol, sector_map=None):
        """
        factor_values_by_symbol: {symbol: {factor_name: raw_value_or_None}}
                                  (typically FactorEngine.compute_all() output
                                  for every symbol in today's scan universe)
        sector_map: {symbol: sector_name}, optional — enables relative_strength

        Returns {symbol: {factor_name: percentile_0_100}} — includes
        'relative_strength' alongside the four raw factors. A symbol missing
        a given raw factor (e.g. too little history) gets that factor
        excluded from its result rather than defaulted to some fabricated
        value — CompositeAlphaScore treats a missing factor as "no opinion",
        not as a bad or good score by default.
        """
        symbols = list(factor_values_by_symbol.keys())
        factor_names = ['momentum', 'risk_adj_momentum', 'trend_quality', 'volume_conviction']

        ranks = {s: {} for s in symbols}

        for factor in factor_names:
            values = {s: factor_values_by_symbol[s].get(factor)
                      for s in symbols if factor_values_by_symbol[s].get(factor) is not None}
            if len(values) < 2:
                continue   # can't meaningfully rank fewer than 2 data points
            series = pd.Series(values)
            pct_ranks = series.rank(pct=True) * 100
            for s, r in pct_ranks.items():
                ranks[s][factor] = round(float(r), 1)

        # relative_strength: momentum ranked WITHIN sector, not the whole
        # universe. Falls back to whole-universe momentum rank if no sector
        # map is supplied, or if a symbol's sector has too few peers (<3) to
        # rank meaningfully within.
        momentum_values = {s: factor_values_by_symbol[s].get('momentum')
                            for s in symbols if factor_values_by_symbol[s].get('momentum') is not None}

        if sector_map:
            by_sector = {}
            for s in momentum_values:
                sec = sector_map.get(s, s)   # unmapped symbol = its own singleton "sector"
                by_sector.setdefault(sec, {})[s] = momentum_values[s]

            for sec, sec_values in by_sector.items():
                if len(sec_values) >= 3:
                    sec_series = pd.Series(sec_values)
                    sec_ranks = sec_series.rank(pct=True) * 100
                    for s, r in sec_ranks.items():
                        ranks[s]['relative_strength'] = round(float(r), 1)
                else:
                    # too few sector peers to rank meaningfully — fall back
                    # to the whole-universe momentum rank instead
                    for s in sec_values:
                        if 'momentum' in ranks.get(s, {}):
                            ranks[s]['relative_strength'] = ranks[s]['momentum']
        else:
            for s in momentum_values:
                if 'momentum' in ranks.get(s, {}):
                    ranks[s]['relative_strength'] = ranks[s]['momentum']

        return ranks


# ═════════════════════════════════════════════════════════════════════════════
# 4. ADAPTIVE WEIGHT CALIBRATOR
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveWeightCalibrator:
    """
    Recalibrates entry-pattern weight multipliers using this bot's own
    realized, closed-trade history — this is the actual "learns from
    itself" component, and the part I'd point to as genuinely different
    from a static, hand-tuned rule set. It is NOT a machine-learning model;
    it is a simple, fully explainable empirical-Bayes-style reweighting,
    deliberately kept simple because a naive version of "adapt from your
    own results" is one of the easiest ways to build something that overfits
    to noise and quietly gets worse. Three safeguards specifically address
    that risk:

      1. MINIMUM SAMPLE SIZE — a pattern needs at least MIN_SAMPLE_SIZE
         closed trades before its own stats are trusted AT ALL. Below that,
         its weight doesn't move (verified: an unrealistically perfect small
         sample, e.g. a handful of all-winning trades, does NOT get
         rewarded with a big weight bump — that's much more likely to be
         luck than skill at that sample size).

      2. SHRINKAGE TOWARD THE POPULATION MEAN — even above the minimum, a
         pattern's own expectancy is blended with the expectancy across ALL
         closed trades, weighted by how much data that specific pattern has
         (more trades = trust its own number more; fewer = pull harder
         toward the population average). This is standard empirical-Bayes/
         James-Stein-style shrinkage — well-established statistical
         practice for exactly this "many small groups, don't overfit any
         one of them" situation, not something invented for this project.

      3. CAPPED WEIGHT SHIFT PER RECALIBRATION — even after shrinkage, no
         single recalibration can move a pattern's weight by more than
         MAX_WEIGHT_SHIFT_PCT from where it currently is. Weights evolve
         gradually across many recalibrations; a single hot or cold streak
         can't whipsaw the system.

    Expectancy (not win rate) is the metric being tracked throughout, for
    the same reason "accuracy" was reframed at the top of this file: a
    pattern with a low win rate but big winners can have excellent
    expectancy, and rewarding raw win-rate would push the system toward
    cutting winners short to inflate a vanity statistic.
    """

    MIN_SAMPLE_SIZE      = 15     # trades needed before a pattern's own stats are trusted at all
    SHRINKAGE_K           = 20    # prior strength — higher = more skeptical, needs more trades to be trusted
    MAX_WEIGHT_SHIFT_PCT = 0.15   # no single recalibration moves a weight by more than this, relative
    MIN_WEIGHT = 0.5
    MAX_WEIGHT = 1.5

    def _expectancy(self, pnl_series):
        n = len(pnl_series)
        if n == 0:
            return {'n': 0, 'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0, 'expectancy': 0.0}
        wins   = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series <= 0]
        win_rate = len(wins) / n
        avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
        avg_loss = float(-losses.mean()) if len(losses) > 0 else 0.0   # positive magnitude
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        return {'n': n, 'win_rate': round(win_rate, 3), 'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2), 'expectancy': round(expectancy, 2)}

    def compute_pattern_stats(self, trades_df, pattern_col='entry_type', pnl_col='net_pnl'):
        """
        trades_df: DataFrame of CLOSED trades (any open/unrealized rows
        should already be filtered out by the caller). Returns
        {pattern_name: {n, win_rate, avg_win, avg_loss, expectancy}}.
        """
        if trades_df is None or len(trades_df) == 0:
            return {}
        df = trades_df.dropna(subset=[pnl_col, pattern_col])
        return {pattern: self._expectancy(group[pnl_col])
                for pattern, group in df.groupby(pattern_col)}

    def calibrate_weights(self, trades_df, previous_weights=None,
                           pattern_col='entry_type', pnl_col='net_pnl'):
        """
        Returns {pattern: weight_multiplier} in [MIN_WEIGHT, MAX_WEIGHT],
        each within MAX_WEIGHT_SHIFT_PCT of its entry in previous_weights
        (default 1.0 for a pattern with no prior weight on record).
        """
        if trades_df is None or len(trades_df) == 0:
            return {}

        df = trades_df.dropna(subset=[pnl_col, pattern_col])
        if len(df) == 0:
            return {}

        pattern_stats = self.compute_pattern_stats(df, pattern_col, pnl_col)
        population     = self._expectancy(df[pnl_col])
        pop_expectancy = population['expectancy']
        previous_weights = previous_weights or {}

        weights, detail = {}, {}
        for pattern, s in pattern_stats.items():
            n = s['n']
            prev_w = previous_weights.get(pattern, 1.0)

            if n < self.MIN_SAMPLE_SIZE:
                weights[pattern] = round(prev_w, 3)
                detail[pattern] = {**s, 'weight': round(prev_w, 3), 'note': 'below min sample — unchanged'}
                continue

            shrink_factor    = n / (n + self.SHRINKAGE_K)
            shrunk_expectancy = shrink_factor * s['expectancy'] + (1 - shrink_factor) * pop_expectancy

            if abs(pop_expectancy) > 1e-6:
                raw_multiplier = 1.0 + (shrunk_expectancy - pop_expectancy) / abs(pop_expectancy) * 0.5
            else:
                raw_multiplier = 1.0 + shrunk_expectancy / 100.0   # fallback if population expectancy ~0

            raw_multiplier = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, raw_multiplier))

            max_up, max_down = prev_w * (1 + self.MAX_WEIGHT_SHIFT_PCT), prev_w * (1 - self.MAX_WEIGHT_SHIFT_PCT)
            capped = max(max_down, min(max_up, raw_multiplier))
            weights[pattern] = round(capped, 3)
            detail[pattern] = {**s, 'shrunk_expectancy': round(shrunk_expectancy, 2),
                                'raw_multiplier': round(raw_multiplier, 3), 'weight': weights[pattern]}

        self.last_run_detail = {'population': population, 'patterns': detail}
        return weights

    # ── Persistence ──────────────────────────────────────────────────────────
    # Each GitHub Actions run is a fresh process with no memory of the last
    # one — the capped-shift-per-recalibration safeguard (tested: correct
    # ordering emerges within ~2 rounds, never jumps) only actually works if
    # "previous_weights" survives between runs. These read/write a small CSV
    # for exactly that, kept as plain, inspectable rows — not a pickle blob —
    # so the weight history is something you can actually open and read.

    WEIGHTS_CSV_COLUMNS = ['pattern', 'weight', 'n', 'win_rate', 'expectancy', 'last_updated']

    def load_weights(self, path):
        """Returns {pattern: weight} from a previous save_weights() call, or {} if none exists yet."""
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_csv(path)
            if 'pattern' not in df.columns or 'weight' not in df.columns:
                return {}
            return dict(zip(df['pattern'], pd.to_numeric(df['weight'], errors='coerce')))
        except Exception as e:
            logger.error(f"AdaptiveWeightCalibrator.load_weights: could not read {path}: {e}")
            return {}

    def save_weights(self, path, weights):
        """
        Writes current weights to disk, including the stats behind each one
        (win_rate/expectancy/n) so the file is self-explanatory on its own,
        not just a bare number. Uses self.last_run_detail, populated by the
        most recent calibrate_weights() call.
        """
        if not weights:
            return False
        try:
            detail = getattr(self, 'last_run_detail', {}).get('patterns', {})
            rows = []
            for pattern, weight in weights.items():
                d = detail.get(pattern, {})
                rows.append({
                    'pattern': pattern, 'weight': weight,
                    'n': d.get('n', ''), 'win_rate': d.get('win_rate', ''),
                    'expectancy': d.get('expectancy', ''),
                    'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                })
            pd.DataFrame(rows, columns=self.WEIGHTS_CSV_COLUMNS).to_csv(path, index=False)
            return True
        except Exception as e:
            logger.error(f"AdaptiveWeightCalibrator.save_weights: could not write {path}: {e}")
            return False


# ═════════════════════════════════════════════════════════════════════════════
# 5. COMPOSITE ALPHA SCORE — orchestrator
# ═════════════════════════════════════════════════════════════════════════════

class CompositeAlphaScore:
    """
    Ties RegimeDetector + FactorEngine + CrossSectionalRanker +
    AdaptiveWeightCalibrator together into one explainable 0-100 score per
    symbol, ranked across the scan universe for a given day.

    On "accuracy" one more time, concretely: score_universe() returns a
    CONVICTION TIER label, not an "accuracy" or win-probability number. A
    tier is a relative ranking signal (top-tier setups are the strongest
    the CURRENT universe/regime combination has to offer, on the factors
    this engine tracks) — it is explicitly NOT a calibrated probability of
    the trade working out, because no synthetic test can produce that
    number honestly. Calibrating real probabilities per tier requires
    backtesting against actual historical price data (this sandbox has no
    live market data access) via the existing run_backtest.py
    infrastructure — that's the natural next step once this is integrated,
    not something to fake here.
    """

    BASE_FACTOR_WEIGHTS = {
        'momentum':           0.25,
        'risk_adj_momentum':  0.20,
        'trend_quality':      0.20,
        'volume_conviction':  0.15,
        'relative_strength':  0.20,
    }

    CONVICTION_TIERS = [
        (80, 'Tier 1 — High Conviction'),
        (60, 'Tier 2 — Moderate Conviction'),
        (40, 'Tier 3 — Marginal'),
        (0,  'Tier 4 — Low Conviction / Pass'),
    ]

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.factor_engine    = FactorEngine()
        self.ranker            = CrossSectionalRanker()
        self.calibrator        = AdaptiveWeightCalibrator()

    def _tier(self, score):
        for threshold, label in self.CONVICTION_TIERS:
            if score >= threshold:
                return label
        return self.CONVICTION_TIERS[-1][1]

    def score_symbol(self, symbol, factor_ranks, regime_result, pattern_weight=1.0):
        """
        factor_ranks: {factor_name: percentile_0_100} for ONE symbol, as
                      produced by CrossSectionalRanker.rank_universe().
        regime_result: RegimeDetector.classify() output.
        pattern_weight: multiplier from AdaptiveWeightCalibrator for
                        whichever entry pattern this symbol's setup matched
                        (1.0 = neutral / no adjustment / not yet calibrated).

        Returns {symbol, composite_score, tier, regime, breakdown}, where
        breakdown shows each factor's percentile, weight, and contribution —
        explainability is deliberate: a black-box score nobody can audit
        isn't more trustworthy for being harder to inspect.
        """
        tilts = regime_result.get('tilts', {})
        breakdown = {}
        weighted_sum, weight_used = 0.0, 0.0

        for factor, base_weight in self.BASE_FACTOR_WEIGHTS.items():
            if factor not in factor_ranks:
                continue   # missing factor = no opinion, not a penalty
            pctl = factor_ranks[factor]
            tilt = tilts.get(factor, 1.0)
            eff_weight = base_weight * tilt
            contribution = pctl * eff_weight
            breakdown[factor] = {
                'percentile': pctl, 'base_weight': base_weight,
                'regime_tilt': tilt, 'contribution': round(contribution, 2),
            }
            weighted_sum += contribution
            weight_used  += eff_weight

        if weight_used < 1e-6:
            return {'symbol': symbol, 'composite_score': None, 'tier': 'No data',
                    'regime': regime_result.get('regime'), 'breakdown': {}}

        raw_score = weighted_sum / weight_used   # renormalise for any missing factors
        dampener  = tilts.get('overall_dampener', 1.0)
        final_score = max(0.0, min(100.0, raw_score * dampener * pattern_weight))

        return {
            'symbol': symbol,
            'composite_score': round(final_score, 1),
            'tier': self._tier(final_score),
            'regime': regime_result.get('regime'),
            'pattern_weight_applied': pattern_weight,
            'overall_dampener_applied': dampener,
            'breakdown': breakdown,
        }

    def score_universe(self, symbol_dfs, index_df, sector_map=None,
                        trades_df=None, pattern_by_symbol=None,
                        pattern_col='entry_type', pnl_col='net_pnl'):
        """
        Full pipeline for one day's scan.

        symbol_dfs: {symbol: ohlcv_df} for every symbol in today's universe
        index_df: broad market index OHLCV, for regime detection
        sector_map: {symbol: sector}, optional — enables relative_strength
        trades_df: this bot's CLOSED trade history, optional — enables
                   adaptive pattern weighting. If not supplied, every
                   symbol gets a neutral pattern_weight of 1.0.
        pattern_by_symbol: {symbol: pattern_name}, optional — which entry
                   pattern each symbol's current setup matches, so the
                   right calibrated weight gets applied to each

        Returns a list of per-symbol result dicts (same shape as
        score_symbol's return), sorted by composite_score descending.
        """
        regime_result = self.regime_detector.classify(index_df)

        factor_values = {s: self.factor_engine.compute_all(df) for s, df in symbol_dfs.items()}
        factor_ranks  = self.ranker.rank_universe(factor_values, sector_map=sector_map)

        pattern_weights = {}
        if trades_df is not None and len(trades_df) > 0:
            pattern_weights = self.calibrator.calibrate_weights(trades_df, pattern_col=pattern_col, pnl_col=pnl_col)

        pattern_by_symbol = pattern_by_symbol or {}
        results = []
        for symbol in symbol_dfs:
            pattern = pattern_by_symbol.get(symbol)
            p_weight = pattern_weights.get(pattern, 1.0) if pattern else 1.0
            results.append(self.score_symbol(symbol, factor_ranks.get(symbol, {}), regime_result, p_weight))

        results.sort(key=lambda r: (r['composite_score'] is not None, r['composite_score']), reverse=True)
        return results


if __name__ == '__main__':
    # Lightweight smoke test / usage demo — NOT a substitute for the
    # dedicated test suite this was built and validated against (synthetic
    # data covering each component individually, see accompanying tests).
    import numpy as np
    np.random.seed(0)

    def demo_df(n, drift, noise):
        prices = [100.0]
        for i in range(1, n):
            prices.append(prices[-1] * (1 + np.random.normal(drift, noise)))
        close = pd.Series(prices)
        return pd.DataFrame({
            'high': close * 1.01, 'low': close * 0.99, 'close': close,
            'volume': pd.Series(np.random.randint(100000, 300000, n).astype(float)),
        })

    universe = {
        'STRONG_MOVER':  demo_df(200, 0.004, 0.008),
        'MODEST_MOVER':  demo_df(200, 0.001, 0.008),
        'FLAT':          demo_df(200, 0.0001, 0.006),
        'DECLINING':     demo_df(200, -0.002, 0.007),
    }
    index = demo_df(200, 0.0015, 0.005)

    engine = CompositeAlphaScore()
    results = engine.score_universe(universe, index)

    print(f"Regime: {results[0]['regime']}\n")
    for r in results:
        print(f"{r['symbol']:<14} score={r['composite_score']:<6} {r['tier']}")
