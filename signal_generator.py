# signal_generator.py  ── v4  (ADAPTIVE BARRIER GEOMETRY + COST-AWARE ECONOMICS)
# ═════════════════════════════════════════════════════════════════════════════
# WHAT v3 GOT RIGHT AND v4 KEEPS: the nine named entry patterns, the
# lookahead-free single-bar evaluation, the debounce, the fundamental hard-fail
# gate, the risk-fraction sizing model, and the public API
# (SignalGenerator(tech, screener) / generate_signal(df, symbol, fundamentals,
# current_equity, market_regime, last_exit_bar) / classify_market_regime).
# Every `entry_type` string is unchanged, so pattern_weights.csv,
# alpha_engine's calibrator and backtest_analytics' stratification all keep
# working against continuous history.
#
# WHAT v4 CHANGES, AND THE EVIDENCE FOR EACH — measured on the 50 closed rows
# in paper_trades.csv (2026-04-18 → 2026-09-09):
#
# 1. STOPS SAT INSIDE THE NOISE BAND.
#    Realised stop distance: median 3.3% of entry, i.e. ~1.3-1.8 daily ATR.
#    For a driftless daily process the probability of touching -1.6σ_d at some
#    point inside a 15-bar window is 2·Φ(-1.6/√15) ≈ 66%. Observed: 26 of 50
#    rows closed "SL Hit", at an average of -0.85R. The stop was measuring the
#    market's daily breathing, not the setup being wrong.
#    v4: stop distance is derived from Kaufman efficiency ratio and a blended
#    ATR/Yang-Zhang volatility estimate, floored at k_min·σ where k_min widens
#    as trend efficiency falls, and anchored to the actual structural
#    invalidation level (swing low / broken resistance / signal-bar low)
#    rather than a fixed ATR multiple per pattern.
#
# 2. TARGETS SAT OUTSIDE THE HORIZON.
#    Target distance: median 12.1% of entry, against a 15-bar horizon whose
#    one-sigma displacement is σ_d·√15 ≈ 7%. Only 2 of 50 rows ever reached
#    "Target Hit"; 22 closed on the 15-day time exit instead. The bot was
#    aiming at a level the holding period could not reach, so the exit that
#    actually ran the strategy was the clock.
#    v4: the target is capped at REACH_FRACTION · σ_d · √H — the distance the
#    name can plausibly cover in the time available — and the trade is passed
#    over when that reachable target fails to clear min_rr. The planned hold
#    window is then back-solved from the target distance
#    (H* = (target/σ_d)²), so the clock stops truncating trades that were
#    still working.
#
# 3. COSTS WERE EATING THE STRATEGY WHOLE.
#    On the 24 rows priced with the current realistic cost model: gross +₹184,
#    commission ₹573, net -₹389. Mean round-trip cost 113 bps; worst rows 654
#    bps (BPCL, ₹318 notional) and 195 bps (KOTAKBANK tranches, ₹1,192
#    notional). The driver is FLAT_CHARGE_PER_SELL (~₹20 DP charge per scrip
#    per sell) landing on rows of ₹1,000-4,000 — and scaled exits tripling the
#    count of those rows per decision.
#    v4: every signal carries its own economics block (cost, cost_bps,
#    expected value, break-even move), a minimum economic notional derived
#    from the flat charge itself (min_notional = FLAT_CHARGE / MAX_FLAT_BPS),
#    a `tranche_ok` flag that only permits splitting when each resulting row
#    still clears that floor, and validate_economics() for the caller to
#    re-check AFTER its own tier/fair-share/risk-budget resizing — which is
#    where the notional actually gets destroyed.
#
# 4. THE PRIMARY PATTERN WAS PICKED BY DICT ORDER.
#    v3 used active_patterns[0], and `breakout` is first in the dict — so any
#    confluence containing a breakout was labelled a breakout, and inherited
#    the tightest stop multiple in the table (1.2 ATR). Realised: breakout
#    n=13, win rate 7.7%, mean -0.517R — the worst pattern in the book, and
#    the one being handed the least room. bb_squeeze (1.0 ATR stop) was
#    second-worst at 0/4 and -0.750R.
#    v4: the primary pattern is the one with the best shrunk realised
#    expectancy among those active, so the trade is labelled — and geometried,
#    and credited in pattern_weights.csv — by its strongest component.
#
# 5. ANY ONE OF NINE PATTERNS WAS A COMPLETE ENTRY DECISION.
#    v4 keeps all nine as triggers but requires a composite entry-quality
#    score (trend alignment, DMI separation, efficiency ratio, relative
#    strength vs Nifty, volume/flow conviction, non-extension) to clear a
#    threshold that rises for patterns with weak realised expectancy and in
#    hostile regimes. Quality then feeds the win-probability estimate inside
#    the EV gate, so it is load-bearing arithmetic rather than a label.
#
# 6. NSE MARKET STRUCTURE WAS ABSENT.
#    v4 adds: rupee-turnover liquidity floor, ADV participation cap, price-band
#    (circuit) inference from the realised return distribution, rejection when
#    the stop sits outside the band the scrip can move in a day (a gap to lower
#    circuit means the stop cannot fill at all), upper-circuit proximity
#    rejection, and ₹0.05 tick rounding on every emitted level.
#
# 7. OVERNIGHT GAP RISK WAS UNPRICED.
#    This is an EOD system on a market closed ~17.5 hours a day: a 3% stop
#    fills at -6% when the scrip opens -6%. Expected loss per stop-out is now
#    stop_distance + GAP_WEIGHT · p90(down gap), and that gap-loaded loss is
#    what the EV gate must clear.
# ═════════════════════════════════════════════════════════════════════════════

import logging
import math
import re

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cost model is shared with the backtester and the live manager; imported, not
# duplicated. A standalone import failure degrades to the same published
# constants rather than taking the module down.
try:
    from trading_costs import (
        round_trip_commission, PCT_COST_PER_LEG, FLAT_CHARGE_PER_SELL,
    )
except ImportError:                                              # pragma: no cover
    PCT_COST_PER_LEG = 0.0013
    FLAT_CHARGE_PER_SELL = 20.0

    def round_trip_commission(entry_price, exit_price, position_size):
        pct = (entry_price + exit_price) * position_size * PCT_COST_PER_LEG
        return round(pct + FLAT_CHARGE_PER_SELL, 2)


SECTOR_PE = {
    'RELIANCE': 25, 'TCS': 30, 'INFY': 28, 'WIPRO': 24, 'HCLTECH': 22,
    'HDFCBANK': 18, 'ICICIBANK': 17, 'SBIN': 10, 'KOTAKBANK': 20, 'AXISBANK': 14,
    'HINDUNILVR': 55, 'NESTLEIND': 70, 'ITC': 25, 'ASIANPAINT': 60, 'TITAN': 80,
    'MARUTI': 28, 'TMPV': 15, 'BAJAJ-AUTO': 25,
    'TATASTEEL': 12, 'JSWSTEEL': 10, 'HINDALCO': 10,
    'BHARTIARTL': 35, 'LT': 28, 'BAJFINANCE': 30, 'ULTRACEMCO': 35,
}


# ═════════════════════════════════════════════════════════════════════════════
# RISK PROFILE
# ═════════════════════════════════════════════════════════════════════════════
# Keys carried over from v3 keep their names so swing_trading_bot.py's direct
# RISK_PROFILE['max_capital_pct'] reference and any external tuning keep
# working. The per-pattern sl_mult_*/tgt_mult_* constants are retired: stop and
# target distance are now computed per-signal from volatility, structure and
# horizon, which is the entire point of v4.
RISK_PROFILE = {
    # ── Position sizing ──────────────────────────────────────────────────────
    'risk_pct_per_trade':   0.04,   # fraction of equity at risk if the stop fills at its level
    'max_capital_pct':      0.30,   # ceiling on one name. 0.30 rather than v3's 0.40 because
                                     # min_notional_inr (below) already implies ~3-4 economically
                                     # viable concurrent positions at ₹50k equity; a 40% ceiling
                                     # invites two names to consume the whole book.

    # ── Data sufficiency ─────────────────────────────────────────────────────
    'min_bars':             70,      # Wilder ADX needs ~2x its period to settle; EMA-50 needs 50;
                                     # Yang-Zhang needs 20. 70 is the point where every input is warm.
    'debounce_bars':         3,

    # ── Entry filters ────────────────────────────────────────────────────────
    'min_adx':              17,      # meaningful again now that ADX is Wilder-correct; v3's 12 was
                                     # calibrated against a differently-scaled series
    'min_di_spread':         3.0,    # +DI must lead -DI by this much
    'rsi_lo':               35,
    'rsi_hi':               78,
    'bear_rsi_cap':         55,
    'allow_bear_longs':     True,
    'max_extension_atr':     2.8,    # reject entries this far above EMA-20 in ATR units — chasing an
                                     # already-extended move is what produced the 1-day stop-outs
    'min_efficiency_ratio':  0.14,   # below this the name is round-tripping, not trending

    # ── Barrier geometry ─────────────────────────────────────────────────────
    'k_stop_base':           1.50,   # stop floor in blended-sigma units at perfect trend efficiency
    'k_stop_noise_slope':    1.10,   # ...widening linearly as efficiency ratio falls
    'k_stop_max':            2.60,   # ...capped, so size never collapses
    'struct_buffer_atr':     0.30,   # clearance below the structural invalidation level
    'reach_fraction':        1.10,   # target ceiling as a multiple of sigma_d*sqrt(H)
    'tgt_mult_momentum':     2.60,   # R-multiple ambition for continuation patterns
    'tgt_mult_reversion':    2.20,   # ...and for pullback/oversold patterns
    'min_rr':                1.65,
    'min_risk_pct':          0.010,  # a stop closer than 1.0% of price is inside the spread+noise
    'max_risk_pct':          0.080,

    # ── Quality gate ─────────────────────────────────────────────────────────
    'base_quality':          0.50,
    'quality_prior_slope':   0.35,   # how hard weak realised expectancy raises the bar
    'regime_quality_add':   {'BULL': 0.00, 'NEUTRAL': 0.03, 'BEAR': 0.09},
    'confluence_bonus':      0.035,  # per extra confirming pattern, capped below
    'confluence_bonus_cap':  0.105,

    # ── Economics ────────────────────────────────────────────────────────────
    'max_flat_cost_bps':    15.0,    # tolerated share of the flat DP charge, in bps of notional.
                                     # min_notional_inr = FLAT_CHARGE_PER_SELL / (bps/10000).
    'max_total_cost_bps':   55.0,    # hard ceiling on modelled round-trip cost
    'cost_hurdle_mult':      1.50,   # expected gross value must be this multiple of modelled cost
    'gap_weight':            0.60,   # share of p90 down-gap added to expected loss per stop-out
    'edge_tilt':             0.60,   # how far entry quality is allowed to tilt the first-passage
                                     # win probability away from its driftless barrier ratio

    # ── NSE microstructure ───────────────────────────────────────────────────
    'min_median_turnover':   5.0e7,  # ₹5 crore median daily traded value
    'max_adv_participation': 0.005,  # position notional as a share of that median
    'min_price':             20.0,   # below this the ₹0.05 tick is a material share of the stop
    'circuit_headroom':      0.80,   # reject when today's close has used this much of the up-band
    'circuit_stop_cover':    1.40,   # the price band must exceed the stop distance by this factor
}



# ═════════════════════════════════════════════════════════════════════════════
# RISK CALIBRATIONS
# ═════════════════════════════════════════════════════════════════════════════
# Two named calibrations rather than edited-in-place constants, so a change of
# risk appetite is a one-word diff and the alternative stays readable beside it.
#
# 'aggressive' buys trade frequency and per-trade size in the three places
# where frequency is actually purchasable, and declines to buy it in the one
# place where it is not:
#
#   BOUGHT — context filters (ADX, DI spread, efficiency ratio, extension,
#            RSI ceiling, turnover floor). These decide how much of the
#            universe reaches the pattern stage, and on the synthetic funnel
#            they are the gates eating ~70% of it. Loosening them raises
#            candidate count roughly linearly.
#   BOUGHT — the economic floor. max_flat_cost_bps 15 -> 22 drops the minimum
#            viable notional from ₹13,333 to ₹9,091, which takes the account
#            from ~3 economically viable slots to ~5. This is the single
#            largest frequency lever available, and it costs 7 bps a trade.
#   BOUGHT — ambition. reach_fraction and tgt_mult_momentum are raised, so
#            winners are allowed to aim further inside the same horizon.
#   DECLINED — k_stop_max stays at 2.60σ. Tightening the stop cap would raise
#            achievable R:R arithmetically and look like free frequency, but
#            it re-creates precisely the defect that produced 26 stop-outs at
#            -0.85R: a stop inside the noise band. Risk appetite is expressed
#            through position size and filter width, never by moving the stop
#            back into the range the stock traverses on an ordinary day.
#
# Switch with ACTIVE_CALIBRATION, or call apply_calibration('balanced') before
# constructing SignalGenerator.
RISK_CALIBRATIONS = {
    'balanced': {},                       # the values defined above
    'aggressive': {
        # ── Size ──────────────────────────────────────────────────────────
        'risk_pct_per_trade':    0.055,   # 4.0% -> 5.5% of equity per trade
        'max_capital_pct':       0.40,    # concentration is how conviction gets expressed

        # ── Frequency: context filters ────────────────────────────────────
        'min_adx':              14,
        'min_di_spread':         1.5,
        'min_efficiency_ratio':  0.10,
        'max_extension_atr':     3.4,
        'rsi_hi':               82,       # buying strength rather than fading it
        'debounce_bars':         2,
        'min_median_turnover':   3.0e7,   # ₹3cr — opens the midcap/high-beta tail

        # ── Frequency: economics ──────────────────────────────────────────
        'max_flat_cost_bps':    22.0,     # min notional ₹13,333 -> ₹9,091
        'max_total_cost_bps':   70.0,
        'cost_hurdle_mult':      1.20,

        # ── Ambition ──────────────────────────────────────────────────────
        'min_rr':                1.50,
        'reach_fraction':        1.25,
        'tgt_mult_momentum':     3.00,
        'tgt_mult_reversion':    2.40,
        'base_quality':          0.42,
        'edge_tilt':             0.80,    # a larger claimed edge per unit of quality;
                                           # the honest cost is that a wrong edge_tilt now
                                           # passes more marginal trades, so calibrate it
                                           # against realised R as soon as ~40 trades carry
                                           # a quality_score
    },
}

ACTIVE_CALIBRATION = 'aggressive'


def apply_calibration(name):
    """Overlay a named calibration onto RISK_PROFILE in place, so existing
    importers of RISK_PROFILE see the change without re-importing."""
    overrides = RISK_CALIBRATIONS.get(name)
    if overrides is None:
        raise KeyError(f"unknown calibration '{name}' — have {list(RISK_CALIBRATIONS)}")
    RISK_PROFILE.update(overrides)
    logger.info(f"✓ Risk calibration: {name}")
    return RISK_PROFILE


apply_calibration(ACTIVE_CALIBRATION)

# ═════════════════════════════════════════════════════════════════════════════
# PATTERN PRIORS
# ═════════════════════════════════════════════════════════════════════════════
# (n_trades, mean_R) as realised by THIS bot, read off paper_trades.csv's 50
# closed rows. Used only through empirical-Bayes shrinkage toward zero — the
# same n/(n+K) device alpha_engine's AdaptiveWeightCalibrator already uses —
# so a 2-trade sample moves the bar slightly and a 20-trade sample moves it
# properly. Nothing here bans a pattern; a weak prior raises the entry-quality
# threshold that pattern must clear, and a strong prior relaxes it.
#
# Refresh these from a closed-trade groupby whenever the sample grows
# meaningfully; alpha_engine's calibrator handles the sizing side of the same
# feedback loop independently.
PATTERN_PRIOR = {
    'pullback':       (20,  0.186),
    'breakout':       (13, -0.517),
    'cmf_accum':      (7,   0.304),
    'bb_squeeze':     (4,  -0.750),
    'stoch_cross':    (3,   1.322),
    'engulfing':      (2,  -1.000),
    'rsi_divergence': (1,  -0.339),
    'momentum_burst': (0,   0.000),
    'ema_cross':      (0,   0.000),
}
PRIOR_SHRINK_K = 12.0     # trades needed for 50% trust in a pattern's own realised mean

MOMENTUM_PATTERNS = {'breakout', 'momentum_burst', 'bb_squeeze', 'ema_cross'}


def shrunk_pattern_expectancy(pattern):
    """Empirical-Bayes mean R for a pattern, shrunk toward 0 (no opinion)."""
    n, mean_r = PATTERN_PRIOR.get(pattern, (0, 0.0))
    return (n / (n + PRIOR_SHRINK_K)) * mean_r


# ═════════════════════════════════════════════════════════════════════════════
# NSE MARKET STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════
class NSEMicrostructure:
    """
    The exchange-specific constraints a generic ATR model has no way to know
    about. All of these are inferred from the OHLCV series already in hand —
    no extra API surface, no paid data, nothing to keep in sync.
    """

    TICK = 0.05                                   # standard NSE cash-segment tick
    BANDS = (0.02, 0.05, 0.10, 0.20)              # dynamic price bands / circuit limits
    BAND_TOLERANCE = 0.0018                       # closeness that counts as "pinned to the band"

    @staticmethod
    def round_to_tick(price, mode='nearest'):
        """Emitted levels land on prices the exchange will actually accept."""
        t = NSEMicrostructure.TICK
        if price is None or not np.isfinite(price):
            return price
        if mode == 'down':
            return round(math.floor(round(price / t, 6)) * t, 2)
        if mode == 'up':
            return round(math.ceil(round(price / t, 6)) * t, 2)
        return round(round(price / t) * t, 2)

    @staticmethod
    def infer_price_band(df, lookback=180):
        """
        Infer the scrip's daily price band from how its close-to-close returns
        behave at the extremes. A stock in a 5% band prints |return| ≈ 0.0500
        repeatedly and never exceeds it; an unbanded large cap wanders past
        every candidate band. Two or more pinned days inside the lookback, with
        nothing beyond that level, identifies the band.

        Returns the band as a fraction, or None when the series moves freely
        (which is the common case for index constituents and is treated as
        "no constraint", not as missing data).
        """
        close = df['close'].tail(lookback)
        if len(close) < 30:
            return None
        rets = (close / close.shift() - 1.0).dropna().abs()
        if rets.empty:
            return None
        for band in NSEMicrostructure.BANDS:
            pinned = ((rets - band).abs() <= NSEMicrostructure.BAND_TOLERANCE).sum()
            beyond = (rets > band + NSEMicrostructure.BAND_TOLERANCE).sum()
            if pinned >= 2 and beyond == 0:
                return band
        return None

    @staticmethod
    def upper_band_usage(df, band):
        """
        How much of today's permitted up-move is already spent, in [0, 1].
        A close sitting at 0.9 of the up-band is a scrip that may open locked
        tomorrow — an unfillable entry, and a position that cannot be exited
        if the lock flips to the down side later.
        """
        if band is None or len(df) < 2:
            return 0.0
        prev_close = float(df['close'].iloc[-2])
        last_close = float(df['close'].iloc[-1])
        if prev_close <= 0:
            return 0.0
        return float(max(0.0, (last_close / prev_close - 1.0)) / band)


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
class SignalGenerator:

    def __init__(self, tech_indicators, fundamental_screener):
        self.tech = tech_indicators
        self.fund = fundamental_screener
        self.R = RISK_PROFILE
        self.micro = NSEMicrostructure()
        # Per-scan funnel tally. Live NSE data is the only place these filters
        # can honestly be calibrated, and "the bot found nothing today" is
        # useless without knowing WHICH gate ate the universe. Read it after a
        # scan with funnel_summary() and move whichever constant is doing the
        # cutting, rather than guessing at the whole dial.
        self.funnel = {}
        logger.info("✓ SignalGenerator v4 — adaptive barrier geometry, cost-gated")

    # ─────────────────────────────────────────────────────────────────────────
    def funnel_summary(self, reset=False):
        """{gate: count} for the scan so far, most frequent first."""
        out = dict(sorted(self.funnel.items(), key=lambda kv: -kv[1]))
        if reset:
            self.funnel = {}
        return out

    @staticmethod
    def _funnel_key(reason):
        """Collapse a reason string to its gate by stripping the numbers."""
        return re.sub(r'[-+\d.]+', '#', str(reason))[:70]

    # ─────────────────────────────────────────────────────────────────────────
    # Public: economic viability, re-checkable after caller-side resizing
    # ─────────────────────────────────────────────────────────────────────────
    def validate_economics(self, details, position_size=None):
        """
        Returns (ok: bool, reason: str|None, economics: dict).

        Call this AFTER any downstream resizing — alpha tier multiplier,
        fair-share capital cap, portfolio risk-budget shrink — because those
        are what turn a ₹20,000 signal into a ₹1,200 row, and ₹1,200 is where
        the ~₹20 flat DP charge becomes 195 bps of round trip. Sizing decisions
        made in three different places can only be cost-checked at the end of
        the chain, so this is deliberately a separate entry point rather than
        something baked into generate_signal alone.
        """
        size = int(position_size if position_size is not None else details.get('position_size', 0))
        entry = float(details['entry_price'])
        target = float(details['target_price'])
        stop = float(details['stop_loss'])

        if size < 1 or entry <= 0:
            return False, 'no position size', {}

        notional = entry * size
        cost = float(round_trip_commission(entry, target, size))
        cost_bps = cost / notional * 1e4

        risk_ps = entry - stop
        reward_ps = target - entry
        gap_loss_ps = entry * float(details.get('gap_down_p90', 0.0)) * self.R['gap_weight']
        p_win = float(details.get('p_win_est', 0.0))
        ev_ps = p_win * reward_ps - (1.0 - p_win) * (risk_ps + gap_loss_ps)
        ev_gross = ev_ps * size

        # Break-even move: how far price must travel just to pay the exchange
        # and the depository. On a ₹1,200 row that is ~2%, which is most of a
        # 15-day expected move — the trade is over before it starts.
        breakeven_pct = cost / notional * 100.0

        economics = {
            'notional': round(notional, 2),
            'est_cost': round(cost, 2),
            'cost_bps': round(cost_bps, 1),
            'breakeven_move_pct': round(breakeven_pct, 3),
            'expected_value': round(ev_gross, 2),
            'ev_to_cost': round(ev_gross / cost, 2) if cost > 0 else None,
            'min_notional_inr': details.get('min_notional_inr'),
        }

        if notional < details.get('min_notional_inr', 0):
            return False, (f"notional ₹{notional:,.0f} below the ₹{details['min_notional_inr']:,.0f} "
                           f"economic floor (flat charges would be {cost_bps:.0f} bps)"), economics
        if cost_bps > self.R['max_total_cost_bps']:
            return False, f"round-trip cost {cost_bps:.0f} bps exceeds the {self.R['max_total_cost_bps']:.0f} bps ceiling", economics
        if ev_gross < self.R['cost_hurdle_mult'] * cost:
            return False, (f"expected value ₹{ev_gross:,.0f} is under {self.R['cost_hurdle_mult']:.1f}x "
                           f"modelled cost ₹{cost:,.0f}"), economics
        return True, None, economics

    @staticmethod
    def recommended_max_open_trades(equity, min_notional=None):
        """
        How many positions this much capital can carry while every one of them
        still clears the flat-charge floor. At ₹50,000 with a ₹13,333 floor the
        answer is 3-4 — which is the honest reason 10 slots × 3 tranches was
        producing ₹1,200 rows at 195 bps.
        """
        floor = min_notional or (FLAT_CHARGE_PER_SELL / (RISK_PROFILE['max_flat_cost_bps'] / 1e4))
        return max(1, int(equity / floor))

    # ─────────────────────────────────────────────────────────────────────────
    # Public: the signal
    # ─────────────────────────────────────────────────────────────────────────
    def generate_signal(self, df, symbol, fundamentals, current_equity=50000,
                        market_regime='BULL', last_exit_bar=None,
                        benchmark_df=None, max_hold_days=15):
        """Public entry point — evaluates the bar and tallies the funnel."""
        signal, details = self._evaluate(
            df, symbol, fundamentals, current_equity=current_equity,
            market_regime=market_regime, last_exit_bar=last_exit_bar,
            benchmark_df=benchmark_df, max_hold_days=max_hold_days,
        )
        key = 'BUY' if signal == 'BUY' else self._funnel_key(details.get('reason', 'unknown'))
        self.funnel[key] = self.funnel.get(key, 0) + 1
        return signal, details

    def _evaluate(self, df, symbol, fundamentals, current_equity=50000,
                  market_regime='BULL', last_exit_bar=None,
                  benchmark_df=None, max_hold_days=15):
        """
        Returns ('BUY', details) or ('HOLD', {'reason': ...}).

        benchmark_df: optional Nifty OHLCV frame. Supplying it activates the
                      relative-strength component of the quality score — the
                      one factor here that is cross-sectional rather than
                      self-referential, and among the most durable predictors
                      in the published equity literature. Absent, the remaining
                      components renormalise and the signal still stands.
        max_hold_days: the exit clock the caller will actually enforce. The
                      target is capped at what this window can deliver, so the
                      strategy stops aiming past its own horizon.
        """
        R = self.R
        T = self.tech

        # ── Gate 0: data sufficiency, regime, debounce ───────────────────────
        if df is None or len(df) < R['min_bars']:
            return 'HOLD', {'reason': f'Need {R["min_bars"]} bars, got {0 if df is None else len(df)}'}
        if market_regime == 'BEAR' and not R['allow_bear_longs']:
            return 'HOLD', {'reason': 'BEAR regime, longs disabled'}

        current_bar = len(df) - 1
        if last_exit_bar is not None and (current_bar - last_exit_bar) < R['debounce_bars']:
            return 'HOLD', {'reason': f'Debounce ({current_bar - last_exit_bar} bars)'}

        # ── Gate 1: fundamentals, cheapest hard filter, run first ────────────
        try:
            _, fund_checks = self.fund.check_fundamental_gate(fundamentals)
            if isinstance(fund_checks, dict) and 'reason' in fund_checks:
                return 'HOLD', {'reason': f'Fundamental hard-fail: {fund_checks["reason"]}'}
        except Exception:
            pass                                    # missing fundamentals is no opinion, not a block

        d = self._build_frame(df)
        latest, prev, prev2 = d.iloc[-1], d.iloc[-2], d.iloc[-3]
        entry_price = float(latest['close'])

        if entry_price < R['min_price']:
            return 'HOLD', {'reason': f'Price ₹{entry_price:.2f} below the ₹{R["min_price"]:.0f} tick-noise floor'}

        # ── Gate 2: NSE market structure ─────────────────────────────────────
        turnover = float(latest['turnover_med']) if np.isfinite(latest['turnover_med']) else 0.0
        if turnover < R['min_median_turnover']:
            return 'HOLD', {'reason': f'Median turnover ₹{turnover/1e7:.2f}cr below the '
                                      f'₹{R["min_median_turnover"]/1e7:.1f}cr liquidity floor'}

        price_band = self.micro.infer_price_band(df)
        band_usage = self.micro.upper_band_usage(df, price_band)
        if price_band is not None and band_usage >= R['circuit_headroom']:
            return 'HOLD', {'reason': f'Close has used {band_usage*100:.0f}% of the {price_band*100:.0f}% '
                                      f'up-band — entry may be unfillable tomorrow'}

        # ── Gate 3: trend / momentum context ─────────────────────────────────
        sigma_d = float(latest['sigma_d'])                  # daily vol as a fraction of price
        sigma_abs = sigma_d * entry_price                   # ...in rupees, the geometry's unit
        er = float(latest['er']) if np.isfinite(latest['er']) else 0.0
        adx = float(latest['adx']) if np.isfinite(latest['adx']) else 0.0
        plus_di = float(latest['plus_di']) if np.isfinite(latest['plus_di']) else 0.0
        minus_di = float(latest['minus_di']) if np.isfinite(latest['minus_di']) else 0.0
        rsi = float(latest['rsi'])

        above_ema20 = entry_price > latest['ema_20']
        above_ema50 = entry_price > latest['ema_50']
        ema50_rising = latest['ema_50'] > d['ema_50'].iloc[-10]
        extension_atr = (entry_price - float(latest['ema_20'])) / max(sigma_abs, 1e-9)

        if market_regime == 'BEAR':
            if not (above_ema50 and ema50_rising):
                return 'HOLD', {'reason': 'BEAR: price below or against a declining EMA-50'}
            if rsi > R['bear_rsi_cap']:
                return 'HOLD', {'reason': f'BEAR: RSI {rsi:.0f} extended'}
        elif not (above_ema20 or (above_ema50 and ema50_rising)):
            return 'HOLD', {'reason': 'Below both EMA-20 and a declining EMA-50'}

        if adx < R['min_adx']:
            return 'HOLD', {'reason': f'ADX {adx:.1f} < {R["min_adx"]} (Wilder)'}
        if (plus_di - minus_di) < R['min_di_spread']:
            return 'HOLD', {'reason': f'+DI {plus_di:.1f} leads -DI {minus_di:.1f} by less than {R["min_di_spread"]}'}
        if not (R['rsi_lo'] < rsi < R['rsi_hi']):
            return 'HOLD', {'reason': f'RSI {rsi:.1f} outside [{R["rsi_lo"]},{R["rsi_hi"]}]'}
        if er < R['min_efficiency_ratio']:
            return 'HOLD', {'reason': f'Efficiency ratio {er:.2f} — path is round-tripping, not trending'}
        if extension_atr > R['max_extension_atr']:
            return 'HOLD', {'reason': f'{extension_atr:.1f}σ above EMA-20 — entering an already-extended move'}

        hist_up = latest['macd_histogram'] > prev['macd_histogram'] > prev2['macd_histogram']
        if not ((latest['macd'] > latest['macd_signal']) or hist_up):
            return 'HOLD', {'reason': 'MACD not constructive'}

        # ── Gate 4: entry patterns ───────────────────────────────────────────
        entry_signals = self._detect_patterns(d, latest, prev, prev2, sigma_abs, above_ema50)
        active_patterns = [k for k, v in entry_signals.items() if v]
        if not active_patterns:
            return 'HOLD', {'reason': 'No pattern triggered'}

        # Primary = best shrunk realised expectancy among those firing, so the
        # trade is named, geometried and credited by its strongest component
        # rather than by dictionary order.
        primary = max(active_patterns, key=shrunk_pattern_expectancy)

        # ── Gate 5: entry quality ────────────────────────────────────────────
        quality, q_parts = self._entry_quality(
            d, latest, sigma_abs, er, plus_di, minus_di, extension_atr,
            above_ema20, above_ema50, ema50_rising, benchmark_df,
        )
        quality += min(R['confluence_bonus'] * (len(active_patterns) - 1), R['confluence_bonus_cap'])
        quality = float(np.clip(quality, 0.0, 1.0))

        required_quality = float(np.clip(
            R['base_quality']
            - R['quality_prior_slope'] * shrunk_pattern_expectancy(primary)
            + R['regime_quality_add'].get(market_regime, 0.03),
            0.38, 0.75,
        ))
        if quality < required_quality:
            return 'HOLD', {'reason': f'Entry quality {quality:.2f} < {required_quality:.2f} required for '
                                      f'{primary} in {market_regime}'}

        # ── Gate 6: barrier geometry ─────────────────────────────────────────
        geom = self._barrier_geometry(
            d, latest, entry_price, sigma_abs, sigma_d, er, primary,
            market_regime, max_hold_days,
        )
        if geom.get('reason'):
            return 'HOLD', {'reason': geom['reason']}

        stop_loss = geom['stop_loss']
        target_price = geom['target_price']
        risk_per_share = geom['risk_per_share']
        risk_pct = risk_per_share / entry_price

        if not (R['min_risk_pct'] <= risk_pct <= R['max_risk_pct']):
            return 'HOLD', {'reason': f'Stop at {risk_pct*100:.2f}% of price is outside the '
                                      f'[{R["min_risk_pct"]*100:.1f}%, {R["max_risk_pct"]*100:.1f}%] usable band'}

        actual_rr = (target_price - entry_price) / risk_per_share
        if actual_rr < R['min_rr']:
            return 'HOLD', {'reason': f'Reachable R:R {actual_rr:.2f} < {R["min_rr"]} within '
                                      f'{max_hold_days} bars — the horizon cannot pay for this stop'}

        # A stop wider than the daily band cannot fill on the day it is
        # breached: the scrip locks at the lower circuit and the exit queues
        # behind everyone else's.
        if price_band is not None and risk_pct * R['circuit_stop_cover'] > price_band:
            return 'HOLD', {'reason': f'Stop {risk_pct*100:.1f}% sits too close to the '
                                      f'{price_band*100:.0f}% price band to be fillable'}

        # ── Gate 7: sizing ───────────────────────────────────────────────────
        gap_stats = T.gap_statistics(df['open'], df['close'])
        position_size, size_note = self._size_position(
            entry_price, risk_per_share, current_equity, quality, required_quality, turnover,
        )
        if position_size < 1:
            return 'HOLD', {'reason': f'Sizing collapsed to zero shares ({size_note})'}

        # ── Gate 8: economics ────────────────────────────────────────────────
        p_win = self._win_probability(risk_per_share, target_price - entry_price, quality)
        min_notional = FLAT_CHARGE_PER_SELL / (R['max_flat_cost_bps'] / 1e4)

        provisional = {
            'entry_price': entry_price, 'stop_loss': stop_loss, 'target_price': target_price,
            'position_size': position_size, 'p_win_est': p_win,
            'gap_down_p90': gap_stats['down_gap_p90'], 'min_notional_inr': round(min_notional, 2),
        }
        ok, econ_reason, economics = self.validate_economics(provisional)
        if not ok:
            return 'HOLD', {'reason': f'Economics: {econ_reason}'}

        # Splitting is only permitted when every resulting row still clears the
        # flat-charge floor on its own. Three ₹4,000 rows pay the ₹20 DP charge
        # three times — 150 bps of pure friction the undivided position never
        # incurs. tranche_manager.build_tranches already accepts `enabled`, so
        # the caller passes this straight through.
        tranche_ok = bool(economics['notional'] >= 3.0 * min_notional)

        confidence = min(9, len(active_patterns))
        atr_value = float(latest['atr'])

        signal_details = {
            # ── v3-compatible keys, unchanged names and types ────────────────
            'symbol': symbol,
            'entry_price': round(entry_price, 2),
            'stop_loss': stop_loss,
            'target_price': target_price,
            'position_size': position_size,
            'risk': round(position_size * risk_per_share, 2),
            'reward': round(position_size * (target_price - entry_price), 2),
            'risk_reward_ratio': round(actual_rr, 2),
            'entry_type': primary,
            'patterns_triggered': active_patterns,
            'confidence': confidence,
            'market_regime': market_regime,
            'breakout': 'breakout' in active_patterns,
            'pullback': 'pullback' in active_patterns,
            'timestamp': latest.get('datetime', pd.Timestamp.now()),

            # ── v4 additions ─────────────────────────────────────────────────
            'quality_score': round(quality, 3),
            'quality_required': round(required_quality, 3),
            'quality_breakdown': q_parts,
            'p_win_est': round(p_win, 3),
            'stop_basis': geom['stop_basis'],
            'stop_sigma_mult': round(risk_per_share / max(sigma_abs, 1e-9), 2),
            'risk_pct_of_price': round(risk_pct * 100, 2),
            'time_exit_bars': geom['time_exit_bars'],
            'horizon_reachable_pct': round(geom['reach_pct'] * 100, 2),
            'sigma_daily_pct': round(sigma_d * 100, 2),
            'efficiency_ratio': round(er, 3),
            'price_band': price_band,
            'band_usage': round(band_usage, 3),
            'median_turnover_inr': round(turnover, 0),
            'gap_down_p90': round(gap_stats['down_gap_p90'], 4),
            'min_notional_inr': round(min_notional, 2),
            'tranche_ok': tranche_ok,
            'economics': economics,
            'size_note': size_note,

            'indicators': {
                'rsi': round(rsi, 2),
                'rsi_9': round(float(latest['rsi_9']), 2),
                'macd': round(float(latest['macd']), 4),
                'macd_hist': round(float(latest['macd_histogram']), 4),
                'adx': round(adx, 2),
                'plus_di': round(plus_di, 2),
                'minus_di': round(minus_di, 2),
                'atr': round(atr_value, 2),
                'sigma_abs': round(sigma_abs, 2),
                'bb_width': round(float(latest['bb_width']), 4),
                'cmf': round(float(latest['cmf']), 3),
                'stoch_k': round(float(latest['stoch_k']), 1),
                'efficiency_ratio': round(er, 3),
                'extension_atr': round(extension_atr, 2),
            },
            'fundamentals': {
                'pe_ratio': fundamentals.get('pe_ratio', 'N/A'),
                'debt_to_equity': fundamentals.get('debt_to_equity', 'N/A'),
                'roe': f"{fundamentals.get('roe_5yr', 0)*100:.1f}%",
                'revenue_growth': f"{fundamentals.get('revenue_cagr', 0)*100:.1f}%",
            },
        }
        return 'BUY', signal_details

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────
    def _build_frame(self, df):
        """One vectorised pass over the window; every series is reused below."""
        T = self.tech
        d = df.copy()
        close, high, low, volume = d['close'], d['high'], d['low'], d['volume']

        d['ema_9'] = T.calculate_ema(close, 9)
        d['ema_12'] = T.calculate_ema(close, 12)
        d['ema_20'] = T.calculate_ema(close, 20)
        d['ema_50'] = T.calculate_ema(close, 50)
        d['rsi'] = T.calculate_rsi(close, 14)
        d['rsi_9'] = T.calculate_rsi(close, 9)

        macd = T.calculate_macd(close)
        d['macd'], d['macd_signal'], d['macd_histogram'] = macd['macd'], macd['signal'], macd['histogram']

        bb = T.calculate_bollinger_bands(close, 20, 2)
        d['bb_upper'], d['bb_lower'], d['bb_middle'] = bb['upper'], bb['lower'], bb['middle']
        d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / d['bb_middle'].where(d['bb_middle'].abs() > 1e-12)

        d['atr'] = T.calculate_atr(high, low, close, 14)
        dmi = T.calculate_dmi(high, low, close, 14)
        d['adx'], d['plus_di'], d['minus_di'] = dmi['adx'], dmi['plus_di'], dmi['minus_di']

        d['volume_sma'] = T.calculate_volume_sma(volume, 20)
        stoch = T.calculate_stochastic(high, low, close, 14)
        d['stoch_k'], d['stoch_d'] = stoch['k'], stoch['d']
        d['wr'] = T.calculate_williams_r(high, low, close, 14)
        d['cmf'] = T.calculate_cmf(high, low, close, volume, 20)
        d['obv'] = T.calculate_obv(close, volume)

        d['sigma_d'] = T.daily_volatility_fraction(d, period=20, atr_period=14)
        d['er'] = T.efficiency_ratio(close, 20)
        d['turnover_med'] = T.median_traded_value(close, volume, 20)
        return d

    def _detect_patterns(self, d, latest, prev, prev2, sigma_abs, above_ema50):
        """
        The same nine triggers by name. Three are materially tightened, each
        because its realised record says the loose version was firing on noise:

          breakout   — v3 accepted a close 0.2% above the 20-day CLOSING high
                       on 1.2x volume, which is a rounding error above a
                       weaker reference level. v4 requires a close above the
                       20-day HIGH by 0.35σ, 1.6x volume, and a close in the
                       upper 40% of the bar's range. Realised v3 record:
                       n=13, 7.7% win, -0.517R.
          bb_squeeze — v3 fired on the first tick of band expansion. v4 also
                       requires the close above the upper band and expansion
                       confirmed against the squeeze's own floor. Realised:
                       n=4, 0% win, -0.750R.
          engulfing  — v4 requires the engulfing body to exceed the prior body
                       outright and close in the top third of its range.
                       Realised: n=2, 0% win, -1.000R.
        """
        vol, vol_sma = float(latest['volume']), float(latest['volume_sma'])
        vol_ratio = vol / vol_sma if vol_sma > 0 else 0.0
        close, open_ = float(latest['close']), float(latest['open'])
        bar_range = max(float(latest['high']) - float(latest['low']), 1e-9)
        close_position = (close - float(latest['low'])) / bar_range

        resistance_20d = float(d['high'].iloc[-21:-1].max())
        breakout = (
            close > resistance_20d + 0.35 * sigma_abs and
            vol_ratio > 1.5 and
            close_position > 0.60
        )

        near_ema20 = abs(close - float(latest['ema_20'])) / float(latest['ema_20']) < 0.025
        pullback = (
            near_ema20 and above_ema50 and
            close > float(prev['close']) and
            close_position > 0.55 and
            vol_ratio > 0.75
        )

        body_now = abs(close - open_)
        body_prev = abs(float(prev['close']) - float(prev['open']))
        engulfing = (
            close > open_ and
            float(prev['close']) < float(prev['open']) and
            close > float(prev['open']) and
            open_ <= float(prev['close']) and
            body_now > body_prev and
            close_position > 0.66
        )

        rsi_divergence = (
            close < float(prev2['close']) and
            float(latest['rsi_9']) > float(prev2['rsi_9']) and
            float(latest['rsi_9']) < 60 and
            above_ema50
        )

        stoch_cross = (
            float(prev['stoch_k']) < 35 and
            float(latest['stoch_k']) > float(latest['stoch_d']) and
            float(prev['stoch_k']) < float(prev['stoch_d'])
        )

        momentum_burst = (
            float(latest['rsi']) > 55 and float(prev['rsi']) <= 55 and
            float(latest['adx']) > float(prev['adx']) and
            vol_ratio > 1.25
        )

        bb_width_floor = float(d['bb_width'].iloc[-8:-1].min())
        bb_squeeze = (
            float(prev['bb_width']) <= bb_width_floor * 1.10 and
            float(latest['bb_width']) > float(prev['bb_width']) * 1.05 and
            close > float(latest['bb_upper']) and
            vol_ratio > 1.2
        )

        ema_cross = (
            float(prev['ema_12']) <= float(prev['ema_20']) and
            float(latest['ema_12']) > float(latest['ema_20']) and
            above_ema50
        )

        obv_rising = float(latest['obv']) > float(d['obv'].iloc[-11])
        cmf_accum = (
            float(latest['cmf']) > 0.08 and above_ema50 and
            obv_rising and vol_ratio > 0.9
        )

        return {
            'breakout': breakout, 'pullback': pullback, 'engulfing': engulfing,
            'rsi_divergence': rsi_divergence, 'stoch_cross': stoch_cross,
            'momentum_burst': momentum_burst, 'bb_squeeze': bb_squeeze,
            'ema_cross': ema_cross, 'cmf_accum': cmf_accum,
        }

    def _entry_quality(self, d, latest, sigma_abs, er, plus_di, minus_di,
                       extension_atr, above_ema20, above_ema50, ema50_rising, benchmark_df):
        """
        Six components in [0, 1], weight-averaged. Relative strength drops out
        cleanly and the rest renormalise when no benchmark is supplied, so a
        missing Nifty frame degrades precision rather than blocking the run.
        """
        q = {}
        q['trend'] = 0.34 * float(above_ema20) + 0.33 * float(above_ema50) + 0.33 * float(ema50_rising)
        q['dmi'] = float(np.clip((plus_di - minus_di) / 25.0, 0.0, 1.0))
        q['efficiency'] = float(np.clip((er - 0.15) / 0.35, 0.0, 1.0))

        vol_sma = float(latest['volume_sma'])
        vol_ratio = float(latest['volume']) / vol_sma if vol_sma > 0 else 0.0
        flow = float(np.clip((float(latest['cmf']) + 0.05) / 0.20, 0.0, 1.0))
        q['volume'] = float(np.clip(0.5 * np.clip((vol_ratio - 0.8) / 0.9, 0.0, 1.0) + 0.5 * flow, 0.0, 1.0))

        # Extension is scored, not just gated: at the EMA-20 the setup has the
        # whole move ahead of it; 2.5σ above it, most of the move is behind.
        q['non_extension'] = float(np.clip(1.0 - (extension_atr - 0.5) / 2.5, 0.0, 1.0))

        weights = {'trend': 0.20, 'dmi': 0.18, 'efficiency': 0.18, 'volume': 0.16, 'non_extension': 0.10}
        rs_weight = 0.18

        rs_value = np.nan
        rs_fn = getattr(self.tech, 'relative_strength', None)
        if rs_fn is not None and benchmark_df is not None and 'close' in getattr(benchmark_df, 'columns', []):
            rs_value = rs_fn(d['close'], benchmark_df['close'], period=20)
        if rs_value is not None and np.isfinite(rs_value):
            # 0 excess return scores 0.5; ±5% over 20 sessions saturates.
            q['rel_strength'] = float(np.clip(0.5 + rs_value / 0.10, 0.0, 1.0))
            weights['rel_strength'] = rs_weight
        total_w = sum(weights.values())

        score = sum(q[k] * w for k, w in weights.items()) / total_w
        return float(score), {k: round(v, 3) for k, v in q.items()}

    def _barrier_geometry(self, d, latest, entry_price, sigma_abs, sigma_d,
                          er, primary, market_regime, max_hold_days):
        """
        Stop, target and planned holding window, solved together.

        Stop: the deeper of (a) the structural level whose breach genuinely
        invalidates THIS pattern and (b) a volatility floor k_min·σ that widens
        as trend efficiency falls — then capped at k_max·σ so position size
        stays meaningful.

        Target: the smaller of the R-multiple ambition and what σ_d·√H can
        actually deliver inside the caller's exit clock. The whole reason v3
        produced 22 time-exits and 2 target-hits is that it only ever computed
        the first of those two numbers.

        Horizon: back-solved from the chosen target, H* = (target/σ_d)², the
        expected first-passage time for a σ_d-per-day random walk. Emitted as
        time_exit_bars so a caller that supports per-trade clocks can stop
        cutting trades that are still inside their own expected window.
        """
        R = self.R
        T = self.tech

        # ── Stop ─────────────────────────────────────────────────────────────
        k_min = R['k_stop_base'] + R['k_stop_noise_slope'] * (1.0 - float(np.clip(er, 0.0, 1.0)))
        if market_regime == 'BEAR':
            k_min *= 1.10
        k_min = float(min(k_min, R['k_stop_max']))

        swing_low, _ = T.recent_swing_low(d['low'], lookback=30, confirm=2)
        buffer = R['struct_buffer_atr'] * sigma_abs + 2 * NSEMicrostructure.TICK

        if primary == 'breakout':
            # A breakout's invalidation is a decisive return BELOW the level it
            # broke — retesting that level from above is ordinary behaviour, not
            # failure. v3 gave this pattern the tightest stop in the table
            # (1.2 ATR), which stopped it out on the retest and booked -0.517R
            # across 13 trades.
            level = float(d['high'].iloc[-21:-1].max())
            basis = 'broken resistance'
        elif primary == 'bb_squeeze':
            level = float(d['low'].iloc[-8:].min())
            basis = 'squeeze range low'
        elif primary == 'engulfing':
            level = float(min(latest['low'], d['low'].iloc[-2]))
            basis = 'engulfing bar low'
        elif primary in ('pullback', 'ema_cross', 'cmf_accum', 'momentum_burst'):
            candidates = [float(latest['ema_20'])]
            if swing_low is not None:
                candidates.append(swing_low)
            level = min(candidates)
            basis = 'swing low / EMA-20'
        else:                                            # stoch_cross, rsi_divergence
            level = swing_low if swing_low is not None else float(d['low'].iloc[-10:].min())
            basis = 'swing low'

        # ── Horizon affordability ────────────────────────────────────────────
        # Two independent constraints meet here, and solving them together is
        # what v3 never did:
        #   FLOOR   — the stop must clear the noise band, k_min·σ.
        #   CEILING — the stop must be payable inside the exit clock. The most
        #             the horizon can deliver is reach = σ_d·√H·reach_fraction,
        #             so any stop wider than reach/min_rr makes the required
        #             R:R arithmetically unreachable no matter how the trade
        #             behaves.
        # When floor exceeds ceiling the setup is not a bad trade, it is an
        # impossible one: this name's noise is larger than what this holding
        # period can pay for, and the correct action is to pass. Solving it
        # this way also means R:R feasibility is guaranteed by construction
        # rather than discovered by a rejection three steps later.
        horizon = max(int(max_hold_days), 3)
        reach_pct = sigma_d * math.sqrt(horizon) * R['reach_fraction']
        reach_dist = entry_price * reach_pct

        stop_floor_dist = k_min * sigma_abs
        stop_ceiling_dist = min(R['k_stop_max'] * sigma_abs, reach_dist / R['min_rr'])
        if stop_floor_dist > stop_ceiling_dist:
            return {'reason': f'Noise floor {k_min:.2f}σ exceeds the widest stop a {horizon}-bar '
                              f'horizon can pay for at {R["min_rr"]}:1'}

        structural_dist = entry_price - (level - buffer)
        stop_dist = float(np.clip(structural_dist, stop_floor_dist, stop_ceiling_dist))
        if stop_dist <= structural_dist - 1e-9:
            basis = f'{basis} (capped at {stop_dist/max(sigma_abs,1e-9):.2f}σ by the {horizon}-bar horizon)'
        elif stop_dist >= structural_dist + 1e-9:
            basis = f'{basis} (widened to the {k_min:.2f}σ noise floor)'

        stop_loss = NSEMicrostructure.round_to_tick(entry_price - stop_dist, mode='down')
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return {'reason': 'Structural stop resolves at or above entry'}

        # ── Target ───────────────────────────────────────────────────────────
        tgt_mult = R['tgt_mult_momentum'] if primary in MOMENTUM_PATTERNS else R['tgt_mult_reversion']
        if market_regime == 'BEAR':
            tgt_mult *= 0.85                              # take what a hostile tape offers

        target_dist = min(tgt_mult * risk_per_share, reach_dist)
        target_price = NSEMicrostructure.round_to_tick(entry_price + target_dist, mode='down')
        target_dist = target_price - entry_price
        if target_dist <= 0:
            return {'reason': 'Reachable target rounds back to entry'}

        # ── Planned holding window ───────────────────────────────────────────
        bars_needed = (target_dist / max(sigma_abs, 1e-9)) ** 2
        time_exit_bars = int(np.clip(math.ceil(bars_needed), 4, horizon))

        return {
            'stop_loss': round(stop_loss, 2),
            'target_price': round(target_price, 2),
            'risk_per_share': risk_per_share,
            'stop_basis': basis,
            'reach_pct': reach_pct,
            'time_exit_bars': time_exit_bars,
        }

    def _size_position(self, entry_price, risk_per_share, current_equity,
                       quality, required_quality, turnover):
        """
        Risk-fraction sizing, scaled by how far entry quality clears its own
        bar, then capped by single-name concentration and by participation in
        the scrip's median traded value.
        """
        R = self.R
        headroom = (quality - required_quality) / max(1.0 - required_quality, 1e-9)
        quality_scalar = float(np.clip(0.75 + 0.50 * headroom, 0.75, 1.25))

        risk_budget = current_equity * R['risk_pct_per_trade'] * quality_scalar
        size_by_risk = risk_budget / risk_per_share
        size_by_capital = (current_equity * R['max_capital_pct']) / entry_price
        size_by_liquidity = (turnover * R['max_adv_participation']) / entry_price

        binding = min(size_by_risk, size_by_capital, size_by_liquidity)
        note = ('risk budget' if binding == size_by_risk else
                'capital cap' if binding == size_by_capital else 'ADV participation')
        return int(binding), f'{note}, quality scalar {quality_scalar:.2f}'

    def _win_probability(self, risk_per_share, reward_per_share, quality):
        """
        Start from the driftless two-barrier result — P(touch +b before -a) =
        a/(a+b) — which is the correct no-edge baseline and makes every
        arrangement of barriers exactly zero-EV before costs. Then tilt it by
        entry quality, which is the only place claimed edge enters the model.

        Stating it this way keeps the arithmetic honest: with no edge the EV
        gate mathematically cannot pass, so a trade only clears when the
        quality score is doing real work. Calibrating edge_tilt against
        realised outcomes is the natural follow-up once enough closed trades
        carry a quality_score.
        """
        a, b = float(risk_per_share), float(reward_per_share)
        if a <= 0 or b <= 0:
            return 0.0
        p_base = a / (a + b)
        tilt = 1.0 + self.R['edge_tilt'] * (quality - 0.5) * 2.0
        return float(np.clip(p_base * tilt, 0.05, 0.85))

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def classify_market_regime(nifty_df):
        """BULL / NEUTRAL / BEAR from Nifty EMA-50 against SMA-200."""
        if nifty_df is None or len(nifty_df) < 200:
            return 'NEUTRAL'
        from technical_indicators import TechnicalIndicators
        tech = TechnicalIndicators()
        ema50 = tech.calculate_ema(nifty_df['close'], 50)
        s200 = tech.calculate_sma(nifty_df['close'], 200)
        c, e, s = nifty_df['close'].iloc[-1], ema50.iloc[-1], s200.iloc[-1]
        if c > e and e > s:
            return 'BULL'
        if c < e and e < s:
            return 'BEAR'
        return 'NEUTRAL'
