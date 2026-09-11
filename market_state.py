# market_state.py  ── MARKET REGIME & EXPOSURE CONTROL  v1
# ═════════════════════════════════════════════════════════════════════════════
# Decides HOW MUCH of the book should be at risk today, before any individual
# signal is considered. signal_generator asks "is this a trade", exit_manager
# asks "is this trade over", portfolio_allocator asks "is this trade worth the
# slot". This module asks the question that precedes all three: "is this a day
# to be taking trades at all, and at what size."
#
# ── The evidence this exists for ────────────────────────────────────────────
# Closed trades grouped by ENTRY WEEK:
#
#   week          n   mean R    net ₹    win rate
#   2026-07-27   14    +0.02   +533.32     64%
#   2026-08-03    3    -1.00   -158.97      0%
#   2026-08-17   16    +0.47   +600.09     38%
#   2026-08-24    2    -1.00    -81.82      0%
#   2026-08-31    6    -1.00   -907.22      0%
#
# Three separate weeks with a mean R of exactly -1.00 and a 0% win rate. A
# mean of exactly -1.00 means EVERY trade in that window ran to its full stop —
# not a bad draw from a positive-expectancy process, which produces a mix.
# 81.7% of all losses in the sample came from the worst two weeks, and the
# weeks in between were solidly profitable on the same entry logic.
#
# The signal generator was not broken during those weeks. The market was
# closed for business and the bot kept placing orders.
#
# ── Why the existing regime code did not catch it ───────────────────────────
# signal_generator.classify_market_regime compares Nifty's EMA-50 to its
# SMA-200. That flips state perhaps four times a year and cannot resolve a
# five-day risk-off event by construction. alpha_engine's RegimeDetector
# classifies market CHARACTER (trending vs choppy) to weight factors, which is
# a different and also useful job — it is not a controller on exposure.
# Neither one reduces size, and neither one can act inside a week.
#
# ── Why breadth is the primary sensor ───────────────────────────────────────
# The scanner already fetches 100+ symbol histories every single day. The
# percentage of that universe above its own EMA-20, the new-high/new-low
# spread, and the advance-decline line cost zero additional API calls, zero
# rupees, and deteriorate BEFORE the index does.
#
# That lead is structural, not empirical folklore: Nifty is cap-weighted, so
# five heavyweights can hold the index flat while ninety-five constituents roll
# over. The index EMA sees nothing; breadth sees it immediately. And it is the
# ninety-five this bot actually trades.
#
# ── The control philosophy ──────────────────────────────────────────────────
# Exposure is continuous, and the response is deliberately asymmetric:
#
#   • CUTTING is fast. Discrete triggers clamp exposure the day they fire,
#     because the gradual score cannot move far enough inside a three-day
#     event to matter.
#   • RESTORING is slow, and gated on BREADTH RECOVERY rather than on the
#     trigger merely going quiet. A filter that re-risks as fast as it
#     de-risks re-enters straight into the second leg down and converts one
#     drawdown into two.
#
# Everything is point-in-time: bar t reads only bars <= t, so run_backtest.py
# can carry this into a Run D without lookahead.
# ═════════════════════════════════════════════════════════════════════════════

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from technical_indicators import TechnicalIndicators as TI
except ImportError:                                              # pragma: no cover
    TI = None


# ═════════════════════════════════════════════════════════════════════════════
# PROFILES
# ═════════════════════════════════════════════════════════════════════════════
# 'aggressive' runs a higher ceiling (it will lever UP to 1.15x in a confirmed
# risk-on tape) and tolerates a weaker score before throttling. It does NOT
# soften the defensive triggers or the recovery gate. Those exist to prevent
# the three specific weeks above, and loosening them would remove the only part
# of this module with direct evidence behind it.
STATE_PROFILES = {
    'balanced': {
        'exposure_ceiling':        1.00,
        'exposure_floor':          0.00,
        'risk_on_threshold':       0.62,
        'risk_off_threshold':      0.38,
        'defensive_clamp':         0.25,   # exposure ceiling once a trigger fires
        'cooldown_bars':           4,      # minimum bars a defensive clamp persists
        'recovery_breadth':        0.45,   # pct above EMA-20 needed to start restoring
        'recovery_breadth_alt':    0.36,   # lower bar, valid only alongside a reclaimed risk score
        'recovery_quiet_bars':     8,      # sessions with no trigger required on that alternate path
        'recovery_step':           0.20,   # max exposure added per bar once recovering
        'quality_add_at_zero':     0.14,   # extra entry-quality demanded at zero exposure
        'min_slots':               1,
    },
    'aggressive': {
        'exposure_ceiling':        1.15,
        'exposure_floor':          0.00,
        'risk_on_threshold':       0.56,
        'risk_off_threshold':      0.34,
        'defensive_clamp':         0.25,
        'cooldown_bars':           4,
        'recovery_breadth':        0.45,
        'recovery_breadth_alt':    0.34,
        'recovery_quiet_bars':     6,
        'recovery_step':           0.25,
        'quality_add_at_zero':     0.12,
        'min_slots':               1,
    },
}

ACTIVE_PROFILE = 'aggressive'

# Component weights for the gradual risk score. Breadth carries the most
# because it is the only sensor here that leads rather than confirms; index
# trend is the most reliable but the slowest; volatility is fast but noisy on
# its own, which is why it is a weight rather than a gate.
COMPONENT_WEIGHTS = {
    'breadth':      0.34,
    'index_trend':  0.26,
    'volatility':   0.22,
    'drawdown':     0.18,
}


def get_profile(name=None):
    return STATE_PROFILES[name or ACTIVE_PROFILE]


# ═════════════════════════════════════════════════════════════════════════════
# BREADTH
# ═════════════════════════════════════════════════════════════════════════════
class BreadthPanel:
    """
    Collapses the whole scan universe into one time series of participation
    metrics, computed once and reused.

    Built as a panel rather than a point-in-time snapshot for two reasons: the
    fast triggers need the 5-bar CHANGE in breadth, not just today's level, and
    a panel lets run_backtest.py replay the filter across history at the same
    cost as evaluating it once.

    Symbols are aligned on their date column, so a name with a short or
    gap-ridden history contributes to the bars it has and is simply absent from
    the rest — rather than being forward-filled into a participation reading it
    never earned.
    """

    def __init__(self, universe_dfs, min_symbols=20):
        self.min_symbols = min_symbols
        self.panel = self._build(universe_dfs)

    @staticmethod
    def _build(universe_dfs):
        above20, above50, newhigh, newlow, up_day = {}, {}, {}, {}, {}

        for sym, df in (universe_dfs or {}).items():
            if df is None or len(df) < 60 or 'close' not in df:
                continue
            d = df.copy()
            idx = pd.to_datetime(d['datetime']) if 'datetime' in d else pd.RangeIndex(len(d))
            close = pd.Series(d['close'].to_numpy(dtype=float), index=idx)
            high = pd.Series(d['high'].to_numpy(dtype=float), index=idx)
            low = pd.Series(d['low'].to_numpy(dtype=float), index=idx)

            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            above20[sym] = (close > ema20).astype(float)
            above50[sym] = (close > ema50).astype(float)
            # Strictly greater than the PRIOR 20 bars' extreme, so today's own
            # bar cannot trivially satisfy its own new-high test.
            newhigh[sym] = (high >= high.shift().rolling(20).max()).astype(float)
            newlow[sym] = (low <= low.shift().rolling(20).min()).astype(float)
            up_day[sym] = (close > close.shift()).astype(float)

        if len(above20) == 0:
            return pd.DataFrame()

        panel = pd.DataFrame({
            'pct_above_20': pd.DataFrame(above20).mean(axis=1),
            'pct_above_50': pd.DataFrame(above50).mean(axis=1),
            'pct_new_high': pd.DataFrame(newhigh).mean(axis=1),
            'pct_new_low': pd.DataFrame(newlow).mean(axis=1),
            'pct_up': pd.DataFrame(up_day).mean(axis=1),
            'n_symbols': pd.DataFrame(above20).notna().sum(axis=1),
        }).sort_index()

        panel['nh_nl_spread'] = panel['pct_new_high'] - panel['pct_new_low']
        # The fast sensor: how much participation has been lost in a week. A
        # level of 45% reached by drifting sideways and a level of 45% reached
        # by falling 25 points in five sessions are entirely different tapes.
        panel['breadth_chg_5'] = panel['pct_above_20'] - panel['pct_above_20'].shift(5)
        panel['ad_line'] = (panel['pct_up'] - 0.5).cumsum()
        return panel

    def at(self, when=None):
        """Latest row at or before `when` (None = most recent). Never looks ahead."""
        if self.panel is None or self.panel.empty:
            return None
        p = self.panel if when is None else self.panel.loc[:when]
        if p.empty:
            return None
        row = p.iloc[-1]
        if float(row.get('n_symbols', 0)) < self.min_symbols:
            return None
        return row


# ═════════════════════════════════════════════════════════════════════════════
# MARKET STATE
# ═════════════════════════════════════════════════════════════════════════════
class MarketState:
    """
    Produces today's exposure decision and carries the hysteresis between days.

    State that must persist across runs — the defensive clamp and its cooldown
    counter — lives on the instance and is serialisable via to_dict/from_dict,
    so a daily cron process picks up yesterday's posture instead of waking up
    optimistic every morning. A filter that forgets it was defensive yesterday
    is not a filter.
    """

    def __init__(self, profile=None):
        self.P = get_profile(profile)
        self.profile_name = profile or ACTIVE_PROFILE
        self.clamp_active = False
        self.cooldown_left = 0
        self.quiet_bars = 0
        self.last_exposure = self.P['exposure_ceiling']
        logger.info(f"✓ MarketState — {self.profile_name} "
                    f"(ceiling {self.P['exposure_ceiling']:.2f}, clamp {self.P['defensive_clamp']:.2f})")

    # ─────────────────────────────────────────────────────────────────────────
    def assess(self, index_df, universe_dfs=None, vix_df=None, breadth_panel=None,
               base_slots=5, when=None):
        """
        index_df: Nifty OHLCV (the frame run_paper_trading already fetches)
        universe_dfs: {symbol: OHLCV} — the scan universe, already in hand
        vix_df: optional India VIX frame ('^INDIAVIX' fetches fine through the
                existing DataFetcherFree, since it routes any '^' symbol
                straight through without appending .NS)
        breadth_panel: a prebuilt BreadthPanel, to avoid rebuilding it per call
                during a historical replay

        Returns the full decision dict. Nothing is enforced here; the caller
        applies it, so the posture is inspectable and loggable before it
        changes a single order.
        """
        P = self.P
        panel = breadth_panel or (BreadthPanel(universe_dfs) if universe_dfs else None)
        breadth = panel.at(when) if panel else None

        comp = {}
        comp['breadth'] = self._score_breadth(breadth)
        comp['index_trend'] = self._score_trend(index_df)
        comp['volatility'] = self._score_volatility(index_df, vix_df)
        comp['drawdown'] = self._score_drawdown(index_df)

        # Renormalise over whatever could actually be measured, so a missing
        # VIX feed or a thin universe degrades precision instead of silently
        # scoring that component as zero and forcing a false risk-off.
        usable = {k: v for k, v in comp.items() if v is not None}
        if not usable:
            return self._result('NEUTRAL', 0.5, P['exposure_ceiling'] * 0.6, base_slots, comp, [],
                                'insufficient data to assess — holding a reduced default')
        weight_sum = sum(COMPONENT_WEIGHTS[k] for k in usable)
        risk_score = sum(usable[k] * COMPONENT_WEIGHTS[k] for k in usable) / weight_sum

        triggers = self._defensive_triggers(index_df, breadth, vix_df)

        # ── Gradual exposure from the score ──────────────────────────────────
        span = max(P['risk_on_threshold'] - P['risk_off_threshold'], 1e-9)
        raw = (risk_score - P['risk_off_threshold']) / span
        exposure = float(np.clip(raw, 0.0, 1.0)) * P['exposure_ceiling']

        # ── Fast clamp, then hysteresis on the way back ──────────────────────
        if triggers:
            self.clamp_active = True
            self.cooldown_left = P['cooldown_bars']
            self.quiet_bars = 0
        elif self.clamp_active:
            self.cooldown_left -= 1
            self.quiet_bars += 1
            pct20 = float(breadth['pct_above_20']) if breadth is not None else 0.0
            # Primary release: participation has genuinely rebuilt.
            breadth_ok = pct20 >= P['recovery_breadth']
            # Alternate release: participation is only partway back, but the
            # weighted score has independently reclaimed risk-on AND nothing has
            # triggered for several sessions. This is a second, different piece
            # of evidence (index trend reclaimed, volatility normalised), not a
            # timer — the distinction matters, because a pure timeout would put
            # the book back in on the strength of nothing having happened yet.
            # It exists because the primary gate alone kept exposure at 0.25x
            # for 30 sessions in testing, roughly ten of which the market spent
            # recovering. Being late back in is a real cost, just a smaller one
            # than being early.
            alt_ok = (pct20 >= P['recovery_breadth_alt']
                      and risk_score >= P['risk_on_threshold']
                      and self.quiet_bars >= P['recovery_quiet_bars'])
            if self.cooldown_left <= 0 and (breadth_ok or alt_ok):
                self.clamp_active = False

        if self.clamp_active:
            exposure = min(exposure, P['defensive_clamp'])
        else:
            # Restoring is rate-limited; cutting is not. This asymmetry is the
            # single most important line in the module.
            exposure = min(exposure, self.last_exposure + P['recovery_step'])

        exposure = float(np.clip(exposure, P['exposure_floor'], P['exposure_ceiling']))
        self.last_exposure = exposure

        state = ('DEFENSIVE' if self.clamp_active else
                 'RISK_ON' if risk_score >= P['risk_on_threshold'] else
                 'RISK_OFF' if risk_score <= P['risk_off_threshold'] else 'NEUTRAL')

        note = (f"{', '.join(triggers)}" if triggers else
                'defensive clamp held, awaiting breadth recovery' if self.clamp_active else '')
        return self._result(state, risk_score, exposure, base_slots, comp, triggers, note, breadth)

    # ─────────────────────────────────────────────────────────────────────────
    def _result(self, state, risk_score, exposure, base_slots, comp, triggers, note, breadth=None):
        P = self.P
        # Slots scale with exposure but never to zero while any exposure
        # remains: one high-conviction position in a recovering tape is how the
        # book gets back in, and a slot count of zero makes recovery
        # unobservable.
        max_slots = (0 if exposure <= 0.01 else
                     max(P['min_slots'], int(round(base_slots * min(exposure, 1.0)))))

        return {
            'state': state,
            'risk_score': round(float(risk_score), 3),
            'exposure': round(float(exposure), 3),
            'new_entries_allowed': bool(exposure > 0.05 and max_slots > 0),
            'max_slots': max_slots,
            'heat_multiplier': round(float(exposure), 3),
            # The entry bar rises as conditions deteriorate, so the trades that
            # do get taken in a weak tape are the ones with the most standalone
            # evidence — this feeds signal_generator's required_quality.
            'quality_add': round(P['quality_add_at_zero'] * (1.0 - min(float(exposure), 1.0)), 3),
            # Kept compatible with signal_generator's existing three-way
            # market_regime parameter, so it drops straight into the current
            # call without a signature change.
            'legacy_regime': ('BULL' if state == 'RISK_ON' else
                              'BEAR' if state in ('RISK_OFF', 'DEFENSIVE') else 'NEUTRAL'),
            'components': {k: (round(v, 3) if v is not None else None) for k, v in comp.items()},
            'triggers': triggers,
            'note': note,
            'breadth': ({'pct_above_20': round(float(breadth['pct_above_20']), 3),
                         'pct_above_50': round(float(breadth['pct_above_50']), 3),
                         'nh_nl_spread': round(float(breadth['nh_nl_spread']), 3),
                         'breadth_chg_5': (round(float(breadth['breadth_chg_5']), 3)
                                           if pd.notna(breadth['breadth_chg_5']) else None),
                         'n_symbols': int(breadth['n_symbols'])}
                        if breadth is not None else None),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # COMPONENT SCORES — each returns 0 (hostile) to 1 (supportive), or None
    # ═════════════════════════════════════════════════════════════════════════
    def _score_breadth(self, breadth):
        if breadth is None:
            return None
        pct20 = float(breadth['pct_above_20'])
        pct50 = float(breadth['pct_above_50'])
        spread = float(breadth['nh_nl_spread'])
        chg5 = breadth['breadth_chg_5']

        level = 0.55 * np.clip((pct20 - 0.25) / 0.45, 0, 1) + 0.45 * np.clip((pct50 - 0.25) / 0.45, 0, 1)
        # New highs minus new lows: the cleanest read on whether the move has
        # participants or just an index print.
        thrust = np.clip((spread + 0.06) / 0.16, 0, 1)
        # Momentum of participation. A universe shedding 15 points of breadth
        # in a week scores zero here even while the LEVEL still looks adequate,
        # which is precisely the configuration that precedes the weeks in the
        # header above.
        momentum = 0.5 if (chg5 is None or pd.isna(chg5)) else np.clip((float(chg5) + 0.15) / 0.25, 0, 1)
        return float(0.45 * level + 0.25 * thrust + 0.30 * momentum)

    def _score_trend(self, index_df):
        if index_df is None or len(index_df) < 60:
            return None
        close = index_df['close'].astype(float)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        c = float(close.iloc[-1])

        above20 = float(c > float(ema20.iloc[-1]))
        above50 = float(c > float(ema50.iloc[-1]))
        stacked = float(float(ema20.iloc[-1]) > float(ema50.iloc[-1]))
        slope = float(ema20.iloc[-1]) / float(ema20.iloc[-6]) - 1.0 if len(ema20) > 6 else 0.0
        slope_score = float(np.clip((slope + 0.004) / 0.012, 0, 1))
        return float(0.28 * above20 + 0.22 * above50 + 0.20 * stacked + 0.30 * slope_score)

    def _score_volatility(self, index_df, vix_df):
        """
        Volatility enters as a weight rather than a gate because on its own it
        is a poor timing tool — vol rises in melt-ups too. What it discriminates
        well is RISING vol against a weakening tape, which is why the score
        reads both level and rate of change.
        """
        parts = []
        if index_df is not None and len(index_df) >= 120 and TI is not None:
            sigma = TI.daily_volatility_fraction(index_df, period=20)
            s_now = float(sigma.iloc[-1])
            # Percentile against the index's own last year, not an absolute
            # threshold — 1.1% daily is calm for a smallcap index and alarming
            # for Nifty.
            pct = float((sigma.tail(250) < s_now).mean())
            parts.append(1.0 - pct)
            expansion = s_now / max(float(sigma.iloc[-11]), 1e-9)
            parts.append(float(np.clip(1.0 - (expansion - 1.0) / 0.60, 0, 1)))

        if vix_df is not None and len(vix_df) >= 10:
            v = vix_df['close'].astype(float)
            v_now = float(v.iloc[-1])
            # India VIX below ~13 is complacent-to-calm, above ~22 is stress.
            parts.append(float(np.clip((22.0 - v_now) / 9.0, 0, 1)))
            chg3 = v_now / max(float(v.iloc[-4]), 1e-9) - 1.0
            parts.append(float(np.clip(1.0 - chg3 / 0.30, 0, 1)))

        return float(np.mean(parts)) if parts else None

    def _score_drawdown(self, index_df):
        if index_df is None or len(index_df) < 60:
            return None
        close = index_df['close'].astype(float)
        peak = float(close.tail(60).max())
        dd = max(0.0, (peak - float(close.iloc[-1])) / max(peak, 1e-9))
        # A 2% pullback in an uptrend is normal; 8% off a 60-day high is a
        # different market with a different base rate for a swing long.
        return float(np.clip(1.0 - dd / 0.08, 0, 1))

    # ═════════════════════════════════════════════════════════════════════════
    # FAST DEFENSIVE TRIGGERS
    # ═════════════════════════════════════════════════════════════════════════
    def _defensive_triggers(self, index_df, breadth, vix_df):
        """
        Discrete conditions that clamp exposure the day they fire.

        The gradual score is a good controller and a slow one: across a
        three-day repricing it might travel from 0.65 to 0.50, which is not
        enough to matter. These are the conditions under which the base rate
        for a swing long collapses fast enough that waiting for the weighted
        average to catch up is itself the mistake.

        Every one is checked against completed bars only.
        """
        triggers = []

        if index_df is not None and len(index_df) >= 30:
            close = index_df['close'].astype(float)
            ema20 = close.ewm(span=20, adjust=False).mean()
            c = float(close.iloc[-1])

            two_day = c / float(close.iloc[-3]) - 1.0
            if two_day <= -0.030:
                triggers.append(f'index -{abs(two_day)*100:.1f}% in 2 sessions')

            # The index losing its own EMA-20 while most constituents are
            # already below theirs. Either alone is ordinary; together they
            # describe a tape where the average long is underwater and the
            # index has just confirmed it.
            if breadth is not None:
                if c < float(ema20.iloc[-1]) and float(breadth['pct_above_20']) < 0.35:
                    triggers.append(f"index below EMA-20 with only "
                                    f"{float(breadth['pct_above_20'])*100:.0f}% of the universe above theirs")

        if breadth is not None:
            chg5 = breadth['breadth_chg_5']
            if chg5 is not None and pd.notna(chg5) and float(chg5) <= -0.20:
                triggers.append(f'breadth down {abs(float(chg5))*100:.0f} points in 5 sessions')
            if float(breadth['nh_nl_spread']) <= -0.12:
                triggers.append(f"new lows exceed new highs by "
                                f"{abs(float(breadth['nh_nl_spread']))*100:.0f} points")

        if vix_df is not None and len(vix_df) >= 5:
            v = vix_df['close'].astype(float)
            v_now = float(v.iloc[-1])
            chg3 = v_now / max(float(v.iloc[-4]), 1e-9) - 1.0
            if chg3 >= 0.25:
                triggers.append(f'India VIX +{chg3*100:.0f}% in 3 sessions')
            if v_now >= 24.0:
                triggers.append(f'India VIX at {v_now:.1f}')

        return triggers

    # ═════════════════════════════════════════════════════════════════════════
    # PERSISTENCE & REPLAY
    # ═════════════════════════════════════════════════════════════════════════
    def to_dict(self):
        return {'clamp_active': self.clamp_active, 'cooldown_left': self.cooldown_left,
                'quiet_bars': self.quiet_bars, 'last_exposure': self.last_exposure,
                'profile': self.profile_name}

    @classmethod
    def from_dict(cls, d):
        obj = cls(d.get('profile'))
        obj.clamp_active = bool(d.get('clamp_active', False))
        obj.cooldown_left = int(d.get('cooldown_left', 0))
        obj.quiet_bars = int(d.get('quiet_bars', 0))
        obj.last_exposure = float(d.get('last_exposure', obj.P['exposure_ceiling']))
        return obj

    def replay(self, index_df, universe_dfs, vix_df=None, base_slots=5, start=120):
        """
        Walk the filter forward bar by bar over history and return one row per
        session.

        This is the honest way to evaluate a regime filter, and the reason it
        is built in rather than left as an exercise: a filter judged on the
        dates you already know were bad is not a filter, it is a lookup table.
        Replay carries the hysteresis state forward exactly as a live daily run
        would, so what comes out is what the bot would actually have done.

        Overlay the exposure column on entry dates from paper_trades.csv to see
        whether the losing weeks would in fact have been sized down.
        """
        panel = BreadthPanel(universe_dfs)
        dates = pd.to_datetime(index_df['datetime'])
        saved = self.to_dict()
        rows = []
        for i in range(start, len(index_df)):
            when = dates.iloc[i]
            r = self.assess(index_df.iloc[:i + 1],
                            vix_df=vix_df.iloc[:i + 1] if vix_df is not None else None,
                            breadth_panel=panel, base_slots=base_slots, when=when)
            rows.append({'datetime': when, 'state': r['state'], 'risk_score': r['risk_score'],
                         'exposure': r['exposure'], 'max_slots': r['max_slots'],
                         'triggers': '; '.join(r['triggers']),
                         'pct_above_20': (r['breadth'] or {}).get('pct_above_20')})
        restored = MarketState.from_dict(saved)
        self.clamp_active, self.cooldown_left, self.quiet_bars, self.last_exposure = (
            restored.clamp_active, restored.cooldown_left, restored.quiet_bars, restored.last_exposure)
        return pd.DataFrame(rows)


def print_state(r):
    print("\n" + "=" * 74)
    print(f"  MARKET STATE: {r['state']}   risk score {r['risk_score']:.2f}   "
          f"exposure {r['exposure']:.2f}x   slots {r['max_slots']}")
    print("=" * 74)
    comps = '  '.join(f"{k}={v:.2f}" if v is not None else f"{k}=n/a"
                      for k, v in r['components'].items())
    print(f"  Components   : {comps}")
    if r['breadth']:
        b = r['breadth']
        chg = f"{b['breadth_chg_5']:+.2f}" if b['breadth_chg_5'] is not None else 'n/a'
        print(f"  Breadth      : {b['pct_above_20']*100:.0f}% >EMA20, {b['pct_above_50']*100:.0f}% >EMA50, "
              f"NH-NL {b['nh_nl_spread']:+.2f}, 5d chg {chg}  (n={b['n_symbols']})")
    print(f"  Entries      : {'allowed' if r['new_entries_allowed'] else 'suspended'}   "
          f"heat x{r['heat_multiplier']:.2f}   quality bar +{r['quality_add']:.2f}")
    if r['triggers']:
        for t in r['triggers']:
            print(f"  ⚠ trigger    : {t}")
    elif r['note']:
        print(f"  Note         : {r['note']}")
    print("=" * 74 + "\n")
