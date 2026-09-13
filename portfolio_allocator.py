# portfolio_allocator.py  ── CAPITAL ALLOCATION & SLOT COMPETITION  v1
# ═════════════════════════════════════════════════════════════════════════════
# The layer that decides WHICH signals get capital, HOW MUCH each gets, and
# WHICH incumbents lose their slot. signal_generator.py answers "is this a
# trade"; exit_manager.py answers "is this trade over"; nothing until now
# answered "is this trade worth the slot more than what is already in it."
#
# That question is the one that actually compounds an account. With five
# economically viable slots and ~1.6 candidates a day, the same signal stream
# and the same exits produce very different equity curves depending only on
# allocation. It is also the last place in this system where a decision is
# still being made by a number with no demonstrated predictive value.
#
# ── What this replaces, and the evidence ────────────────────────────────────
#
# 1. REPLACEMENT DECIDED BY AN UNVALIDATED SCORE.
#    run_paper_trading.py evicts the weakest incumbent when a new candidate's
#    composite alpha score beats it by REPLACE_SCORE_MULTIPLE (1.40). But the
#    50 closed rows show that score separating nothing: Tier 2 mean -0.086R
#    against Tier 3 mean -0.072R, with Tier 3 winning MORE often (45% vs 29%,
#    n=41). A 40% margin on a quantity with no measured discrimination is a
#    precise-looking coin flip. Worse, it compares a FORECAST for the
#    candidate against a FORECAST MADE AT ENTRY for the incumbent — a stale
#    opinion about a position that has since had days to prove itself.
#    Here both sides are re-expressed as the same forward quantity: expected
#    return on capital per slot-day, from today's price, today's stop and
#    today's thesis health.
#
# 2. SWITCHING TREATED AS FREE.
#    Every eviction pays an exit cost on the incumbent and an entry cost on the
#    candidate — ~48 bps round trip at the current calibration, and the ₹20
#    flat depository charge lands whole on whichever row is smaller. A swap
#    that improves expected return by less than the cost of making it is a
#    strictly losing trade that looks like optimisation. The hurdle below makes
#    the switch pay for itself inside the new trade's own horizon.
#
# 3. PROTECTIONS THAT PROTECT THE WRONG POSITIONS.
#    PROTECT_PROFIT_PCT shields any position up more than 3% unrealised — even
#    one whose EMA-20 has broken, whose -DI has taken the lead and which has
#    gone nowhere for eight sessions. Being up 3% is a fact about the past.
#    Here the protections stand while the thesis stands and stand down once
#    exit_manager flags the position stagnant or its health signals fire.
#
# 4. SIZE DECIDED INDEPENDENTLY OF EDGE.
#    A flat risk_pct_per_trade stakes the same fraction on a 0.48-probability
#    2.4:1 setup and a 0.38-probability 1.6:1 setup. The second has negative
#    expectancy and still gets full size. Fractional Kelly on the estimates
#    signal_generator already emits (p_win_est, R:R) makes stake respond to
#    edge, and returns a negative fraction — an automatic rejection — exactly
#    when the arithmetic says there is no edge to stake.
#
# 5. FAIR-SHARE SLICING THAT DESTROYED NOTIONAL.
#    free_cash / remaining_slots x FAIR_SHARE_FLEX produced the ₹1,200 rows
#    that paid 195 bps. Sizing here starts from edge and is then clipped by
#    heat, concentration and cash — and a position that cannot clear the
#    economic floor after those clips is declined rather than shrunk into one
#    that cannot pay for itself.
#
# ── The currency everything is priced in ────────────────────────────────────
# Expected return on capital per slot-day:
#
#       roc_per_day = E[net rupees] / notional / expected_days_held
#
# Candidates and incumbents are both measured this way, forward-looking from
# today. It is the only unit in which "take this new trade" and "keep that old
# one" are the same kind of statement, and it is the quantity whose sum over
# the period IS the account's growth rate.
# ═════════════════════════════════════════════════════════════════════════════

import logging
import math

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from trading_costs import round_trip_commission, PCT_COST_PER_LEG, FLAT_CHARGE_PER_SELL
except ImportError:                                              # pragma: no cover
    PCT_COST_PER_LEG, FLAT_CHARGE_PER_SELL = 0.0013, 20.0

    def round_trip_commission(entry_price, exit_price, position_size):
        return round((entry_price + exit_price) * position_size * PCT_COST_PER_LEG
                     + FLAT_CHARGE_PER_SELL, 2)


# ═════════════════════════════════════════════════════════════════════════════
# PROFILES
# ═════════════════════════════════════════════════════════════════════════════
ALLOCATOR_PROFILES = {
    'balanced': {
        'kelly_lambda':          0.25,   # fraction of full Kelly actually staked
        'max_risk_pct':          0.040,  # hard ceiling on any single stake, regardless of Kelly
        'max_capital_pct':       0.30,
        'max_portfolio_heat':    0.16,   # summed open risk as a share of equity
        'heat_floor':            0.35,   # heat budget never throttles below this multiple
        'drawdown_soft_max':     0.25,   # drawdown at which the throttle reaches its floor
        'switch_margin':         2.0,    # a swap must out-earn its own cost by this multiple
        'max_correlation':       0.75,
        'max_per_sector':        3,
        'min_edge_roc_per_day':  0.0006, # ~0.06% of deployed capital per day, net
        'settlement_days':       1,
    },
    'aggressive': {
        'kelly_lambda':          0.35,
        'max_risk_pct':          0.055,
        'max_capital_pct':       0.40,
        'max_portfolio_heat':    0.22,
        'heat_floor':            0.40,
        'drawdown_soft_max':     0.30,
        'switch_margin':         1.5,
        'max_correlation':       0.82,   # concentration in one theme is a deliberate choice here
        'max_per_sector':        4,
        'min_edge_roc_per_day':  0.0004,
        'settlement_days':       1,
    },
    'growth': {
        # Matched to signal_generator's 'growth' calibration. Kelly lambda is
        # raised toward the level the Monte Carlo put peak median growth at,
        # and max_risk_pct is lifted so the Kelly fraction is what binds rather
        # than a ceiling set for a different book. Heat rises because two or
        # three positions at 3.5% risk each plus pyramid adds needs room —
        # without it the heat cap silently re-imposes the timid configuration
        # this profile exists to replace.
        'kelly_lambda':          0.45,
        'max_risk_pct':          0.045,
        'max_capital_pct':       0.35,
        'max_portfolio_heat':    0.14,   # 3 positions at ~2% + pyramid adds
        'heat_floor':            0.40,
        'drawdown_soft_max':     0.30,
        'switch_margin':         1.4,
        'max_correlation':       0.85,
        'max_per_sector':        2,
        'min_edge_roc_per_day':  0.0004,
        'settlement_days':       1,
    },
}

ACTIVE_PROFILE = 'growth'

# The alpha composite enters the ranking as a small tilt rather than a gate.
# It may well carry signal that 41 trades cannot resolve, so discarding it
# outright would throw away a real prior; trusting it to order the book would
# repeat the mistake the tier statistics already exposed. A +/-7.5% tilt lets
# it break ties between economically similar candidates and nothing more.
# Raise ALPHA_TILT once a tier breakdown on >=100 trades shows separation.
ALPHA_TILT = 0.15

# Sentiment enters the same way and for the same reason, with its own bound.
# The two are kept separate rather than summed into one "score" because they
# fail independently: alpha is a price/volume composite that is always
# computable, sentiment is a news read that is frequently absent for a midcap
# and must be able to contribute nothing without dragging a candidate toward
# the middle. A symbol with no usable headlines gets no tilt at all, not a
# neutral 50 — the distinction between "average news" and "no news" matters,
# and sentiment_engine's ranker already excludes zero-confidence symbols from
# its percentiles for exactly this reason.
SENTIMENT_TILT = 0.12

# Cross-sectional momentum already gates the scan, so by allocation time every
# candidate is in the top tier. The tilt only orders WITHIN that tier, which is
# why it is the largest of the three: it is the one input with published
# out-of-sample support behind it, and it is discriminating among names that
# have already cleared the same bar.
MOMENTUM_TILT = 0.20


def get_profile(name=None):
    return ALLOCATOR_PROFILES[name or ACTIVE_PROFILE]


# ═════════════════════════════════════════════════════════════════════════════
# EDGE ARITHMETIC
# ═════════════════════════════════════════════════════════════════════════════
def kelly_fraction(p_win, reward_per_share, risk_per_share):
    """
    Full-Kelly fraction of bankroll to RISK on a binary payoff.

        f* = (p*b - q) / b,   b = reward/risk,  q = 1 - p

    Two properties earn its place over a flat risk percentage:

      • It responds to edge. A 0.48/2.4:1 setup and a 0.38/1.6:1 setup are
        staked differently because they deserve to be.
      • It goes NEGATIVE precisely when p*b < q — when the arithmetic says
        there is no edge. The caller reads that as "decline", which makes the
        no-edge case a structural rejection rather than a judgement call.

    Full Kelly is far too violent for a five-slot book with estimated (not
    known) probabilities; callers stake kelly_lambda * f*. Fractional Kelly
    also degrades gracefully when p is overestimated, which it will be — at
    lambda 0.35 a p overstated by 0.05 still leaves the stake below full
    Kelly for the true p.
    """
    if risk_per_share <= 0 or reward_per_share <= 0:
        return 0.0
    b = reward_per_share / risk_per_share
    return float((p_win * b - (1.0 - p_win)) / b)


def candidate_economics(details, size, profile=None):
    """
    Forward expectation for a NEW position, in rupees and in return on capital
    per slot-day. Uses signal_generator v4's own emitted estimates rather than
    re-deriving them, so the allocator and the entry gate cannot disagree about
    what a trade is worth.
    """
    entry = float(details['entry_price'])
    stop = float(details['stop_loss'])
    target = float(details['target_price'])
    p = float(details.get('p_win_est', 0.0))
    gap = float(details.get('gap_down_p90', 0.0))
    horizon = max(int(details.get('time_exit_bars', 15) or 15), 1)

    risk_ps = entry - stop
    reward_ps = target - entry
    if risk_ps <= 0 or reward_ps <= 0 or size < 1:
        return None

    notional = entry * size
    cost = float(round_trip_commission(entry, target, size))
    # Expected loss carries the overnight gap: an EOD stop does not fill at its
    # level when the scrip opens through it, and pretending otherwise
    # systematically overstates every candidate's expected value.
    loss_ps = risk_ps + entry * gap * 0.60
    ev_rupees = (p * reward_ps - (1.0 - p) * loss_ps) * size - cost

    return {
        'notional': round(notional, 2),
        'cost': round(cost, 2),
        'ev_rupees': round(ev_rupees, 2),
        'horizon': horizon,
        'roc_per_day': ev_rupees / notional / horizon,
        'risk_rupees': round(risk_ps * size, 2),
        'rr': reward_ps / risk_ps,
    }


def incumbent_economics(trade, evaluation, current_price, profile=None,
                        edge_retention=0.60):
    """
    Forward expectation for a position ALREADY HELD — deliberately not its
    realised performance.

    Realised R/day is the wrong comparison and it is the one the current
    replacement rule effectively makes. A position that has already covered
    most of the distance to its target has an excellent realised R/day and very
    little left to give; a position that just opened has a poor one and its
    whole move ahead. What matters for the slot is what remains:

        remaining reward = target - price       (what is still on offer)
        remaining risk   = price - stop         (what is still exposed, POST-trail)
        days left        = horizon - held

    The trail makes this asymmetric in the holder's favour over time, which is
    correct and intended: a position whose stop has ratcheted to +1.2R is
    risking very little capital for its remaining upside, and should be
    genuinely hard to evict. Thesis health scales the probability, so a
    position with a broken thesis loses that protection the moment it breaks.
    """
    P = get_profile(profile)
    entry = float(trade['entry_price'])
    stop = float(trade['stop_loss'])
    target = float(trade['target_price'])
    size = int(trade.get('position_size', 1) or 1)
    horizon = int(_num(trade.get('time_exit_bars'), 15))
    held = int(_num(trade.get('hold_days'), 0))

    price = float(current_price)
    notional = price * size
    if notional <= 0:
        return None

    remaining_reward = max(target - price, 0.0)
    # Floor the remaining risk: once the stop has trailed to within a hair of
    # price, reward/risk explodes toward infinity and the position becomes
    # un-evictable on a rounding artefact rather than on merit.
    remaining_risk = max(price - stop, 0.002 * price)
    days_left = max(horizon - held, 1)

    health = evaluation.get('health') or {}
    n_broken = int(health.get('n_signals', 0))

    # Driftless barrier probability, then the SAME kind of edge tilt
    # signal_generator applies to a candidate — because pricing held positions
    # as edgeless while pricing new ones as edged would make every candidate
    # look better than every incumbent by construction, and churn the book
    # daily at ~48 bps a lap. The position was entered on a quality score; it
    # keeps that claimed edge for as long as its thesis holds, and each broken
    # health signal takes a fifth of it away.
    p_base = remaining_risk / (remaining_risk + remaining_reward) if remaining_reward > 0 else 0.0
    quality = _num(trade.get('quality_score'), 0.62)   # neutral-positive default for legacy rows
    entry_tilt = 1.0 + edge_retention * (quality - 0.5) * 2.0
    p = float(np.clip(p_base * entry_tilt * (1.0 - 0.18 * n_broken), 0.02, 0.90))

    exit_cost = price * size * PCT_COST_PER_LEG + FLAT_CHARGE_PER_SELL
    ev_rupees = (p * remaining_reward - (1.0 - p) * remaining_risk) * size - exit_cost

    return {
        'notional': round(notional, 2),
        'exit_cost': round(exit_cost, 2),
        'ev_rupees': round(ev_rupees, 2),
        'days_left': days_left,
        'roc_per_day': ev_rupees / notional / days_left,
        'risk_rupees': round(remaining_risk * size, 2),
        'n_broken': n_broken,
        'stagnant': bool(evaluation.get('stagnant')),
        'r_multiple': float(evaluation.get('r_multiple', 0.0)),
        'progress_to_target': ((price - entry) / (target - entry)) if target > entry else 0.0,
        'p_forward': round(p, 3),
    }


# ═════════════════════════════════════════════════════════════════════════════
# ALLOCATOR
# ═════════════════════════════════════════════════════════════════════════════
class PortfolioAllocator:

    def __init__(self, profile=None, sector_map=None):
        self.P = get_profile(profile)
        self.profile_name = profile or ACTIVE_PROFILE
        self.sector_map = sector_map or {}
        logger.info(f"✓ PortfolioAllocator — {self.profile_name} "
                    f"(kelly λ={self.P['kelly_lambda']}, heat cap {self.P['max_portfolio_heat']*100:.0f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    def heat_budget(self, equity, peak_equity, exposure=None):
        """
        Total open risk the book is allowed to carry, throttled continuously by
        drawdown rather than switched off at a cliff.

        The existing circuit breaker halts trading at 35% drawdown. That is a
        useful floor and a poor controller: it applies no pressure at 10%, 20%
        or 30%, then everything at once. A linear throttle de-risks as the
        drawdown deepens, which both reduces the chance of ever reaching the
        breaker and leaves capacity to participate in the recovery — the
        breaker's own worst property is that it is fully out of the market at
        the point where mean reversion is most likely.
        """
        peak = max(float(peak_equity or equity), 1e-9)
        dd = max(0.0, (peak - float(equity)) / peak)

        # ── One drawdown authority, not three ────────────────────────────────
        # market_state already de-risks on drawdown, this method de-risks on
        # drawdown, and run_paper_trading has a circuit breaker that halts on
        # drawdown. Three controllers reading the same input and acting
        # independently do not add safety — they multiply. Two throttles at
        # 0.6 each leave 36% of the intended exposure, which is not a decision
        # anyone made, and it is unattributable after the fact.
        #
        # So when the caller supplies a market-state exposure, that IS the
        # drawdown response and this method stops applying its own. When no
        # exposure is supplied — a standalone allocator, a unit test — the
        # internal throttle remains, so the class is still safe on its own.
        if exposure is not None:
            throttle = float(np.clip(float(exposure), 0.0, 1.5))
        else:
            throttle = float(np.clip(1.0 - (dd / max(self.P['drawdown_soft_max'], 1e-9)),
                                     self.P['heat_floor'], 1.0))
        return float(equity) * self.P['max_portfolio_heat'] * throttle, dd, throttle

    # ─────────────────────────────────────────────────────────────────────────
    def size_candidate(self, details, equity, cash_available=None, alpha_score=None):
        """
        Intrinsic stake from edge, clipped by concentration and the economic
        floor. Returns (size, note) with size 0 when no viable stake exists.
        cash_available is accepted for signature compatibility and is applied
        later, in _finalize — see the note inside.

        Order matters. Kelly first, because a negative fraction should decline
        the trade outright rather than be clipped into a small position — a
        small stake on a negative-expectancy setup is still a negative-
        expectancy setup, just slower.
        """
        entry = float(details['entry_price'])
        risk_ps = entry - float(details['stop_loss'])
        reward_ps = float(details['target_price']) - entry
        p = float(details.get('p_win_est', 0.0))
        if risk_ps <= 0 or reward_ps <= 0:
            return 0, 'degenerate geometry'

        f_star = kelly_fraction(p, reward_ps, risk_ps)
        if f_star <= 0:
            return 0, f'Kelly fraction {f_star:+.3f} — no edge at p={p:.2f}, {reward_ps/risk_ps:.2f}:1'

        f_used = min(self.P['kelly_lambda'] * f_star, self.P['max_risk_pct'])
        size_by_edge = (equity * f_used) / risk_ps
        size_by_conc = (equity * self.P['max_capital_pct']) / entry

        # Deliberately NOT clipped by today's cash here. This is the INTRINSIC
        # stake — what the opportunity deserves — and ranking has to be
        # independent of the balance, or a strong candidate arriving on a
        # fully-invested day is scored as weak and never competes for a slot
        # at all. Under T+1 settlement that failure is total: cash is short on
        # exactly the days an eviction would free some, so every swap would be
        # ruled out before its merits were examined. Cash is applied in
        # _finalize, after the switching decision has been made.
        size = int(min(size_by_edge, size_by_conc))
        binding = 'edge/Kelly' if size_by_edge <= size_by_conc else 'concentration cap'

        min_notional = float(details.get('min_notional_inr', 0) or 0)
        if size * entry < min_notional:
            return 0, (f'{binding} caps at ₹{size*entry:,.0f}, under the '
                       f'₹{min_notional:,.0f} economic floor')
        return size, f'{binding} (f*={f_star:.3f}, staked {f_used*100:.2f}%)'

    # ─────────────────────────────────────────────────────────────────────────
    def rank_candidates(self, candidates, equity, cash_available):
        """
        candidates: [{'symbol', 'details', 'alpha_score'(opt), 'returns'(opt)}]
        Returns them sized, priced and ordered by tilted roc_per_day, best first.
        """
        ranked = []
        for c in candidates:
            details = c['details']
            size, note = self.size_candidate(details, equity, cash_available, c.get('alpha_score'))
            if size < 1:
                ranked.append({**c, 'size': 0, 'econ': None, 'score': -1e9, 'note': note})
                continue
            econ = candidate_economics(details, size, self.profile_name)
            if econ is None:
                ranked.append({**c, 'size': 0, 'econ': None, 'score': -1e9, 'note': 'no economics'})
                continue
            tilt = 1.0
            alpha = c.get('alpha_score')
            if alpha is not None:
                # alpha scores run 0-100; centre at 55, the project's own neutral placeholder
                tilt *= 1.0 + ALPHA_TILT * float(np.clip((float(alpha) - 55.0) / 45.0, -1.0, 1.0))
            mom = c.get('momentum_percentile')
            if mom is not None:
                tilt *= 1.0 + MOMENTUM_TILT * float(np.clip((float(mom) - 70.0) / 30.0, -1.0, 1.0))
            sent = c.get('sentiment_percentile')
            if sent is not None:
                tilt *= 1.0 + SENTIMENT_TILT * float(np.clip((float(sent) - 50.0) / 50.0, -1.0, 1.0))
            ranked.append({**c, 'size': size, 'econ': econ,
                           'score': econ['roc_per_day'] * tilt, 'note': note})
        return sorted(ranked, key=lambda r: -r['score'])

    # ─────────────────────────────────────────────────────────────────────────
    def plan(self, candidates, incumbents, equity, cash_available,
             peak_equity=None, max_slots=5, returns_frame=None, exposure=None):
        """
        The daily allocation decision.

        candidates: [{'symbol','details','alpha_score'(opt)}] — v4 BUY signals
        incumbents: [{'symbol','trade','evaluation','price'}] — open positions
                    with their exit_manager.ExitEngine.evaluate() output
        returns_frame: optional DataFrame of daily returns, columns = symbols,
                    used for the correlation guard. Absent, sector caps carry
                    the diversification load alone.

        Returns a plan dict. Nothing is executed here — the caller applies it,
        so the decision is inspectable and loggable before any order exists.
        """
        P = self.P
        heat_cap, dd, throttle = self.heat_budget(equity, peak_equity, exposure)
        open_heat = sum(float(i['trade'].get('position_size', 0) or 0) *
                        max(float(i['price']) - float(i['trade']['stop_loss']), 0.0)
                        for i in incumbents)
        heat_room = max(heat_cap - open_heat, 0.0)

        sector_counts = {}
        for i in incumbents:
            s = self.sector_map.get(i['symbol'], i['symbol'])
            sector_counts[s] = sector_counts.get(s, 0) + 1

        inc_econ = {}
        for i in incumbents:
            e = incumbent_economics(i['trade'], i['evaluation'], i['price'], self.profile_name)
            if e:
                inc_econ[i['symbol']] = e

        ranked = self.rank_candidates(candidates, equity, cash_available)
        held_symbols = {i['symbol'] for i in incumbents}

        plan = {'entries': [], 'evictions': [], 'declined': [],
                'diagnostics': {'equity': round(float(equity), 2),
                                'drawdown_pct': round(dd * 100, 2),
                                'heat_throttle': round(throttle, 3),
                                'heat_cap': round(heat_cap, 2),
                                'open_heat': round(open_heat, 2),
                                'heat_room': round(heat_room, 2),
                                'slots_used': len(incumbents), 'max_slots': max_slots,
                                'sector_counts': sector_counts,
                                'candidates_ranked': len(ranked)}}

        free_slots = max(max_slots - len(incumbents), 0)
        cash = float(cash_available)
        # An evicted position is gone for the rest of this pass. Without this,
        # four candidates all "displace" the same stagnant incumbent and the
        # plan commits four times the capital the account has.
        live_incumbents = list(incumbents)

        for r in ranked:
            symbol, details, econ = r['symbol'], r['details'], r['econ']
            if symbol in held_symbols:
                plan['declined'].append((symbol, 'already held'))
                continue
            if r['size'] < 1 or econ is None:
                plan['declined'].append((symbol, r['note']))
                continue
            if econ['roc_per_day'] < P['min_edge_roc_per_day']:
                plan['declined'].append((symbol, f"edge {econ['roc_per_day']*1e4:.1f} bps/day below the "
                                                 f"{P['min_edge_roc_per_day']*1e4:.1f} bps/day floor"))
                continue

            sector = self.sector_map.get(symbol, symbol)
            sector_full = sector_counts.get(sector, 0) >= P['max_per_sector']

            corr_block = self._correlation_block(symbol, held_symbols, returns_frame)
            if corr_block:
                plan['declined'].append((symbol, corr_block))
                continue

            victim = None
            proceeds = 0.0
            if free_slots > 0 and not sector_full:
                reason_prefix = f"free slot — {r['note']}"
            else:
                victim = self._find_switch(r, econ, live_incumbents, inc_econ, plan,
                                           sector_filter=sector if sector_full else None)
                if victim is None:
                    plan['declined'].append((symbol, 'no incumbent worth displacing'))
                    continue
                # Sale proceeds fund the replacement; they are not free capital
                # until the exit cost is paid out of them.
                # Indian cash equity settles T+1: the rupees from today's sale
                # are not available to fund today's purchase. Modelling them as
                # instant made every swap look fundable and would, live, produce
                # either a rejected order or unintended margin usage — and it
                # quietly flattered the paper record by letting the book run
                # more positions than settled capital could support.
                #
                # The eviction still executes on its merits: a broken incumbent
                # should be out regardless of what replaces it. The replacement
                # simply waits for the cash, and tomorrow's scan re-evaluates
                # the candidate on tomorrow's evidence — which is better than
                # committing today to an order that fills a day later anyway.
                proceeds = (0.0 if P.get('settlement_days', 1) > 0
                            else victim['econ']['notional'] - victim['econ']['exit_cost'])
                reason_prefix = f"displaces {victim['symbol']} — {victim['rationale']}"

            size, econ, risk_rupees, ok, why = self._finalize(
                details, r['size'], econ, cash + proceeds, heat_room + (
                    victim['econ']['risk_rupees'] if victim else 0.0))
            if not ok:
                if victim is not None and P.get('settlement_days', 1) > 0:
                    # The swap was justified but only settled cash can fund it.
                    # Free the slot today, buy tomorrow.
                    plan['evictions'].append(victim)
                    live_incumbents = [i for i in live_incumbents
                                       if i['symbol'] != victim['symbol']]
                    held_symbols.discard(victim['symbol'])
                    v_sector = self.sector_map.get(victim['symbol'], victim['symbol'])
                    sector_counts[v_sector] = max(sector_counts.get(v_sector, 1) - 1, 0)
                    heat_room += victim['econ']['risk_rupees']
                    plan['pending_settlement'] = round(
                        plan.get('pending_settlement', 0.0)
                        + victim['econ']['notional'] - victim['econ']['exit_cost'], 2)
                    plan['declined'].append(
                        (symbol, f"{victim['symbol']} evicted; entry waits on T+1 settlement "
                                 f"(₹{victim['econ']['notional']:,.0f} unsettled)"))
                else:
                    plan['declined'].append((symbol, why))
                continue

            if victim is not None:
                plan['evictions'].append(victim)
                live_incumbents = [i for i in live_incumbents if i['symbol'] != victim['symbol']]
                held_symbols.discard(victim['symbol'])
                v_sector = self.sector_map.get(victim['symbol'], victim['symbol'])
                sector_counts[v_sector] = max(sector_counts.get(v_sector, 1) - 1, 0)
                heat_room += victim['econ']['risk_rupees']
                cash += proceeds
            else:
                free_slots -= 1

            plan['entries'].append({'symbol': symbol, 'size': size, 'details': details,
                                    'econ': econ, 'reason': reason_prefix,
                                    'roc_bps_day': round(econ['roc_per_day'] * 1e4, 2)})
            held_symbols.add(symbol)
            cash -= size * float(details['entry_price'])
            heat_room -= risk_rupees
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        plan['diagnostics']['cash_remaining'] = round(cash, 2)
        plan['diagnostics']['pending_settlement'] = plan.get('pending_settlement', 0.0)
        plan['diagnostics']['heat_room_remaining'] = round(heat_room, 2)
        return plan

    # ─────────────────────────────────────────────────────────────────────────
    def _finalize(self, details, size, econ, cash, heat_room):
        """
        Clip a ranked stake to the capital and risk actually left at this point
        in the pass, then re-price it. Both constraints shrink as higher-ranked
        candidates are funded ahead of this one, so a size computed against the
        opening cash and heat is an opening bid, not an allocation.

        A stake that survives the clips but lands under the economic floor is
        declined rather than taken: a position too small to pay its own
        depository charge is not a smaller version of the trade, it is a
        different and worse one.
        """
        entry = float(details['entry_price'])
        risk_ps = entry - float(details['stop_loss'])
        min_notional = float(details.get('min_notional_inr', 0) or 0)

        size = int(min(size, max(cash, 0.0) / entry, max(heat_room, 0.0) / max(risk_ps, 1e-9)))
        if size < 1:
            return 0, econ, 0.0, False, f'no room left (cash ₹{cash:,.0f}, heat ₹{heat_room:,.0f})'
        if size * entry < min_notional:
            return 0, econ, 0.0, False, (f'₹{size*entry:,.0f} available for this name is under the '
                                         f'₹{min_notional:,.0f} economic floor')
        econ = candidate_economics(details, size, self.profile_name) or econ
        return size, econ, econ['risk_rupees'], True, ''

    # ─────────────────────────────────────────────────────────────────────────
    def _fit_to_heat(self, details, size, risk_rupees, heat_room):
        """
        Shrink a position to fit the remaining heat budget, and decline it if
        shrinking would push it under the economic floor. Scaling down into a
        row that pays 200 bps to express a smaller opinion is not risk
        management — it is the same opinion with the edge removed.
        """
        if risk_rupees <= heat_room:
            return size, risk_rupees, ''
        entry = float(details['entry_price'])
        risk_ps = entry - float(details['stop_loss'])
        scaled = int(max(heat_room, 0.0) / max(risk_ps, 1e-9))
        min_notional = float(details.get('min_notional_inr', 0) or 0)
        if scaled < 1 or scaled * entry < min_notional:
            return 0, 0.0, (f'heat budget leaves ₹{heat_room:,.0f} of risk room — '
                            f'too little to open above the economic floor')
        return scaled, scaled * risk_ps, 'heat-capped'

    # ─────────────────────────────────────────────────────────────────────────
    def _find_switch(self, cand, cand_econ, incumbents, inc_econ, plan, sector_filter=None):
        """
        The switching hurdle. A swap is taken only when the candidate's
        forward edge beats the incumbent's by enough to repay both sides of
        the transaction inside the candidate's own horizon:

            (roc_cand - roc_inc) * notional * horizon
                >  (exit_cost_inc + entry_cost_cand) * switch_margin

        Expressed in rupees rather than ratios on purpose. A 40%-better score
        on a ₹9,000 position is roughly ₹30 of expected improvement against
        ₹45 of switching cost — a ratio test calls that a clear upgrade and
        the rupees call it a loss. The rupees are right.

        Protections stand while the thesis stands: a position more than 80% of
        the way to target, or comfortably profitable, is shielded — unless
        exit_manager has flagged it stagnant or two or more of its health
        signals have fired, at which point its recent performance stops being
        a reason to keep it.
        """
        P = self.P
        best = None
        for i in incumbents:
            sym = i['symbol']
            econ_i = inc_econ.get(sym)
            if econ_i is None:
                continue
            if sector_filter is not None and self.sector_map.get(sym, sym) != sector_filter:
                continue

            thesis_intact = (econ_i['n_broken'] < 2) and not econ_i['stagnant']
            if thesis_intact and (econ_i['progress_to_target'] > 0.80 or econ_i['r_multiple'] > 0.75):
                continue                                  # earning its slot, leave it alone

            gain_rupees = ((cand_econ['roc_per_day'] - econ_i['roc_per_day'])
                           * cand_econ['notional'] * cand_econ['horizon'])
            switch_cost = econ_i['exit_cost'] + cand_econ['cost']
            if gain_rupees <= switch_cost * P['switch_margin']:
                continue

            surplus = gain_rupees - switch_cost * P['switch_margin']
            if best is None or surplus > best['surplus']:
                flags = []
                if econ_i['stagnant']:
                    flags.append('stagnant')
                if econ_i['n_broken']:
                    flags.append(f"{econ_i['n_broken']} thesis signals broken")
                best = {'symbol': sym, 'trade': i['trade'], 'price': i['price'],
                        'econ': econ_i, 'surplus': round(surplus, 2),
                        'rationale': (f"+₹{gain_rupees:,.0f} expected vs ₹{switch_cost:,.0f} switching cost"
                                      + (f" ({', '.join(flags)})" if flags else ''))}
        return best

    # ─────────────────────────────────────────────────────────────────────────
    def _correlation_block(self, symbol, held_symbols, returns_frame):
        """
        Sector labels are a coarse proxy for what actually matters — whether
        two positions lose money on the same day. Two midcap financiers in
        different sector buckets can run at 0.9 correlation, and a book of five
        of those is one position wearing five sets of costs.
        """
        if returns_frame is None or symbol not in returns_frame.columns:
            return None
        overlap = [s for s in held_symbols if s in returns_frame.columns]
        if not overlap:
            return None
        target = returns_frame[symbol].tail(60)
        for s in overlap:
            other = returns_frame[s].tail(60)
            pair = pd.concat([target, other], axis=1).dropna()
            if len(pair) < 30:
                continue
            rho = float(pair.corr().iloc[0, 1])
            if np.isfinite(rho) and rho > self.P['max_correlation']:
                return f'{rho:.2f} correlated with open position {s}'
        return None


# ═════════════════════════════════════════════════════════════════════════════
def print_plan(plan):
    d = plan['diagnostics']
    print("\n" + "=" * 74)
    print("  PORTFOLIO ALLOCATION PLAN")
    print("=" * 74)
    print(f"  Equity ₹{d['equity']:,.0f} | drawdown {d['drawdown_pct']:.1f}% | "
          f"heat throttle {d['heat_throttle']:.2f}")
    print(f"  Heat  ₹{d['open_heat']:,.0f} open of ₹{d['heat_cap']:,.0f} cap "
          f"(₹{d['heat_room']:,.0f} room) | slots {d['slots_used']}/{d['max_slots']}")
    if d.get('pending_settlement'):
        print(f"  ₹{d['pending_settlement']:,.0f} from today's sales settles T+1 — "
              f"available to deploy tomorrow")
    if plan['evictions']:
        print("\n  ── Evictions ──")
        for e in plan['evictions']:
            print(f"    ✂ {e['symbol']:<14} {e['rationale']}")
    if plan['entries']:
        print("\n  ── Entries ──")
        for e in plan['entries']:
            print(f"    ▲ {e['symbol']:<14} {e['size']:>4} sh  ₹{e['econ']['notional']:>9,.0f}  "
                  f"edge {e['roc_bps_day']:>6.2f} bps/day  | {e['reason']}")
    if plan['declined']:
        print("\n  ── Declined ──")
        for sym, why in plan['declined'][:12]:
            print(f"    · {sym:<14} {why}")
    print("=" * 74 + "\n")


def _num(value, fallback):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return fallback
        if isinstance(value, str) and not value.strip():
            return fallback
        if pd.isna(value):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback
