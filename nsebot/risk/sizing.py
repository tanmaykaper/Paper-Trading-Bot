"""Aggressive, evidence-weighted position sizing.

    risk budget  = equity x clip(kelly_fraction x f*, floor, cap)
                          x regime size_mult x breaker size_mult
    shares       = risk budget / risk per share
    then clipped by every hard constraint, tightest wins:
        buying power   cash x leverage (CNC 1x, MIS up to 5x)
        concentration  max_position_pct of sleeve equity, in notional
        liquidity      max_adv_participation of median daily traded value
        heat           room left under the portfolio-wide open-risk cap

Kelly is computed on R-multiples: with win rate p and payoff b (average
winning R over average losing R), the growth-optimal fraction of capital to
RISK per trade is f* = p - (1-p)/b. Half-Kelly keeps roughly three quarters
of full-Kelly growth with a fraction of its drawdown — the aggressive end of
what survives an estimation error, which with tens of trades is large.

The estimate is Bayesian: a prior from the real-data research run on this
exact signal, counted as `prior_strength` trades, blended with the bot's own
closed trades. Ten trades barely move it; two hundred dominate it. If the
measured f* turns negative, the sizer drops to the floor and says so — it
never pretends an edge exists to justify a bigger bet.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeEstimate:
    win_rate: float
    payoff: float
    kelly_full: float
    n_realised: int
    prior_weight: float          # share of the estimate still coming from the prior

    @property
    def edge_positive(self):
        return self.kelly_full > 0


def estimate_edge(cfg, realised_r=()):
    """Posterior win rate and payoff from the prior plus realised R-multiples."""
    r = np.asarray([x for x in realised_r if x is not None and np.isfinite(x)], dtype=float)
    n = len(r)
    n0 = max(int(cfg.prior_strength), 0)
    wins = float((r > 0).sum())
    p = (cfg.prior_win_rate * n0 + wins) / (n0 + n) if (n0 + n) else cfg.prior_win_rate

    payoff = cfg.prior_payoff
    if n >= 5 and (r > 0).any() and (r <= 0).any():
        avg_win, avg_loss = r[r > 0].mean(), abs(r[r <= 0].mean())
        if avg_loss > 1e-9:
            w = n / (n + n0) if (n + n0) else 1.0
            payoff = (1.0 - w) * cfg.prior_payoff + w * (avg_win / avg_loss)

    kelly = p - (1.0 - p) / payoff if payoff > 0 else -1.0
    return EdgeEstimate(round(p, 4), round(payoff, 3), round(kelly, 4), n,
                        round(n0 / (n0 + n), 3) if (n0 + n) else 1.0)


def risk_fraction(cfg, edge):
    """Fraction of equity to risk on the next trade, before regime/breaker scaling."""
    if not edge.edge_positive:
        return cfg.risk_floor_pct
    return float(np.clip(cfg.kelly_fraction * edge.kelly_full, cfg.risk_floor_pct, cfg.risk_cap_pct))


@dataclass
class SizeDecision:
    qty: int
    risk_rupees: float
    notional: float
    risk_pct: float
    binding: str                 # which constraint set the final size
    notes: str = ''

    @property
    def ok(self):
        return self.qty >= 1


def size_position(cfg, *, equity, cash, entry, stop, edge, regime_mult=1.0, breaker_mult=1.0,
                  median_turnover=None, heat_room=None, lot_size=1):
    """Shares to trade. Every input in rupees; `lot_size` rounds down to the
    exchange's market lot (1 for NSE cash equity)."""
    risk_ps = abs(float(entry) - float(stop))
    if equity <= 0 or entry <= 0 or risk_ps <= 0:
        return SizeDecision(0, 0.0, 0.0, 0.0, 'invalid', 'non-positive equity, price or stop distance')

    rf = risk_fraction(cfg, edge) * float(regime_mult) * float(breaker_mult)
    budget = equity * rf
    caps = {'kelly risk': budget / risk_ps,
            'buying power': max(cash, 0.0) * cfg.leverage / entry,
            'concentration': equity * cfg.max_position_pct / entry}
    if median_turnover and median_turnover > 0:
        caps['liquidity'] = median_turnover * cfg.max_adv_participation / entry
    if heat_room is not None:
        caps['portfolio heat'] = max(heat_room, 0.0) / risk_ps

    binding = min(caps, key=caps.get)
    lot = max(int(lot_size), 1)
    qty = int(math.floor(caps[binding] / lot)) * lot
    note = '' if edge.edge_positive else f'measured edge negative (Kelly {edge.kelly_full:+.3f}) — floor size'
    return SizeDecision(qty, round(qty * risk_ps, 2), round(qty * entry, 2),
                        round(qty * risk_ps / equity, 5), binding, note)
