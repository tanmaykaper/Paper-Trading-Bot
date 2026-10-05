"""Turn ranked signals into sized orders under portfolio-wide limits.

Greedy in signal rank order, and deliberately simple — V2's allocator scored
forward return per slot-day, rotated incumbents and lost money on every one
of 26 rotations. Here a candidate gets a slot if, at its turn:

  * a slot is free (max_positions) and today's new-entry budget remains
    (the regime dial's max_new_entries)
  * its sector is not already at max_per_sector
  * it is not already held
  * the sizer returns >= 1 share after every cap (buying power, concentration,
    liquidity, remaining portfolio heat)

Open positions are never evicted to make room. The exit engine decides when a
position ends; the allocator only fills empty space.
"""

from dataclasses import dataclass, field

from .sizing import SizeDecision, size_position


@dataclass
class Order:
    symbol: str
    side: str
    qty: int
    ref_price: float
    stop: float
    sizing: SizeDecision
    signal: object = None
    product: str = 'CNC'

    @property
    def notional(self):
        return self.qty * self.ref_price


@dataclass
class AllocationPlan:
    orders: list = field(default_factory=list)
    declined: list = field(default_factory=list)       # (symbol, reason)
    heat_cap: float = 0.0
    heat_used: float = 0.0
    slots_free: int = 0


def open_risk(pos):
    """Rupees still at risk on an open position. A stop trailed past entry
    has locked in profit, so it contributes nothing to heat."""
    per_share = pos.sign * (pos.entry_price - pos.stop)
    return max(per_share, 0.0) * pos.qty


def allocate(signals, open_positions, cfg, *, equity, cash, edge, regime_mult=1.0,
             breaker_mult=1.0, max_new=None, sector_of=lambda s: 'OTHER',
             turnover_of=lambda s: None, lot_size_of=lambda s: 1, product='CNC'):
    plan = AllocationPlan()
    held = {p.symbol for p in open_positions}
    sectors = {}
    for p in open_positions:
        sectors[sector_of(p.symbol)] = sectors.get(sector_of(p.symbol), 0) + 1

    plan.heat_cap = cfg.max_portfolio_heat_pct * equity
    plan.heat_used = sum(open_risk(p) for p in open_positions)
    slots = max(cfg.max_positions - len(open_positions), 0)
    budget = slots if max_new is None else min(slots, int(max_new))
    plan.slots_free = slots
    cash_left = float(cash)

    for sig in signals:
        if sig.symbol in held:
            plan.declined.append((sig.symbol, 'already held'))
            continue
        if budget <= 0:
            plan.declined.append((sig.symbol, 'no slot / new-entry budget left today'))
            continue
        sector = sector_of(sig.symbol)
        if sectors.get(sector, 0) >= cfg.max_per_sector:
            plan.declined.append((sig.symbol, f'sector {sector} at its cap of {cfg.max_per_sector}'))
            continue
        size = size_position(cfg, equity=equity, cash=cash_left, entry=sig.ref_price,
                             stop=sig.stop, edge=edge, regime_mult=regime_mult,
                             breaker_mult=breaker_mult, median_turnover=turnover_of(sig.symbol),
                             heat_room=plan.heat_cap - plan.heat_used,
                             lot_size=lot_size_of(sig.symbol))
        if not size.ok:
            plan.declined.append((sig.symbol, f'sized to zero (binding: {size.binding})'))
            continue
        plan.orders.append(Order(sig.symbol, sig.side, size.qty, sig.ref_price, sig.stop,
                                 size, sig, product))
        held.add(sig.symbol)
        sectors[sector] = sectors.get(sector, 0) + 1
        plan.heat_used += size.risk_rupees
        cash_left -= size.notional / max(cfg.leverage, 1e-9)
        budget -= 1
    return plan
