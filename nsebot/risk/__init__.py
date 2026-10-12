"""Risk layer: how much, when to get out, and when to stop.

sizing      fractional Kelly on R-multiples with a cap stack (cash, margin,
            concentration, liquidity, portfolio heat)
exits       stop / trail / time / MIS square-off, one pure function per bar
breakers    consecutive-loss, daily-loss, max-drawdown, trades/day, kill file
allocator   ranked signals -> sized orders under slot, sector and heat limits
"""

from .allocator import AllocationPlan, Order, allocate, open_risk
from .breakers import BreakerVerdict, CircuitBreakers
from .exits import ExitDecision, Position, advance, evaluate
from .sizing import EdgeEstimate, SizeDecision, estimate_edge, risk_fraction, size_position

__all__ = ['AllocationPlan', 'Order', 'allocate', 'open_risk', 'BreakerVerdict',
           'CircuitBreakers', 'ExitDecision', 'Position', 'advance', 'evaluate',
           'EdgeEstimate', 'SizeDecision', 'estimate_edge', 'risk_fraction', 'size_position']
