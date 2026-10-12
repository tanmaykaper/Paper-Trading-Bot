"""Thread-safe rate limiting for broker APIs.

Kite Connect enforces per-second limits by endpoint family and per-minute /
per-day caps on orders. Exceeding them returns HTTP 429 and, repeatedly,
can get an app throttled — so the client must pace itself rather than rely
on retries. Two primitives:

  TokenBucket    smooth per-second pacing with a small burst allowance
  WindowCounter  hard caps over a sliding window (orders per minute / day)

Published Kite limits at the time of writing (verify against the current
Kite Connect documentation; they are constructor arguments, not constants
buried in code):
  quote 1/s · historical 3/s · orders 10/s · everything else 10/s
  orders: 200/minute, 3,000/day per user
"""

import threading
import time
from collections import deque

KITE_RATES = {'quote': 1.0, 'historical': 3.0, 'order': 10.0, 'default': 10.0}
KITE_ORDER_PER_MINUTE = 200
KITE_ORDER_PER_DAY = 3000


class RateLimitExceeded(RuntimeError):
    """A hard cap (per-day orders) is exhausted — waiting will not help today."""


class TokenBucket:

    def __init__(self, rate_per_s, burst=None, clock=time.monotonic, sleep=time.sleep):
        self.rate = float(rate_per_s)
        self.capacity = float(burst if burst is not None else max(rate_per_s, 1.0))
        self.tokens = self.capacity
        self.clock, self.sleep = clock, sleep
        self.last = clock()
        self.lock = threading.Lock()

    def acquire(self, n=1.0):
        while True:
            with self.lock:
                now = self.clock()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= n - 1e-9:          # tolerance: 0.999999... is a whole token
                    self.tokens = max(self.tokens - n, 0.0)
                    return
                wait = max((n - self.tokens) / self.rate, 1e-4)
            self.sleep(wait)


class WindowCounter:

    def __init__(self, limit, window_s, clock=time.monotonic, sleep=time.sleep, block=True):
        self.limit, self.window = int(limit), float(window_s)
        self.clock, self.sleep, self.block = clock, sleep, block
        self.events = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = self.clock()
                while self.events and now - self.events[0] >= self.window:
                    self.events.popleft()
                if len(self.events) < self.limit:
                    self.events.append(now)
                    return
                if not self.block:
                    raise RateLimitExceeded(f'{self.limit} calls per {self.window:.0f}s exhausted')
                wait = self.window - (now - self.events[0])
            self.sleep(max(wait, 0.01))


class KiteRateLimiter:
    """Per-category pacing plus order caps, shared by every Kite call."""

    def __init__(self, rates=None, orders_per_minute=KITE_ORDER_PER_MINUTE,
                 orders_per_day=KITE_ORDER_PER_DAY, clock=time.monotonic, sleep=time.sleep):
        rates = dict(KITE_RATES, **(rates or {}))
        self.buckets = {k: TokenBucket(v, clock=clock, sleep=sleep) for k, v in rates.items()}
        self.per_minute = WindowCounter(orders_per_minute, 60, clock, sleep, block=True)
        self.per_day = WindowCounter(orders_per_day, 86_400, clock, sleep, block=False)

    def acquire(self, category):
        if category == 'order':
            self.per_day.acquire()
            self.per_minute.acquire()
        self.buckets.get(category, self.buckets['default']).acquire()
