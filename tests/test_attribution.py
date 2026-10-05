import numpy as np

from nsebot.config import BotConfig
from nsebot.engine.market_view import DailyMarket
from nsebot.research.attribution import _seg, run_variant
from conftest import daily_frame


def test_attribution_variants_run_and_parity_takes_every_signal():
    rng = np.random.default_rng(0)
    u = {}
    for i in range(12):
        r = rng.normal(0.0015, 0.015, 420)
        for k in rng.choice(np.arange(260, 410), 3, replace=False):
            r[k:k + 3] = -0.04
        u[f'S{i}'] = daily_frame(400 * np.exp(np.cumsum(r)), 300_000.0)
    idx = daily_frame(20000 * np.exp(np.cumsum(np.full(420, 0.0005))))
    market = DailyMarket(u, idx)
    parity = run_variant(market, 'P0', BotConfig(), 0.0, True, 'flat')
    deployed = run_variant(market, 'P4', BotConfig(), 5.0, False, 'real')
    assert len(parity['trades']) >= len(deployed['trades']) > 0
    s = _seg(parity, market.dates[0], market.dates[-1])
    assert s['n'] == len(parity['trades']) and abs(s['avg_notional'] - 15_000) < 1_000


def test_notional_candidate_never_trades_below_the_economic_floor():
    from nsebot.research.attribution import make_notional_allocate
    rng = np.random.default_rng(1)
    u = {}
    for i in range(12):
        r = rng.normal(0.0015, 0.015, 420)
        for k in rng.choice(np.arange(260, 410), 3, replace=False):
            r[k:k + 3] = -0.04
        u[f'S{i}'] = daily_frame(400 * np.exp(np.cumsum(r)), 300_000.0)
    market = DailyMarket(u, daily_frame(20000 * np.exp(np.cumsum(np.full(420, 0.0005)))))
    res = run_variant(market, 'C1', BotConfig(), 5.0, False, 'budget',
                      allocator=make_notional_allocate(0.20, 8_000.0))
    t = res['trades']
    assert len(t) > 0 and ((t['entry_price'] * t['qty']) >= 7_900).all()


def test_deployed_engine_reproduces_candidate_c1():
    """Phase 4b must trade exactly what C1 was measured as."""
    from nsebot.research.attribution import make_notional_allocate
    rng = np.random.default_rng(7)
    u = {}
    for i in range(14):
        r = rng.normal(0.0015, 0.015, 430)
        for k in rng.choice(np.arange(260, 420), 4, replace=False):
            r[k:k + 3] = -0.04
        u[f'S{i}'] = daily_frame(400 * np.exp(np.cumsum(r)), 300_000.0)
    market = DailyMarket(u, daily_frame(20000 * np.exp(np.cumsum(np.full(430, -0.0003)))))
    c1 = run_variant(market, 'C1', BotConfig(), 5.0, False, 'budget',
                     allocator=make_notional_allocate(0.20, 8_000.0))
    deployed = run_variant(market, 'deployed', BotConfig(), 5.0, False, 'real')
    cols = ['symbol', 'entry_time', 'qty', 'entry_price', 'exit_price']
    a = c1['trades'][cols].reset_index(drop=True)
    b = deployed['trades'][cols].reset_index(drop=True)
    assert len(a) > 5
    assert a.equals(b)
