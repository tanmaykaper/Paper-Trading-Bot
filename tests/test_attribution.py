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
