"""Portfolio backtest of the deployed swing engine — the real code path.

Not a separate simulator: it drives SwingEngine.run() one session at a time
over history, with the same PaperBroker, Ledger, sizing, allocator, exits,
circuit breakers and regime dial the daily runner uses. The only difference
from live is that the DailyMarket view is built once over the full history
instead of once per day — and the view only ever answers point-in-time
questions (bars on or before the session being processed).

    python -m nsebot backtest --out research_results
"""

import logging
import os
import tempfile

import numpy as np
import pandas as pd

from .broker.paper import PaperBroker
from .config import BotConfig
from .engine.market_view import DailyMarket
from .engine.swing import SwingEngine
from .ledger import Ledger
from .signals import ReversionSignalEngine

logger = logging.getLogger(__name__)
IS_FRACTION = 0.6


def _curve(eq):
    eq = eq.dropna()
    if len(eq) < 20:
        return {}
    r = eq.pct_change().dropna()
    yrs = len(r) / 252
    vol = r.std() * np.sqrt(252)
    return {'cagr': (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan,
            'sharpe': r.mean() * 252 / vol if vol > 0 else np.nan,
            'maxdd': float((eq / eq.cummax() - 1).min()), 'ret': eq.iloc[-1] / eq.iloc[0] - 1}


def _trades(t):
    if t is None or len(t) == 0:
        return {'n': 0}
    net = t['net_pnl'].astype(float)
    return {'n': len(t), 'win': float((net > 0).mean()), 'net_sum': float(net.sum()),
            'avg_net': float(net.mean()), 'avg_r': float(t['net_r'].astype(float).mean()),
            'costs': float(t['costs'].astype(float).sum())}


def run_backtest(universe, index_df, cfg=None, initial_cash=50_000.0, warmup=200, workdir=None):
    cfg = cfg or BotConfig()
    market = DailyMarket(universe, index_df, ReversionSignalEngine(cfg.reversion))
    root = workdir or tempfile.mkdtemp(prefix='nsebot_bt_')
    ledger = Ledger(root, 'swing', initial_cash)
    engine = SwingEngine(cfg, PaperBroker(), ledger, workdir=root, log=lambda m: None)
    dates = market.dates[warmup:]
    for d in dates:
        engine.run(market, d)
    eq = ledger.equity_history()
    eq = pd.Series(eq['equity'].astype(float).to_numpy(), index=pd.to_datetime(eq['session']))
    trades = ledger.closed_trades()
    return market, eq, trades, ledger


def report(universe, index_df, cfg=None, initial_cash=50_000.0):
    market, eq, trades, ledger = run_backtest(universe, index_df, cfg, initial_cash)
    cut = eq.index[int(len(eq) * IS_FRACTION)] if len(eq) > 10 else eq.index[0]
    nifty = (index_df.set_index(pd.to_datetime(index_df['datetime']))['close']
             .reindex(eq.index).ffill())
    tr = trades.copy()
    if len(tr):
        tr['exit_time'] = pd.to_datetime(tr['exit_time'])
    halves = {'Full': (eq.index[0], eq.index[-1]), 'In-sample': (eq.index[0], cut),
              'Out-of-sample': (cut, eq.index[-1])}
    md = ['## Portfolio backtest — the deployed swing engine, end to end', '',
          f"₹{initial_cash:,.0f} paper book · {eq.index[0].date()} → {eq.index[-1].date()} "
          f"({len(eq)} sessions) · same code path as the daily runner (signals, half-Kelly sizing, "
          f"allocator, exits, breakers, regime dial, Zerodha costs, next-open fills).", '',
          '| period | return | CAGR | Sharpe | max DD | Nifty CAGR | trades | win | net ₹ | avg net R | costs ₹ |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, (a, b) in halves.items():
        c = _curve(eq.loc[a:b])
        n = _curve(nifty.loc[a:b])
        t = _trades(tr[(tr['exit_time'] >= a) & (tr['exit_time'] <= b)] if len(tr) else tr)
        if not c:
            continue
        md.append(f"| {name} | {c['ret'] * 100:+.1f}% | {c['cagr'] * 100:+.1f}% | {c['sharpe']:.2f} | "
                  f"{c['maxdd'] * 100:.1f}% | {n.get('cagr', float('nan')) * 100:+.1f}% | {t['n']} | "
                  + (f"{t['win'] * 100:.0f}% | {t['net_sum']:+,.0f} | {t['avg_r']:+.3f} | {t['costs']:,.0f} |"
                     if t['n'] else '— | — | — | — |'))
    if len(tr):
        md += ['', '| exit reason | trades | net ₹ | avg net R |', '|---|---|---|---|']
        for reason, g in tr.groupby(tr['exit_reason'].str.split(':').str[0]):
            md.append(f"| {reason} | {len(g)} | {g['net_pnl'].sum():+,.0f} | {g['net_r'].mean():+.3f} |")
        exposure = (eq.index.size and len(tr) / eq.index.size)
        md += ['', f"Average trades per session: {exposure:.2f} · final equity ₹{eq.iloc[-1]:,.0f} · "
                   f"worst trade ₹{tr['net_pnl'].min():+,.0f} · best ₹{tr['net_pnl'].max():+,.0f}"]
    return '\n'.join(md), eq, trades


def main(out_dir='research_results'):
    from .data import YahooProvider
    from .universe import INDEX_SYMBOL, SWING_UNIVERSE
    os.makedirs(out_dir, exist_ok=True)
    prov = YahooProvider()
    universe = prov.daily(SWING_UNIVERSE, lookback_days=520)
    index_df = prov.daily([INDEX_SYMBOL], lookback_days=520)[INDEX_SYMBOL]
    md, eq, trades = report(universe, index_df)
    eq.to_csv(os.path.join(out_dir, 'backtest_equity.csv'))
    trades.to_csv(os.path.join(out_dir, 'backtest_trades.csv'), index=False)
    with open(os.path.join(out_dir, 'BACKTEST.md'), 'w') as fh:
        fh.write(md + '\n')
    return md
