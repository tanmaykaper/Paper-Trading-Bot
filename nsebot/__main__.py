"""nsebot command line.

    python -m nsebot swing      [--broker paper|kite] [--state state]   end-of-day CNC cycle
    python -m nsebot intraday   [--broker paper|kite] [--state state]   MIS session loop to 15:10
    python -m nsebot backtest   [--out research_results]                full-engine portfolio backtest
    python -m nsebot status     [--state state]                         sleeves at a glance

Paper is the default broker. Kite needs KITE_API_KEY plus a daily token from
`python -m nsebot.kite_login`, and live order placement needs a static IP
(SEBI retail-algo rules) — see nsebot/broker/kite.py.

Exit codes: 0 ok · 2 Kite token missing/expired · 3 market data unusable.
"""

import argparse
import logging
import os
import sys

from .broker.paper import PaperBroker
from .config import BotConfig
from .ledger import Ledger
from .notify import intraday_report, job_summary, send_email, swing_report

logger = logging.getLogger('nsebot')

MIN_RESOLVED = 0.70


def _broker(kind):
    if kind == 'kite':
        from .broker.kite import KiteBroker, KiteSession
        return KiteBroker(KiteSession())
    return PaperBroker()


def _capital(mode):
    return float(os.environ.get(f'{mode.upper()}_CAPITAL', 50_000))


def cmd_swing(args):
    from .data import YahooProvider
    from .engine.market_view import DailyMarket, completed_session
    from .engine.swing import SwingEngine
    from .universe import INDEX_SYMBOL, SWING_UNIVERSE

    cfg = BotConfig()
    ledger = Ledger(args.state, 'swing', _capital('swing'))
    broker = _broker(args.broker)
    held = [p.symbol for p in ledger.positions] + [p['spec']['symbol'] for p in ledger.state['pending']]
    symbols = list(dict.fromkeys(SWING_UNIVERSE + held))
    prov = YahooProvider()
    universe = prov.daily(symbols, lookback_days=400)
    index = prov.daily([INDEX_SYMBOL], lookback_days=400).get(INDEX_SYMBOL)
    resolved = len(universe) / max(len(symbols), 1)
    if index is None or resolved < MIN_RESOLVED:
        msg = (f'market data unusable: index {"missing" if index is None else "ok"}, '
               f'{len(universe)}/{len(symbols)} symbols — state left untouched')
        logger.error(msg)
        send_email('nsebot swing — DATA FAILURE', msg)
        return 3
    missing_held = sorted(set(held) - set(universe))
    if missing_held:
        logger.warning(f'no data for held/pending {missing_held} — their bars are processed next run')
    market = DailyMarket(universe, index)
    asof = completed_session(index)
    if asof is None:
        logger.error('no completed session in the index data')
        return 3
    rep = SwingEngine(cfg, broker, ledger, workdir='.', log=logger.info).run(market, asof)
    md = swing_report(rep, ledger.state['initial_cash'])
    print(md)
    job_summary(md)
    if rep.get('filled') or rep.get('closed') or rep.get('placed') or rep.get('breakers'):
        send_email(f"nsebot swing {rep['asof']} — equity ₹{rep.get('equity', 0):,.0f}", md)
    return 0


def cmd_intraday(args):
    from .data import YahooProvider
    from .engine.intraday import IntradayEngine

    cfg = BotConfig()
    ledger = Ledger(args.state, 'intraday', _capital('intraday'))
    engine = IntradayEngine(cfg, _broker(args.broker), ledger, workdir='.', log=logger.info)
    reports = engine.run_session(YahooProvider(chunk_size=40, pause_s=0.5))
    md = intraday_report(reports, ledger)
    print(md)
    job_summary(md)
    send_email(f"nsebot intraday {ledger.state.get('session')}", md)
    return 0


def cmd_backtest(args):
    from .backtest import main as bt
    md = bt(args.out)
    print(md)
    job_summary(md)
    return 0


def cmd_status(args):
    for mode in ('swing', 'intraday'):
        L = Ledger(args.state, mode, _capital(mode))
        t = L.closed_trades()
        net = float(t['net_pnl'].sum()) if len(t) else 0.0
        print(f"{mode:9s} cash ₹{L.cash:,.0f} · open {len(L.positions)} · pending "
              f"{len(L.state.get('pending', []))} · closed {len(t)} (net ₹{net:+,.0f}) · "
              f"last session {L.state.get('last_processed') or L.state.get('session')}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='nsebot')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('swing', 'intraday'):
        p = sub.add_parser(name)
        p.add_argument('--broker', choices=['paper', 'kite'], default=os.environ.get('NSEBOT_BROKER', 'paper'))
        p.add_argument('--state', default='state')
    p = sub.add_parser('backtest')
    p.add_argument('--out', default='research_results')
    p = sub.add_parser('status')
    p.add_argument('--state', default='state')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stdout)
    try:
        return {'swing': cmd_swing, 'intraday': cmd_intraday, 'backtest': cmd_backtest,
                'status': cmd_status}[args.cmd](args)
    except Exception as e:
        if type(e).__name__ == 'TokenExpired':
            logger.error(f'Kite token problem: {e}')
            send_email('nsebot — KITE TOKEN EXPIRED', f'{e}\n\nNew entries halted. Exchange-side stops '
                                                        f'(GTT / SL-M) remain in force.')
            return 2
        raise


if __name__ == '__main__':
    sys.exit(main())
