# test_regression.py  ── INVARIANT SUITE FOR THE v11 STACK
# ═════════════════════════════════════════════════════════════════════════════
# Every bug this project has actually suffered shared one shape: the code kept
# running and produced a plausible number that was wrong. A calendar count
# compared against a bar threshold. A stop trailing from a stale reference. A
# fundamentals parser matching nothing and falling through to healthy defaults.
# A string written into a float column, aborting the daily run on exactly the
# days a fill occurred. None of them raised. None would have been caught by
# "does it run".
#
# So these are not unit tests of implementation detail — they are assertions
# about PROPERTIES that must hold for the system to be trustworthy, chosen
# because each one has a matching incident behind it. Run before deploying any
# change:  python test_regression.py
#
# Synthetic data throughout, deterministic seeds, no network. Runs in seconds.
# ═════════════════════════════════════════════════════════════════════════════

import sys
import numpy as np
import pandas as pd

RESULTS = []


def check(name, condition, detail=''):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    return bool(condition)


def bars(n=120, s0=100.0, mu=0.0008, vol=0.012, seed=1):
    rng = np.random.default_rng(seed)
    c = s0 * np.exp(np.cumsum(rng.normal(mu, vol, n)))
    o = c * np.exp(rng.normal(0, vol * 0.3, n))
    return pd.DataFrame({
        'datetime': pd.bdate_range('2025-01-01', periods=n),
        'open': o, 'high': np.maximum(o, c) * 1.006, 'low': np.minimum(o, c) * 0.994,
        'close': c, 'volume': rng.lognormal(np.log(4e5), 0.3, n)})


# ═══ 1. Costs ════════════════════════════════════════════════════════════════
def test_costs():
    print("\n[costs]")
    from trading_costs import round_trip_commission, DPChargeLedger, cost_breakdown
    b = cost_breakdown(1000, 1100, 20)
    check("itemised components sum to the total",
          abs(b['buy']['total'] + b['sell']['total'] + b['dp_charge'] - b['total']) < 0.01)
    check("stamp duty is buy-side only", b['sell'].get('stamp_duty', 0) == 0)

    # The incident: a 3-tranche same-day exit was billed the DP charge 3x.
    led = DPChargeLedger()
    same_day = sum(led.round_trip('AAA', 400, 430, 10, '2026-09-11') for _ in range(3))
    naive = sum(round_trip_commission(400, 430, 10) for _ in range(3))
    check("DP charge billed once per scrip per day", same_day < naive - 30,
          f"₹{same_day:.0f} vs ₹{naive:.0f} naive")
    check("ledger resets on a new date",
          led.round_trip('AAA', 400, 430, 10, '2026-09-12') > 25)

    # Cost must fall as notional rises — the entire basis of the economic floor.
    bps = [round_trip_commission(500, 540, s) / (500 * s) * 1e4 for s in (2, 20, 200)]
    check("cost in bps falls monotonically with size", bps[0] > bps[1] > bps[2],
          f"{[round(x) for x in bps]} bps")


# ═══ 2. Exits ════════════════════════════════════════════════════════════════
def test_exits():
    print("\n[exits]")
    from exit_manager import compute_trailing_stop, resolve_fill, ExitEngine

    # The incident trailing_stop.py v1 documented: risk recomputed from the
    # CURRENT stop, so tiers stopped progressing once the stop had moved.
    stop, path = 95.0, []
    for price in (104, 109, 118, 113, 121):
        stop = compute_trailing_stop(100.0, 95.0, stop, price,
                                     highest_high=price, current_atr=2.5, position_size=30)
        path.append(stop)
    check("trailing stop never loosens", all(b >= a for a, b in zip(path, path[1:])), str(path))
    check("trailing stop stays below price", all(s < p for s, p in zip(path, (104, 109, 118, 113, 121))))

    # The incident: HAPPSTMNDS exited at its entry price for -₹112 net.
    be = compute_trailing_stop(100.0, 95.0, 95.0, 107.0, highest_high=107.5,
                               current_atr=1.0, position_size=2)
    check("breakeven floor covers round-trip cost", be > 100.0, f"floor ₹{be:.2f} vs entry ₹100")

    bar = lambda o, h, l: pd.Series({'open': o, 'high': h, 'low': l, 'close': (h + l) / 2})
    check("gap through the stop fills at the open",
          resolve_fill(bar(92, 96, 91), 95.0, 115.0)[0] == 92)
    check("a bar touching both barriers resolves adversely",
          resolve_fill(bar(100, 116, 94), 95.0, 115.0)[0] == 95.0)

    eng = ExitEngine('aggressive')
    tr = {'entry_price': 100., 'stop_loss': 96., 'initial_stop_loss': 96., 'target_price': 112.,
          'position_size': 30, 'time_exit_bars': 14, 'quality_score': 0.7}
    check("results proximity closes the position",
          'Pre-Earnings' in (eng.evaluate(tr, current_price=104., bars_held=5,
                                          bars_to_earnings=1)['exit_reason'] or ''))
    check("a distant results date does not",
          eng.evaluate(tr, current_price=104., bars_held=5, bars_to_earnings=9)['action'] != 'EXIT')


# ═══ 3. Signals and geometry ═════════════════════════════════════════════════
def test_signals():
    print("\n[signals]")
    from technical_indicators import TechnicalIndicators as TI
    from fundamental_screener import FundamentalScreener
    from signal_generator import SignalGenerator, apply_earnings_constraint, RISK_PROFILE

    d = bars(200, seed=7)
    adx = TI.calculate_dmi(d['high'], d['low'], d['close'], 14)
    check("ADX is bounded to [0,100]",
          bool(adx['adx'].dropna().between(0, 100).all()))
    check("Wilder RSI stays inside [0,100]",
          bool(TI.calculate_rsi(d['close'], 14).dropna().between(0, 100).all()))
    flat = d.copy(); flat[['open', 'high', 'low', 'close']] = 100.0
    check("volatility floor holds on a zero-range series",
          float(TI.daily_volatility_fraction(flat).iloc[-1]) > 0)

    sg = SignalGenerator(TI(), FundamentalScreener())
    # Degenerate inputs must produce HOLD, never an exception and never a trade.
    for label, frame in (('None', None), ('short', bars(20)), ('flat', flat)):
        sig, det = sg.generate_signal(frame, 'X', {}, 50000)
        check(f"degenerate input ({label}) yields HOLD with a reason",
              sig == 'HOLD' and 'reason' in det)

    det = {'time_exit_bars': 14, 'entry_price': 100., 'stop_loss': 95., 'target_price': 112.}
    _, ok_near, _ = apply_earnings_constraint(det, 2)
    nd, ok_far, _ = apply_earnings_constraint(det, 9)
    check("entry inside the earnings blackout is refused", not ok_near)
    check("horizon truncates to clear results", ok_far and nd['time_exit_bars'] < 14,
          f"14 → {nd['time_exit_bars']}")
    check("an unknown results date imposes no constraint",
          apply_earnings_constraint(det, None)[1])


# ═══ 4. Allocation ═══════════════════════════════════════════════════════════
def test_allocation():
    print("\n[allocation]")
    from portfolio_allocator import PortfolioAllocator, kelly_fraction

    check("Kelly is negative when there is no edge", kelly_fraction(0.30, 1.5, 1.0) < 0)
    check("Kelly rises with win probability",
          kelly_fraction(0.55, 2.4, 1.0) > kelly_fraction(0.45, 2.4, 1.0))

    a = PortfolioAllocator('aggressive', {})
    det = {'entry_price': 600., 'stop_loss': 570., 'target_price': 678., 'p_win_est': 0.50,
           'gap_down_p90': 0.011, 'time_exit_bars': 14, 'min_notional_inr': 9091.}
    plan = a.plan([{'symbol': 'A', 'details': det}], [], equity=52000,
                  cash_available=26000, peak_equity=52000, max_slots=5)
    deployed = sum(e['econ']['notional'] for e in plan['entries'])
    check("deployment never exceeds available cash", deployed <= 26000 + 1,
          f"₹{deployed:,.0f} of ₹26,000")

    # The incident: four candidates each "displaced" the same incumbent,
    # committing 4x the account's capital in one pass.
    inc = [{'symbol': s, 'price': 300., 'evaluation': {'r_multiple': -0.2, 'stagnant': True,
            'health': {'n_signals': 3}},
            'trade': {'entry_price': 310, 'stop_loss': 295, 'target_price': 340,
                      'position_size': 40, 'time_exit_bars': 15, 'hold_days': 12,
                      'quality_score': 0.62}} for s in 'BCDE']
    cands = [{'symbol': f'N{i}', 'details': det} for i in range(4)]
    p2 = a.plan(cands, inc, equity=52000, cash_available=30000, peak_equity=52000, max_slots=4)
    victims = [e['symbol'] for e in p2['evictions']]
    check("no incumbent is evicted twice in one pass", len(victims) == len(set(victims)), str(victims))

    # The incident: three independent drawdown controllers multiplying.
    no_exp = a.heat_budget(45000, 50000)[0]
    with_exp = a.heat_budget(45000, 50000, exposure=0.5)[0]
    check("market-state exposure is the sole drawdown authority",
          abs(with_exp - 52000 * 0 - 45000 * a.P['max_portfolio_heat'] * 0.5) < 1,
          f"₹{with_exp:,.0f} (standalone ₹{no_exp:,.0f})")


# ═══ 5. Calibration and data integrity ═══════════════════════════════════════
def test_calibration_and_data():
    print("\n[calibration & data]")
    from calibration import WinCalibrator
    cal = WinCalibrator('/tmp/_t_cal.json')
    check("an unfitted calibrator reproduces the prior exactly",
          all(abs(cal.p_win(q, p) - p) < 1e-12 for q in (0.4, 0.7, 0.95) for p in (0.30, 0.45)))

    rng = np.random.default_rng(5)
    n = 400
    q = rng.uniform(0.4, 0.95, n)
    flat = pd.DataFrame({'quality_score': q, 'p_win_est': np.full(n, 0.34),
                         'net_pnl': np.where(rng.random(n) < 0.42, 500.0, -400.0)})
    c2 = WinCalibrator('/tmp/_t_cal2.json').fit(flat)
    spread = max(c2.p_win(x, 0.34) for x in (0.45, 0.9)) - min(c2.p_win(x, 0.34) for x in (0.45, 0.9))
    check("no edge is invented when quality carries no signal", spread < 0.01,
          f"spread {spread:.3f}, significance z={c2.signal_z}")

    # ...and a real relationship IS picked up, so the guard is not simply off.
    p_true = np.clip(0.18 + 0.55 * (q - 0.40), 0.05, 0.85)
    real = pd.DataFrame({'quality_score': q, 'p_win_est': np.full(n, 0.34),
                         'net_pnl': np.where(rng.random(n) < p_true, 500.0, -400.0)})
    c3 = WinCalibrator('/tmp/_t_cal3.json').fit(real)
    check("a genuine quality-to-outcome relationship is detected",
          c3.signal_detected and c3.p_win(0.9, 0.34) > c3.p_win(0.5, 0.34),
          f"z={c3.signal_z}, p(0.5)={c3.p_win(0.5,0.34):.3f} → p(0.9)={c3.p_win(0.9,0.34):.3f}")

    from data_fetcher_free import _sanitize_ohlcv, data_quality
    d = bars(80, s0=1000.0, seed=3)
    for col in ('open', 'high', 'low', 'close'):
        d.loc[40:, col] = d.loc[40:, col] / 5.0
    raw = float((d.close / d.close.shift() - 1).abs().max())
    clean, notes = _sanitize_ohlcv(d, 'SPLIT')
    unchecked, _ = _sanitize_ohlcv(d.copy(), '^NSEI', check_actions=False)
    check("indices are exempt from split detection",
          abs(float(unchecked.close.iloc[0]) - float(d.close.iloc[0])) < 1e-9)
    check("a corporate action is back-adjusted, not treated as a real move",
          float((clean.close / clean.close.shift() - 1).abs().max()) < 0.10,
          f"raw {raw*100:.0f}% → {float((clean.close/clean.close.shift()-1).abs().max())*100:.1f}%")
    check("the most recent traded price is preserved",
          abs(float(clean.close.iloc[-1]) - float(d.close.iloc[-1])) < 1e-9)

    sidx = pd.bdate_range(end=pd.Timestamp.today() - pd.Timedelta(days=25), periods=40)
    stale = bars(len(sidx)).iloc[:len(sidx)].copy()
    stale['datetime'] = sidx
    check("a stale frame is refused", not data_quality(stale)['tradeable'])
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=40)
    fresh = bars(len(idx)).iloc[:len(idx)].copy()
    fresh['datetime'] = idx
    check("a fresh frame is accepted", data_quality(fresh)['tradeable'])


# ═══ 6. Manager and end-to-end ═══════════════════════════════════════════════
def test_manager():
    print("\n[manager]")
    import os
    for f in ('/tmp/_t_trades.csv', '/tmp/_t_eq.csv'):
        if os.path.exists(f):
            os.remove(f)
    from paper_trading_manager import PaperTradingManager, _STACK_COLS
    m = PaperTradingManager(initial_equity=50000, csv_path='/tmp/_t_trades.csv',
                            equity_csv_path='/tmp/_t_eq.csv', max_open_trades=5)
    schema = pd.read_csv('/tmp/_t_trades.csv').columns
    check("the v11 columns exist natively in the schema",
          all(c in schema for c in _STACK_COLS))

    # The incident: a string written into an all-NaN float column raised and
    # aborted the whole run on every day a fill occurred.
    m.open_trade('AAA', 100.0, 95.0, 112.0, 40, 'pullback',
                 extra_fields={'quality_score': 0.81, 'time_exit_bars': 7,
                               'entry_adx': 26.4})
    from orchestrator import TradingOrchestrator
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o.trades_csv = '/tmp/_t_trades.csv'
    o._patch_latest('AAA', {'market_state_at_entry': 'RISK_ON'})
    row = pd.read_csv('/tmp/_t_trades.csv').iloc[0]
    check("a string column can be written without raising",
          row['market_state_at_entry'] == 'RISK_ON')
    check("extra fields persist through open_trade",
          float(row['quality_score']) == 0.81 and float(row['time_exit_bars']) == 7)

    # The incident that killed the first live run: get_open_trades() returns
    # list-of-dicts while get_closed_trades() returns a DataFrame, and the
    # orchestrator called .iterrows() on the records.
    from orchestrator import _as_frame, _bars_held
    raw = m.get_open_trades()
    check("get_open_trades returns records, not a DataFrame",
          not isinstance(raw, pd.DataFrame), type(raw).__name__)
    frame = _as_frame(raw)
    check("the orchestrator normalises records into a DataFrame",
          isinstance(frame, pd.DataFrame) and hasattr(frame, 'iterrows') and len(frame) == 1)
    check("numeric columns survive the records round trip",
          float(frame['entry_price'].iloc[0]) == 100.0
          and float(frame['time_exit_bars'].iloc[0]) == 7)
    check("_as_frame tolerates None and empty input",
          len(_as_frame(None)) == 0 and len(_as_frame([])) == 0)
    check("bars_held is preferred over calendar hold_days",
          _bars_held({'bars_held': 6, 'hold_days': 9}) == 6
          and _bars_held({'hold_days': 9}) == 9 and _bars_held({}) == 0)

    # The incident one crash later: free_cash and current_equity are @property
    # on the live manager and methods on SimulatedBook, and current_equity is a
    # backward-compat ALIAS for free_cash — not equity.
    from orchestrator import _value
    check("accessors resolve whether property or method",
          _value(m, 'free_cash') is not None
          and _value(type('M', (), {'free_cash': lambda s: 7.0})(), 'free_cash') == 7.0)
    check("a missing accessor returns the default, not an exception",
          _value(m, 'no_such_thing', default='ok') == 'ok')

    o2 = TradingOrchestrator.__new__(TradingOrchestrator)
    o2.mgr = m
    cash, equity = o2._capital({'AAA': 104.0})
    check("equity is total portfolio value, not uninvested cash",
          equity > cash and abs(equity - 50000) < 500,
          f"cash ₹{cash:,.0f} vs equity ₹{equity:,.0f}")

    # The incident: hold time counted in calendar days against a bar threshold.
    for _ in range(7):
        m.update_trades({'AAA': 102.0}, max_hold_days=18)
    closed = pd.read_csv('/tmp/_t_trades.csv').iloc[0]
    check("the time exit counts sessions, honouring the trade's own horizon",
          closed['status'] == 'CLOSED' and int(float(closed['bars_held'])) == 7,
          f"{closed['exit_reason']} at {closed['bars_held']} bars")


if __name__ == "__main__":
    print("=" * 66)
    print("  v11 REGRESSION SUITE — invariants, each with an incident behind it")
    print("=" * 66)
    for fn in (test_costs, test_exits, test_signals, test_allocation,
               test_calibration_and_data, test_manager):
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} raised", False, repr(e))
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 66)
    print(f"  {passed}/{len(RESULTS)} passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"    FAILED: {name} {detail}")
    print("=" * 66)
    sys.exit(0 if passed == len(RESULTS) else 1)
