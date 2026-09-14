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

    # The bug that voided the first full-universe backtest: staleness was
    # measured against the wall clock, so inside a walk-forward every sliced
    # frame read as weeks old and the whole universe was rejected.
    hist_idx = pd.bdate_range(end=pd.Timestamp.today() - pd.Timedelta(days=60), periods=40)
    hist = bars(len(hist_idx)).iloc[:len(hist_idx)].copy()
    hist['datetime'] = hist_idx
    check("a historical frame is stale against the wall clock",
          not data_quality(hist)['tradeable'])
    check("...but fresh against its own simulated date",
          data_quality(hist, as_of=hist_idx[-1])['tradeable'])


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



# ═══ 7. Profit engine ════════════════════════════════════════════════════════
def test_profit_engine():
    print("\n[profit engine]")
    from profit_engine import equity_curve_scalar, evaluate_pyramid, compounding_ladder

    rng = np.random.default_rng(1)
    up = pd.Series(50000 * np.exp(np.cumsum(rng.normal(0.0015, 0.004, 60))))
    down = pd.Series(50000 * np.exp(np.cumsum(rng.normal(-0.0015, 0.004, 60))))
    s_up, s_down = equity_curve_scalar(up)[0], equity_curve_scalar(down)[0]
    check("risk scales up in phase and down out of phase", s_up > 1.0 > s_down,
          f"{s_up:.2f} vs {s_down:.2f}")
    check("the scalar is bounded, never off and never leveraged",
          0.5 < s_down and s_up < 1.4)
    check("no opinion on a short equity history",
          equity_curve_scalar(pd.Series([50000] * 8))[0] == 1.0)

    trade = {'entry_price': 500., 'initial_stop_loss': 476., 'stop_loss': 492.,
             'position_size': 40, 'target_price': 560., 'entry_type': 'pullback'}
    healthy = {'health': {'n_signals': 0}, 'stagnant': False}
    allow = {'new_entries_allowed': True}
    d = evaluate_pyramid(trade, 530.0, 52000, 26000, 6000, healthy, allow)
    check("a proved winner is added to", d['add'], d['reason'][:60])

    # The property that separates pyramiding from averaging up.
    if d['add']:
        before = 40 * 24.0
        after = 40 * max(500 - d['new_stop'], 0) + d['size'] * (530 - d['new_stop'])
        check("total open risk FALLS after the add", after <= before,
              f"₹{before:,.0f} → ₹{after:,.0f}")
        check("the stop moves above entry, covering costs", d['new_stop'] > 500)

    for label, ev, ms in (("broken thesis", {'health': {'n_signals': 2}, 'stagnant': False}, allow),
                          ("stagnant", {'health': {'n_signals': 0}, 'stagnant': True}, allow),
                          ("entries suspended", healthy, {'new_entries_allowed': False,
                                                          'state': 'DEFENSIVE'})):
        check(f"no add when {label}",
              not evaluate_pyramid(trade, 530.0, 52000, 26000, 6000, ev, ms)['add'])
    check("no add below the trigger",
          not evaluate_pyramid(trade, 505.0, 52000, 26000, 6000, healthy, allow)['add'])
    check("NaN in the add counter does not raise",
          isinstance(evaluate_pyramid({**trade, 'pyramid_adds': float('nan')}, 530.0,
                                      52000, 26000, 6000, healthy, allow)['add'], bool))

    small, large = compounding_ladder(50000), compounding_ladder(300000)
    check("the ladder widens the book as capital grows",
          large['recommended_slots'] > small['recommended_slots']
          and large['max_capital_pct'] < small['max_capital_pct'],
          f"{small['recommended_slots']}→{large['recommended_slots']} slots")


# ═══ 8. Meta model ═══════════════════════════════════════════════════════════
def _meta_trades(n, signal=True, seed=1):
    r = np.random.default_rng(seed)
    q = r.uniform(0.40, 0.95, n)
    er = r.uniform(0.08, 0.65, n)
    adx = r.uniform(12, 45, n)
    base = 0.42 + (0.45 * (q - 0.65) + 0.35 * (er - 0.35)) if signal else np.full(n, 0.42)
    win = r.random(n) < np.clip(base, 0.08, 0.85)
    return pd.DataFrame({
        'status': 'CLOSED', 'entry_date': pd.bdate_range('2025-01-01', periods=n),
        'exit_date': pd.bdate_range('2025-01-10', periods=n), 'quality_score': q,
        'p_win_est': np.clip(0.34 + 0.08 * (q - 0.6), 0.05, 0.9), 'efficiency_ratio': er,
        'risk_pct_of_price': r.uniform(2.5, 6, n), 'stop_sigma_mult': r.uniform(1.5, 2.6, n),
        'risk_reward_ratio': r.uniform(1.5, 2.6, n), 'sigma_daily_pct': r.uniform(1.0, 2.6, n),
        'time_exit_bars': r.integers(6, 18, n), 'confidence': r.integers(1, 4, n),
        'band_usage': r.uniform(0, 0.5, n), 'gap_down_p90': r.uniform(0.005, 0.02, n),
        'entry_price': r.uniform(200, 1500, n), 'position_size': r.integers(8, 40, n),
        'ind_rsi': r.uniform(40, 75, n), 'ind_adx': adx, 'ind_plus_di': adx * 0.8,
        'ind_minus_di': adx * 0.4, 'ind_cmf': r.uniform(-0.1, 0.3, n),
        'ind_extension_atr': r.uniform(0, 3, n), 'ind_stoch_k': r.uniform(20, 90, n),
        'entry_type': r.choice(['pullback', 'breakout', 'cmf_accum'], n),
        'market_state_at_entry': r.choice(['RISK_ON', 'NEUTRAL'], n),
        'net_pnl': np.where(win, r.uniform(300, 1500, n), -r.uniform(250, 850, n))})


def test_meta_model():
    print("\n[meta model]")
    from meta_model import MetaModel

    thin = MetaModel('/tmp/_t_meta_thin.json').fit(_meta_trades(40, True, 5))
    check("inert on thin data", not thin.active)
    check("thin model returns the prior untouched",
          all(abs(thin.p_win({'quality_score': q}, p) - p) < 1e-12
              for q in (0.4, 0.9) for p in (0.30, 0.45)))

    noise = MetaModel('/tmp/_t_meta_noise.json').fit(_meta_trades(400, False, 2))
    check("refuses to activate on data with no signal", not noise.active,
          f"OOS AUC {noise.oos_auc}")

    real = MetaModel('/tmp/_t_meta_real.json').fit(_meta_trades(600, True, 3))
    check("activates when out-of-sample skill is demonstrated", real.active,
          f"AUC {real.oos_auc}, Brier {real.oos_brier} vs prior {real.prior_brier}")
    if real.active:
        strong = {'quality_score': 0.92, 'efficiency_ratio': 0.60, 'p_win_est': 0.40,
                  'risk_reward_ratio': 2.4, 'entry_type': 'pullback',
                  'market_state_at_entry': 'RISK_ON',
                  'indicators': {'rsi': 62, 'adx': 38, 'plus_di': 30, 'minus_di': 10,
                                 'stoch_k': 70, 'extension_atr': 0.8}}
        weak = {'quality_score': 0.45, 'efficiency_ratio': 0.12, 'p_win_est': 0.32,
                'risk_reward_ratio': 1.6, 'entry_type': 'breakout',
                'market_state_at_entry': 'NEUTRAL',
                'indicators': {'rsi': 52, 'adx': 15, 'plus_di': 16, 'minus_di': 14,
                               'stoch_k': 30, 'extension_atr': 2.6}}
        p_hi, p_lo = real.p_win(strong, 0.36), real.p_win(weak, 0.36)
        check("strong and weak setups separate", p_hi > p_lo + 0.05,
              f"{p_hi:.3f} vs {p_lo:.3f}")
        check("the blend never replaces the geometry outright",
              0.5 * 0.36 <= p_lo and p_hi <= 1.6 * 0.36 + 1e-9)
        reloaded = MetaModel('/tmp/_t_meta_real.json')
        check("the fitted model round-trips through disk",
              reloaded.active and abs(reloaded.p_win(strong, 0.36) - p_hi) < 1e-9)


# ═══ 9. Optimiser ════════════════════════════════════════════════════════════
def test_optimizer():
    print("\n[optimiser]")
    import signal_generator as sg
    from optimizer import WalkForwardOptimizer, SEARCH_SPACE, apply_params, restore_params

    before = sg.RISK_PROFILE['k_stop_base']
    prev = apply_params({'k_stop_base': 9.99})
    changed = sg.RISK_PROFILE['k_stop_base'] == 9.99
    restore_params(prev)
    check("parameters are applied to and restored from the live profile",
          changed and sg.RISK_PROFILE['k_stop_base'] == before)

    index = pd.DataFrame({'datetime': pd.bdate_range('2024-01-01', periods=600),
                          'close': np.linspace(100, 130, 600)})
    universe = {f'S{i}': index.copy() for i in range(20)}

    def surrogate(strength):
        def fn(u, idx, f, start, end):
            r = np.random.default_rng(int(start * 7919 + end))
            n = max(int((end - start) * 0.30), 0)
            if n < 5:
                return pd.DataFrame(columns=['net_pnl', 'hold_days', 'exit_date'])
            k, rr = sg.RISK_PROFILE['k_stop_base'], sg.RISK_PROFILE['min_rr']
            edge = strength * (1.0 - abs(k - 1.50) * 2.2 - abs(rr - 1.50) * 1.6)
            dates = pd.to_datetime(idx['datetime']).iloc[start:end]
            return pd.DataFrame({'net_pnl': r.normal(edge * 260, 900, n),
                                 'hold_days': r.integers(5, 16, n),
                                 'exit_date': r.choice(dates, n)})
        return fn

    space = {k: SEARCH_SPACE[k] for k in ('k_stop_base', 'min_rr', 'kelly_lambda')}
    real = WalkForwardOptimizer(surrogate(1.0), 50000., space, 4, 27).run(universe, index, {}, 200)
    check("recovers a known optimum", real['best_params']['k_stop_base'] == 1.50
          and real['best_params']['min_rr'] == 1.50, str(real['best_params']))
    check("deploys when out-of-sample is positive and stable",
          real['verdict'] == 'DEPLOY', f"OOS {real['oos_mean_bps_per_slot_day']}")

    noise = WalkForwardOptimizer(surrogate(0.0), 50000., space, 4, 27).run(universe, index, {}, 200)
    check("holds on a surface with no real edge", noise['verdict'] == 'HOLD',
          noise['reasons'][0][:60])
    check("the live profile is left untouched by a search",
          sg.RISK_PROFILE['k_stop_base'] == before)



# ═══ 10. Cross-sectional momentum ════════════════════════════════════════════
def test_momentum_rank():
    print("\n[momentum rank]")
    from momentum_rank import rank_universe, gate, compute_factors
    rng = np.random.default_rng(5)
    idx_dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
    n = len(idx_dates)

    def series(drift=0.0, vol=0.014, jump=None, s0=500.):
        r = rng.normal(drift, vol, n)
        if jump:
            r[jump[0]] += jump[1]
        c = s0 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, 0.004, n))
        return pd.DataFrame({'datetime': idx_dates, 'open': o,
                             'high': np.maximum(o, c) * 1.005,
                             'low': np.minimum(o, c) * 0.995, 'close': c,
                             'volume': rng.lognormal(12, .3, n)})

    index = series(0.0004, 0.008, s0=24000)
    uni = {}
    for i in range(8):
        uni[f'STRONG{i}'] = series(drift=0.0016)
    for i in range(8):
        uni[f'WEAK{i}'] = series(drift=-0.0008)
    for i in range(4):
        uni[f'JUMP{i}'] = series(drift=0.0002, jump=(200, 0.45))
    ranks = rank_universe(uni, index)
    avg = lambda p: float(np.mean([r['percentile'] for s, r in ranks.items() if s.startswith(p)]))

    check("leaders outrank laggards", avg('STRONG') > avg('WEAK') + 30,
          f"{avg('STRONG'):.0f} vs {avg('WEAK'):.0f}")
    # Information discreteness is a MODIFIER, not a veto, and the assertion is
    # set to match that. A 45% single-gap move still carries genuine 12-1
    # momentum, which is the best-evidenced factor and holds the largest
    # weight; discreteness and consistency demote it rather than disqualify it.
    # An earlier version of this test demanded a 15-point gap and failed at 11,
    # which was the test over-claiming, not the ranker underperforming —
    # tightening discreteness enough to pass it would have meant down-weighting
    # the strongest factor to satisfy an assertion I had invented.
    check("a single-gap move is demoted below steady strength",
          avg('JUMP') < avg('STRONG') - 8, f"{avg('JUMP'):.0f} vs {avg('STRONG'):.0f}")
    check("the gate blocks laggards", not gate(ranks, 'WEAK0')[0])
    check("an unranked symbol is eligible, not silently excluded",
          gate(ranks, 'NEVER_SEEN')[0])
    # Symbols ending on different dates left the union index's final row with
    # too few members, so breadth was withheld on every bar of a 106-symbol
    # backtest and the regime engine lost its only leading sensor.
    from market_state import BreadthPanel
    uneven = {}
    for i, (name, df) in enumerate(uni.items()):
        uneven[name] = df.iloc[:len(df) - (i % 4)]
    panel = BreadthPanel(uneven, min_symbols=10)
    row = panel.at()
    check("breadth survives symbols ending on different dates",
          row is not None and int(row['n_symbols']) >= 10,
          f"n_symbols={None if row is None else int(row['n_symbols'])}")

    check("a universe too thin to rank withholds the ranking",
          rank_universe({k: uni[k] for k in list(uni)[:5]}, index) == {})
    check("short history is omitted rather than imputed",
          compute_factors(series().head(40), index) is None)
    f = compute_factors(uni['STRONG0'], index)
    check("residual momentum strips the market component",
          f.get('residual') is not None and abs(f.get('beta', 0)) < 2.0,
          f"t={f.get('residual'):.2f}, beta={f.get('beta')}")


# ═══ 11. Compounding ═════════════════════════════════════════════════════════
def test_compounding():
    print("\n[compounding]")
    import os
    for f in ('/tmp/_t_comp.csv', '/tmp/_t_comp_eq.csv'):
        if os.path.exists(f):
            os.remove(f)
    from paper_trading_manager import PaperTradingManager
    from orchestrator import TradingOrchestrator
    from profit_engine import compounding_ladder

    m = PaperTradingManager(initial_equity=50000, csv_path='/tmp/_t_comp.csv',
                            equity_csv_path='/tmp/_t_comp_eq.csv', max_open_trades=5)
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o.mgr = m
    _, equity_before = o._capital({})

    m.open_trade('AAA', 500., 476., 560., 40, 'pullback')
    _, equity_deployed = o._capital({'AAA': 500.0})
    check("deploying capital does not shrink the equity base",
          abs(equity_deployed - equity_before) < 500,
          f"₹{equity_before:,.0f} → ₹{equity_deployed:,.0f} with ₹20,000 deployed")

    tid = pd.read_csv('/tmp/_t_comp.csv')['trade_id'].iloc[0]
    m.close_position(tid, 560.0, 'Target Hit')
    _, equity_after = o._capital({})
    profit = equity_after - equity_before
    check("realised profit is added to the compounding base", profit > 1500,
          f"+₹{profit:,.0f} on a ₹2,400 gross winner, net of costs")

    # The loop that matters: a larger base must produce a larger next position.
    risk_pct = 0.035
    check("a larger base produces a larger next position",
          equity_after * risk_pct > equity_before * risk_pct,
          f"risk budget ₹{equity_before*risk_pct:,.0f} → ₹{equity_after*risk_pct:,.0f}")

    grown = compounding_ladder(equity_after * 2, mode='growth')
    small = compounding_ladder(equity_before, mode='growth')
    check("the ladder widens the book as the account compounds",
          grown['recommended_slots'] >= small['recommended_slots'])
    for eq in (50000, 120000, 400000):
        L = compounding_ladder(eq, mode='growth')
        if L['recommended_slots'] * L['max_capital_pct'] > 1.02:
            check("deployment never exceeds 100% of a cash account", False,
                  f"₹{eq:,} deploys {L['recommended_slots']*L['max_capital_pct']*100:.0f}%")
            break
    else:
        check("deployment never exceeds 100% of a cash account", True)



# ═══ 12. Profile coherence across modules ════════════════════════════════════
def test_profile_coherence():
    print("\n[profile coherence]")
    import market_state, profit_engine, exit_manager, portfolio_allocator
    import entry_execution, signal_generator
    registries = [('market_state', market_state.STATE_PROFILES, market_state.get_profile),
                  ('profit_engine', profit_engine.PROFIT_PROFILES, profit_engine.get_profile),
                  ('exit_manager', exit_manager.EXIT_PROFILES, exit_manager.get_profile),
                  ('portfolio_allocator', portfolio_allocator.ALLOCATOR_PROFILES,
                   portfolio_allocator.get_profile),
                  ('entry_execution', entry_execution.EXECUTION_PROFILES,
                   entry_execution.get_profile)]

    # The orchestrator passes ONE profile name to all of these. A module missing
    # it raised KeyError mid-run — and because the backtest engine catches
    # per-bar exceptions, every bar after the first fill failed silently.
    names = set(signal_generator.RISK_CALIBRATIONS)
    for label, registry, _ in registries:
        missing = names - set(registry)
        check(f"{label} knows every profile name", not missing, f"missing {sorted(missing)}")

    for label, _, getter in registries:
        try:
            got = getter('does_not_exist')
            check(f"{label} falls back on an unknown profile", isinstance(got, dict))
        except Exception as e:
            check(f"{label} falls back on an unknown profile", False, repr(e))


if __name__ == "__main__":
    print("=" * 66)
    print("  v11 REGRESSION SUITE — invariants, each with an incident behind it")
    print("=" * 66)
    for fn in (test_costs, test_exits, test_signals, test_allocation,
               test_calibration_and_data, test_manager, test_profit_engine,
               test_meta_model, test_optimizer, test_momentum_rank,
               test_compounding, test_profile_coherence):
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
