# backtest_analytics.py
#
# Performance analytics for swing_trading_bot.backtest_portfolio() output.
# Kept as its own module, independently tested against hand-computed values
# on synthetic data — Sharpe/Sortino/drawdown calculations are exactly the
# kind of thing that's easy to get subtly wrong (wrong annualisation factor,
# population vs sample std, which downside-deviation convention), and a
# subtle bug here would mean drawing real conclusions from wrong numbers.
#
# Two entry points:
#   compute_performance_report(trades_df, equity_df, initial_equity) -> dict
#   print_performance_report(report)                                -> None
#   compare_reports(report_a, report_b, label_a, label_b)            -> None
#       for the alpha-engine-ON vs OFF A/B comparison — the real question
#       this whole framework exists to answer isn't "is the absolute number
#       good," it's "does the alpha layer actually add anything over the
#       baseline."
#
# All ratios use a 0% risk-free rate / minimum acceptable return, and 252
# trading days/year for annualisation — standard, common conventions, but
# stated explicitly here since both are assumptions, not universal constants.

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def _max_drawdown(equity_series):
    """Returns (max_drawdown_pct, trough_index) — max_drawdown_pct is positive (e.g. 0.23 = 23% decline)."""
    running_max = equity_series.cummax()
    drawdown    = (running_max - equity_series) / running_max.replace(0, np.nan)
    if drawdown.isna().all():
        return 0.0, None
    trough_idx = drawdown.idxmax()
    return float(drawdown.max()), trough_idx


def _sharpe(daily_returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    if len(daily_returns) < 2:
        return 0.0
    std = daily_returns.std(ddof=1)
    # Epsilon, not exact-zero: a "constant" return series is never exactly
    # zero-variance in floating point (0.001 isn't exactly representable in
    # binary), so std ends up as some tiny-but-nonzero value like 1e-19 —
    # dividing by that blows Sharpe up to an astronomical, meaningless
    # number instead of correctly reporting "no meaningful volatility here."
    if std < 1e-9:
        return 0.0
    return float(daily_returns.mean() / std * np.sqrt(periods_per_year))


def _sortino(daily_returns, target=0.0, periods_per_year=TRADING_DAYS_PER_YEAR):
    """
    Downside deviation convention: EVERY period contributes to the average,
    with returns at/above target counted as zero shortfall — not just the
    subset of below-target periods. This is the standard Sortino (1994)
    definition; a "std-dev of only the negative returns" alternative exists
    elsewhere but understates risk by excluding all the zero-shortfall
    periods from the denominator's sample size.
    """
    if len(daily_returns) < 2:
        return 0.0
    shortfall = np.minimum(daily_returns - target, 0.0)
    downside_dev = np.sqrt((shortfall ** 2).mean())
    if downside_dev < 1e-9:
        return 0.0
    return float((daily_returns.mean() - target) / downside_dev * np.sqrt(periods_per_year))


def _cagr(initial_equity, final_equity, n_periods, periods_per_year=TRADING_DAYS_PER_YEAR):
    if initial_equity <= 0 or final_equity <= 0 or n_periods <= 0:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((final_equity / initial_equity) ** (1 / years) - 1)


def _trade_stats(pnl_series):
    n = len(pnl_series)
    if n == 0:
        return {'n': 0, 'win_rate': None, 'avg_win': None, 'avg_loss': None,
                'expectancy': None, 'profit_factor': None}
    wins   = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]
    win_rate      = len(wins) / n
    avg_win       = float(wins.mean())    if len(wins)   > 0 else 0.0
    avg_loss      = float(-losses.mean()) if len(losses) > 0 else 0.0
    expectancy    = win_rate * avg_win - (1 - win_rate) * avg_loss
    gross_profit  = float(wins.sum())
    gross_loss    = float(-losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else None)
    return {
        'n': n, 'win_rate': round(win_rate, 3), 'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2), 'expectancy': round(expectancy, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor not in (None, float('inf')) else profit_factor,
        'total_pnl': round(float(pnl_series.sum()), 2),
    }


def compute_performance_report(trades_df, equity_df, initial_equity):
    """
    trades_df: DataFrame from backtest_portfolio() — one row per CLOSED
               trade, needs at minimum 'net_pnl', 'hold_days'; optionally
               'alpha_tier', 'market_regime', 'entry_type' for stratified
               breakdowns (present automatically if use_alpha_engine=True
               was used, or if those columns exist regardless).
    equity_df: DataFrame with an 'equity' column, one row per simulated bar
               (backtest_portfolio()'s equity_curve).
    initial_equity: starting capital.

    Returns a dict — see print_performance_report for the human-readable form.
    """
    report = {'initial_equity': initial_equity, 'n_trades': 0}

    if trades_df is None or len(trades_df) == 0:
        report['note'] = 'no trades to analyse'
        return report

    pnl = trades_df['net_pnl'].astype(float)
    overall = _trade_stats(pnl)
    report.update(overall)
    report['n_trades'] = overall['n']
    report['avg_hold_days'] = round(float(trades_df['hold_days'].mean()), 1) if 'hold_days' in trades_df else None

    if equity_df is not None and len(equity_df) > 0 and 'equity' in equity_df.columns:
        eq = equity_df['equity'].astype(float)
        final_equity = float(eq.iloc[-1])
        daily_returns = eq.pct_change().dropna()

        max_dd, trough_idx = _max_drawdown(eq)
        cagr = _cagr(initial_equity, final_equity, len(eq))
        sharpe  = _sharpe(daily_returns)
        sortino = _sortino(daily_returns)
        calmar  = (cagr / max_dd) if max_dd > 0 else (float('inf') if cagr > 0 else 0.0)

        report.update({
            'final_equity':   round(final_equity, 2),
            'total_return_pct': round((final_equity / initial_equity - 1) * 100, 2),
            'cagr_pct':       round(cagr * 100, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'sharpe':         round(sharpe, 2),
            'sortino':        round(sortino, 2),
            'calmar':         round(calmar, 2) if calmar != float('inf') else calmar,
            'n_bars':         len(eq),
        })
    else:
        report['note'] = 'no equity curve supplied — return/risk-adjusted metrics unavailable, trade stats only'

    # ── Stratified breakdowns — the actual evidence for whether the alpha
    #    engine's ranking has real predictive value: does a higher tier
    #    actually show better expectancy, or is it noise? ──────────────────
    for strat_col, label in [('alpha_tier', 'by_tier'), ('market_regime', 'by_regime'), ('entry_type', 'by_pattern')]:
        if strat_col in trades_df.columns and trades_df[strat_col].notna().any():
            breakdown = {}
            for key, group in trades_df.dropna(subset=[strat_col]).groupby(strat_col):
                breakdown[key] = _trade_stats(group['net_pnl'].astype(float))
            report[label] = breakdown

    return report


def print_performance_report(report, title="BACKTEST PERFORMANCE REPORT"):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)

    if report.get('n_trades', 0) == 0:
        print(f"  {report.get('note', 'No trades.')}")
        print("=" * 72 + "\n")
        return

    print(f"\n  ── Returns ─────────────────────────────────────────────────────")
    if 'final_equity' in report:
        print(f"  Initial Equity        : ₹{report['initial_equity']:>12,.2f}")
        print(f"  Final Equity          : ₹{report['final_equity']:>12,.2f}")
        print(f"  Total Return          : {report['total_return_pct']:>+12.2f}%")
        print(f"  CAGR                  : {report['cagr_pct']:>+12.2f}%")
    else:
        print(f"  {report.get('note', '')}")

    print(f"\n  ── Risk-adjusted ───────────────────────────────────────────────")
    if 'sharpe' in report:
        print(f"  Sharpe Ratio          : {report['sharpe']:>12.2f}   (0% risk-free rate, annualised)")
        print(f"  Sortino Ratio         : {report['sortino']:>12.2f}   (downside deviation vs 0% target)")
        print(f"  Max Drawdown          : {report['max_drawdown_pct']:>12.2f}%")
        calmar_str = f"{report['calmar']:.2f}" if report['calmar'] != float('inf') else "inf (no drawdown)"
        print(f"  Calmar Ratio          : {calmar_str:>12}   (CAGR / max drawdown)")

    print(f"\n  ── Trade stats ─────────────────────────────────────────────────")
    print(f"  Total Trades          : {report['n_trades']:>12}")
    print(f"  Win Rate              : {report['win_rate']*100:>11.1f}%")
    pf_str = f"{report['profit_factor']:.2f}" if report['profit_factor'] not in (None, float('inf')) else str(report['profit_factor'])
    print(f"  Profit Factor         : {pf_str:>12}   (gross profit / gross loss)")
    print(f"  Expectancy per trade  : ₹{report['expectancy']:>+11,.2f}")
    print(f"  Avg Win / Avg Loss    : ₹{report['avg_win']:>+9,.2f} / ₹{-report['avg_loss']:>+9,.2f}")
    if report.get('avg_hold_days') is not None:
        print(f"  Avg Hold Days         : {report['avg_hold_days']:>12.1f}")

    for strat_col, label in [('by_tier', 'Conviction Tier'), ('by_regime', 'Market Regime'), ('by_pattern', 'Entry Pattern')]:
        if strat_col in report:
            print(f"\n  ── Stratified by {label} ─{'─'*(46-len(label))}")
            for key, stats in report[strat_col].items():
                pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] not in (None, float('inf')) else str(stats['profit_factor'])
                print(f"    {str(key):<28} n={stats['n']:<5} win_rate={stats['win_rate']*100:>5.1f}%  "
                      f"expectancy=₹{stats['expectancy']:>+8,.2f}  profit_factor={pf}")

    print("=" * 72 + "\n")


def compare_reports(report_a, report_b, label_a="Alpha engine ON", label_b="Alpha engine OFF (baseline)"):
    """
    Side-by-side A/B comparison — this is the actual question that matters:
    does the alpha layer add real value over the baseline signal_generator-
    only policy, not just "is the combined system's number positive."
    """
    print("\n" + "=" * 72)
    print(f"  A/B COMPARISON: {label_a}  vs  {label_b}")
    print("=" * 72)

    if report_a.get('n_trades', 0) == 0 or report_b.get('n_trades', 0) == 0:
        print("  One or both runs produced no trades — cannot compare.")
        print("=" * 72 + "\n")
        return

    rows = [
        ('Total Trades',    'n_trades',          '{:.0f}',  False),
        ('Win Rate',        'win_rate',          '{:.1%}',  False),
        ('Profit Factor',   'profit_factor',     '{:.2f}',  False),
        ('Expectancy/trade','expectancy',        '₹{:+,.2f}', False),
        ('Total Return',    'total_return_pct',  '{:+.2f}%', False),
        ('CAGR',            'cagr_pct',          '{:+.2f}%', False),
        ('Sharpe',          'sharpe',            '{:.2f}',  False),
        ('Sortino',         'sortino',           '{:.2f}',  False),
        ('Max Drawdown',    'max_drawdown_pct',  '{:.2f}%', True),   # lower is better
        ('Calmar',          'calmar',            '{:.2f}',  False),
    ]
    print(f"\n  {'Metric':<20}{'A: ' + label_a:<26}{'B: ' + label_b:<26}{'Edge'}")
    print(f"  {'-'*20}{'-'*26}{'-'*26}{'-'*10}")
    for name, key, fmt, lower_is_better in rows:
        va, vb = report_a.get(key), report_b.get(key)
        if va is None or vb is None:
            continue
        try:
            va_s, vb_s = fmt.format(va), fmt.format(vb)
        except (ValueError, TypeError):
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            better_a = (va < vb) if lower_is_better else (va > vb)
            edge = f"A (+{abs(va-vb):.2f})" if better_a and va != vb else (f"B (+{abs(va-vb):.2f})" if va != vb else "tie")
        else:
            edge = ""
        print(f"  {name:<20}{va_s:<26}{vb_s:<26}{edge}")
    print("=" * 72 + "\n")
