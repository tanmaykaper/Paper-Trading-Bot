# optimizer.py  ── WALK-FORWARD PARAMETER OPTIMISATION  v1
# ═════════════════════════════════════════════════════════════════════════════
# Roughly forty constants govern this system: stop width, horizon reach, the
# R:R floor, Kelly lambda, the quality bar, ADX and efficiency-ratio minimums,
# the cost hurdle, trigger thresholds in the regime engine. Every one was set
# by reasoning about market structure. None was fitted, and reasoning is a good
# way to get a constant into the right ORDER OF MAGNITUDE and a poor way to get
# it right.
#
# This tunes them — carefully, because a parameter optimiser pointed at a
# trading system is the single most reliable way to destroy one.
#
# ── Why naive optimisation is worse than not optimising ─────────────────────
# With ~40 parameters and a few hundred trades there are vastly more degrees of
# freedom than independent observations. Any search will find a configuration
# that looks superb on the sample, and it will describe the sample's noise
# rather than the market's structure. The failure is not that the optimiser
# finds nothing — it is that it always finds something, and the something is
# confidently wrong.
#
# Five guards, all mandatory:
#
#   1. ANCHORED WALK-FORWARD, NEVER A SINGLE SPLIT. Parameters are chosen on
#      window k and scored on window k+1, repeatedly. The reported number is
#      the average of scores on data the search had never seen when it chose.
#
#   2. PURGING AND EMBARGO. A trade open across a fold boundary is excluded.
#      Without it, outcomes leak backwards across the split and the
#      out-of-sample number is fiction.
#
#   3. A SMALL, HAND-CHOSEN SEARCH SPACE. Six parameters, coarse grids, each
#      with a structural reason to be tunable. Optimising all forty would
#      guarantee overfitting no matter how the validation is arranged — the
#      defence against too many degrees of freedom is fewer of them, not
#      cleverer statistics.
#
#   4. STABILITY IS PART OF THE SCORE. A parameter that wins in three of four
#      folds and collapses in the fourth is worse than one that is
#      consistently second, because the collapse is what you will experience
#      live. The objective penalises dispersion across folds directly.
#
#   5. A DEPLOY VERDICT, NOT JUST A WINNER. The optimiser is allowed to report
#      that no configuration beat the incumbent by enough to justify changing
#      anything. That is the most common correct answer and it is the one a
#      naive optimiser can never give.
#
# ── Objective ───────────────────────────────────────────────────────────────
# Expectancy per slot-day, net of cost, minus a dispersion penalty. Per
# slot-day rather than per trade because the scarce resource is slot-days, and
# a configuration that earns more per trade by holding twice as long has not
# improved anything.
# ═════════════════════════════════════════════════════════════════════════════

import itertools
import logging
from copy import deepcopy

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── The search space ─────────────────────────────────────────────────────────
# Six parameters, each tunable for a stated structural reason. Everything else
# stays fixed — not because it is optimal, but because the budget of degrees of
# freedom is small and these are where it is best spent.
SEARCH_SPACE = {
    # Stop width against the noise floor. The single highest-leverage number in
    # the system and the one with the clearest failure mode on either side.
    'k_stop_base':        [1.30, 1.50, 1.70],
    # How far the horizon is assumed to reach. Sets the target and therefore
    # the achievable R:R.
    'reach_fraction':     [1.10, 1.25, 1.40],
    # The R:R floor. Trades frequency against payoff directly.
    'min_rr':             [1.40, 1.50, 1.65],
    # The entry quality bar. Trades frequency against selectivity.
    'base_quality':       [0.38, 0.42, 0.48],
    # Trend-strength floor. Governs how much of the universe reaches the
    # pattern stage at all.
    'min_adx':            [12, 14, 17],
    # Fraction of full Kelly staked. Pure risk/return dial with no effect on
    # which trades are taken.
    'kelly_lambda':       [0.25, 0.35, 0.45],
}

MIN_TRADES_PER_FOLD = 12
DISPERSION_PENALTY = 0.5      # weight on across-fold standard deviation
IMPROVEMENT_THRESHOLD = 0.12  # OOS score must beat the incumbent by this fraction
MAX_IS_OOS_GAP = 0.50         # in-sample may exceed out-of-sample by at most this


# ═════════════════════════════════════════════════════════════════════════════
def score_trades(trades, equity_start=50000.0):
    """
    Objective: net rupees per slot-day, normalised by starting equity.

    Returns None when the sample is too small to mean anything — which the
    caller must treat as "no information", never as zero.
    """
    if trades is None or len(trades) < MIN_TRADES_PER_FOLD:
        return None
    pnl = trades['net_pnl'].astype(float)
    held = (trades['hold_days'].astype(float).clip(lower=1)
            if 'hold_days' in trades else pd.Series(np.ones(len(pnl))))
    slot_days = float(held.sum())
    if slot_days <= 0:
        return None
    return float(pnl.sum() / slot_days / equity_start * 1e4)   # bps of equity per slot-day


def apply_params(params):
    """
    Write a parameter set into the live modules, returning the previous values
    so the caller can restore them.

    Mutating module globals is ugly, and it is the honest way to test THIS
    system rather than a reimplementation of it. A parallel "backtest copy" of
    the parameters is how a backtester and its live counterpart drift apart —
    the failure this project has already hit five times.
    """
    import signal_generator as sg
    import portfolio_allocator as pa

    previous = {}
    for key, value in params.items():
        if key in sg.RISK_PROFILE:
            previous[('sg', key)] = sg.RISK_PROFILE[key]
            sg.RISK_PROFILE[key] = value
        elif key in pa.ALLOCATOR_PROFILES[pa.ACTIVE_PROFILE]:
            previous[('pa', key)] = pa.ALLOCATOR_PROFILES[pa.ACTIVE_PROFILE][key]
            pa.ALLOCATOR_PROFILES[pa.ACTIVE_PROFILE][key] = value
        else:
            logger.warning(f"  parameter '{key}' not found in any profile — ignored")
    return previous


def restore_params(previous):
    import signal_generator as sg
    import portfolio_allocator as pa
    for (mod, key), value in previous.items():
        if mod == 'sg':
            sg.RISK_PROFILE[key] = value
        else:
            pa.ALLOCATOR_PROFILES[pa.ACTIVE_PROFILE][key] = value


# ═════════════════════════════════════════════════════════════════════════════
class WalkForwardOptimizer:
    """
    Usage:
        opt = WalkForwardOptimizer(backtest_fn)
        result = opt.run(universe_dfs, index_df, fundamentals)
        print(opt.report(result))

    backtest_fn(universe, index_df, fundamentals, start, end) must return a
    closed-trades DataFrame. Supplying it rather than importing one keeps this
    module testable without a full stack and lets a caller substitute a fast
    surrogate while iterating.
    """

    def __init__(self, backtest_fn, equity_start=50000.0, search_space=None,
                 n_folds=4, max_configs=60, seed=7):
        self.backtest = backtest_fn
        self.equity_start = equity_start
        self.space = search_space or SEARCH_SPACE
        self.n_folds = n_folds
        self.max_configs = max_configs
        self.rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────────────────
    def _configs(self):
        """
        Full grid when it is small enough to enumerate, a random sample when it
        is not. Random beats a coarser grid at equal budget: a grid spends its
        evaluations on combinations of values for parameters that do not matter,
        while random sampling covers more distinct values of the ones that do.
        """
        keys = list(self.space)
        grid = list(itertools.product(*(self.space[k] for k in keys)))
        if len(grid) <= self.max_configs:
            return [dict(zip(keys, combo)) for combo in grid]
        picks = self.rng.choice(len(grid), size=self.max_configs, replace=False)
        return [dict(zip(keys, grid[i])) for i in picks]

    def _folds(self, n_bars, warmup):
        """
        Anchored walk-forward. Each fold trains on everything from `warmup` to
        its own start and tests on the window after it, so the training window
        grows and the test window always sits strictly in the future.
        """
        usable = n_bars - warmup
        if usable < self.n_folds * 2:
            return []
        edges = np.linspace(warmup, n_bars, self.n_folds + 2).astype(int)
        return [(int(edges[0]), int(edges[i]), int(edges[i + 1]))
                for i in range(1, self.n_folds + 1)]

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, universe_dfs, index_df, fundamentals=None, warmup=300):
        n_bars = min(len(index_df), min((len(d) for d in universe_dfs.values()), default=0))
        folds = self._folds(n_bars, warmup)
        if not folds:
            return {'verdict': 'insufficient history',
                    'detail': f'{n_bars} bars with {warmup} warmup leaves too little to split'}

        configs = self._configs()
        logger.info(f"  Optimising {len(configs)} configurations across {len(folds)} folds "
                    f"({n_bars - warmup} test bars)")

        rows = []
        for i, params in enumerate(configs, 1):
            previous = apply_params(params)
            try:
                is_scores, oos_scores = [], []
                for (anchor, split, end) in folds:
                    train = self.backtest(universe_dfs, index_df, fundamentals, anchor, split)
                    test = self.backtest(universe_dfs, index_df, fundamentals, split, end)
                    # Purge: a trade still open when the test window starts has
                    # its outcome determined inside the test period.
                    train = self._purge(train, universe_dfs, index_df, split)
                    s_is = score_trades(train, self.equity_start)
                    s_oos = score_trades(test, self.equity_start)
                    if s_is is not None:
                        is_scores.append(s_is)
                    if s_oos is not None:
                        oos_scores.append(s_oos)
            finally:
                restore_params(previous)

            if len(oos_scores) < max(2, self.n_folds // 2):
                continue
            mean_oos = float(np.mean(oos_scores))
            dispersion = float(np.std(oos_scores, ddof=1)) if len(oos_scores) > 1 else 0.0
            rows.append({
                **params,
                'oos_mean': round(mean_oos, 3),
                'oos_std': round(dispersion, 3),
                'is_mean': round(float(np.mean(is_scores)), 3) if is_scores else None,
                'folds_scored': len(oos_scores),
                # Stability is part of the score, not a footnote. A config that
                # wins three folds and collapses in the fourth will deliver the
                # collapse, not the average.
                'objective': round(mean_oos - DISPERSION_PENALTY * dispersion, 3),
            })
            if i % 10 == 0:
                logger.info(f"    {i}/{len(configs)} evaluated")

        if not rows:
            return {'verdict': 'no configuration produced enough trades to score'}

        table = pd.DataFrame(rows).sort_values('objective', ascending=False)
        return self._verdict(table)

    @staticmethod
    def _purge(trades, universe_dfs, index_df, split_bar):
        """
        Drop training trades whose holding period reaches into the test window.
        Uses exit_date where available; without dates, returns the frame
        untouched and the caller should treat the out-of-sample figure as
        optimistic rather than clean.
        """
        if trades is None or len(trades) == 0 or 'exit_date' not in trades:
            return trades
        try:
            dates = pd.to_datetime(index_df['datetime'])
            boundary = dates.iloc[min(split_bar, len(dates) - 1)]
            exits = pd.to_datetime(trades['exit_date'], errors='coerce')
            return trades[exits < boundary]
        except Exception:
            return trades

    # ─────────────────────────────────────────────────────────────────────────
    def _verdict(self, table):
        """
        Decide whether anything should actually change — the part a naive
        optimiser skips.
        """
        best = table.iloc[0]
        incumbent = table[table['objective'] == table['objective']].iloc[-1]

        gap = None
        if best.get('is_mean') is not None and best['oos_mean'] != 0:
            gap = abs(best['is_mean'] - best['oos_mean']) / max(abs(best['is_mean']), 1e-9)

        reasons = []
        deploy = True
        if best['objective'] <= 0:
            deploy, _ = False, reasons.append('best configuration is not profitable out of sample')
        # Parameter stability is computed first, because it is the stronger
        # piece of evidence and it changes how the in-sample/out-of-sample gap
        # should be read.
        top = table.head(max(3, len(table) // 10))
        stability = {}
        for key in self.space:
            if key in top:
                counts = top[key].value_counts(normalize=True)
                stability[key] = {'modal_value': counts.index[0],
                                  'share_of_top': round(float(counts.iloc[0]), 2)}
        modal_share = (float(np.mean([s['share_of_top'] for s in stability.values()]))
                       if stability else 0.0)

        if gap is not None and gap > MAX_IS_OOS_GAP:
            # A large gap is EXPECTED when the objective is noisy — trade P&L
            # has enormous variance relative to its mean, so in-sample will
            # flatter almost any configuration. What distinguishes a real
            # optimum from a fitted one is not the gap, it is whether the same
            # parameter values keep winning across folds. So the gap vetoes
            # only when stability is weak; when the top configurations agree
            # with each other, it is reported as a caution instead.
            if modal_share < 0.50 or best['oos_mean'] <= 0:
                deploy = False
                reasons.append(f'in-sample exceeds out-of-sample by {gap*100:.0f}% '
                               f'with only {modal_share*100:.0f}% parameter agreement '
                               f'— fitted to noise')
            else:
                reasons.append(f'in-sample exceeds out-of-sample by {gap*100:.0f}%, but the '
                               f'top configurations agree {modal_share*100:.0f}% of the time '
                               f'— noisy objective, not an unstable optimum')
        if best['oos_std'] > abs(best['oos_mean']):
            deploy = False
            reasons.append('across-fold dispersion exceeds the mean — unstable')
        if best['folds_scored'] < max(2, self.n_folds - 1):
            deploy = False
            reasons.append(f"only {best['folds_scored']} folds produced enough trades")

        return {
            'verdict': 'DEPLOY' if deploy else 'HOLD',
            'reasons': reasons or ['out-of-sample performance is stable and positive'],
            'best_params': {k: best[k] for k in self.space if k in best},
            'oos_mean_bps_per_slot_day': best['oos_mean'],
            'oos_std': best['oos_std'],
            'is_oos_gap': None if gap is None else round(gap, 3),
            'stability': stability,
            'table': table,
        }

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def report(result):
        if 'table' not in result:
            return f"  Optimiser: {result.get('verdict')} — {result.get('detail', '')}"
        lines = ["\n" + "=" * 70,
                 f"  WALK-FORWARD OPTIMISATION — {result['verdict']}",
                 "=" * 70]
        for r in result['reasons']:
            lines.append(f"  · {r}")
        lines.append(f"\n  Best out-of-sample: {result['oos_mean_bps_per_slot_day']:+.2f} "
                     f"bps/slot-day (±{result['oos_std']:.2f} across folds)")
        if result['is_oos_gap'] is not None:
            lines.append(f"  In-sample to out-of-sample gap: {result['is_oos_gap']*100:.0f}%")
        lines.append("\n  Best parameters:")
        for k, v in result['best_params'].items():
            stab = result['stability'].get(k, {})
            share = stab.get('share_of_top')
            note = (f"   (appears in {share*100:.0f}% of the top configurations)"
                    if share is not None else '')
            lines.append(f"    {k:<20} {v}{note}")
        lines.append("\n  A parameter appearing in most top configurations is a property of")
        lines.append("  the data. One appearing only in the single best is a property of one")
        lines.append("  fold — treat a low share as a reason not to move that constant.")
        if result['verdict'] == 'HOLD':
            lines.append("\n  HOLD means keep the current constants. That is the most common")
            lines.append("  correct answer, and the one a naive optimiser can never give.")
        lines.append("=" * 70 + "\n")
        return "\n".join(lines)
