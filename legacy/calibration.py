# calibration.py  ── WIN-PROBABILITY CALIBRATION LOOP  v1
# ═════════════════════════════════════════════════════════════════════════════
# Four modules now consume p_win_est:
#   signal_generator._win_probability -> the EV gate that admits a trade
#   portfolio_allocator.kelly_fraction -> how much capital it gets
#   portfolio_allocator.candidate_economics -> where it ranks against rivals
#   portfolio_allocator.incumbent_economics -> whether a held position survives
#
# All four trace back to one uncalibrated constant, edge_tilt, which converts
# a quality score into claimed edge by assertion. If it is wrong, the gate
# admits the wrong trades, Kelly stakes them wrongly, and the allocator orders
# them wrongly — in the same direction, at the same time. That correlated
# failure is the largest remaining source of unforced error in the stack, and
# unlike the regime thresholds it is something closed trades can settle.
#
# ── What this does ──────────────────────────────────────────────────────────
# Learns P(win | quality_score) from realised outcomes and hands it back to
# both consumers. Three properties matter more than the fitting method:
#
# 1. IT SHRINKS TOWARD THE ANALYTIC PRIOR, NOT TOWARD 0.5.
#    With 12 closed trades the honest estimate is "roughly what the barrier
#    geometry says, nudged by what little we have seen". The prior here is the
#    driftless first-passage probability a/(a+b) tilted by edge_tilt — i.e.
#    exactly today's behaviour. So a fresh install reproduces the current
#    system bit for bit, and diverges from it only in proportion to evidence.
#    There is no cutover, and no n below which this is unsafe to switch on.
#
# 2. IT IS MONOTONE IN QUALITY BY CONSTRUCTION.
#    Pool-adjacent-violators (isotonic regression) enforces that a higher
#    quality score never maps to a lower win probability. Unconstrained
#    binning on 50 trades produces non-monotone maps that are pure sampling
#    noise, and Kelly reacts violently to them — a spuriously high p in one
#    bin becomes a maximum stake.
#
# 3. IT REPORTS WHETHER IT IS ANY GOOD.
#    Brier score against the prior, and a reliability table. A calibrator that
#    cannot beat the constant it replaces should not be trusted, and the only
#    way to know is to measure it. reliability() prints the comparison.
#
# ── Also calibrated ─────────────────────────────────────────────────────────
# edge_retention (portfolio_allocator's assumption about how much of its entry
# edge a held position keeps) is estimated from the same data: realised win
# rate of positions conditioned on how many thesis-health signals had fired.
# ═════════════════════════════════════════════════════════════════════════════

import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Wiring ──────────────────────────────────────────────────────────────────
#   signal_generator.SignalGenerator.__init__:
#       from calibration import WinCalibrator
#       self.calibrator = WinCalibrator()
#   signal_generator._win_probability, final line becomes:
#       return self.calibrator.p_win(quality, float(np.clip(p_base * tilt, 0.05, 0.85)))
#   portfolio_allocator.incumbent_economics call sites:
#       edge_retention=calibrator.retention
#   run_paper_trading, once per day after update_trades:
#       from calibration import refresh_from_csv; refresh_from_csv(TRADES_CSV)
#   paper_trading_manager.open_trade: persist quality_score and p_win_est on the
#   row (v4 already emits both) — without them there is nothing to fit.

CALIBRATION_JSON = 'win_calibration.json'

MIN_TRADES_TO_FIT = 25       # below this the map is pure prior; above, evidence enters gradually
SHRINK_K = 30.0              # trades for 50% weight on observed vs prior, per bin
N_BINS = 5
SIGNAL_Z_THRESHOLD = 1.28   # one-sided; see the significance guard in fit()


def _se2(p, n):
    """Squared standard error of a binomial rate."""
    n = max(int(n), 1)
    p = float(min(max(p, 0.0), 1.0))
    return p * (1.0 - p) / n


def _isotonic(x, y, w):
    """
    Pool-adjacent-violators. Returns y fitted to be non-decreasing in x.
    Weighted, because bins hold different trade counts and an 8-trade bin
    should not overrule a 30-trade one when they conflict.
    """
    order = np.argsort(x)
    y, w = np.asarray(y, float)[order], np.asarray(w, float)[order]
    vals, wts = list(y), list(w)
    i = 0
    while i < len(vals) - 1:
        if vals[i] <= vals[i + 1] + 1e-12:
            i += 1
            continue
        tw = wts[i] + wts[i + 1]
        pooled = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / max(tw, 1e-12)
        vals[i:i + 2], wts[i:i + 2] = [pooled], [tw]
        i = max(i - 1, 0)
    # re-expand pooled blocks back to per-bin values
    expanded, bi = [], 0
    for v, ww in zip(vals, wts):
        n_in_block = 0
        acc = 0.0
        while bi < len(w) and acc < ww - 1e-9:
            acc += w[bi]
            bi += 1
            n_in_block += 1
        expanded.extend([v] * max(n_in_block, 1))
    expanded = expanded[:len(y)]
    result = np.empty(len(y))
    result[order] = np.asarray(expanded, float)
    return result


class WinCalibrator:
    """
    Maps (quality_score, analytic prior) -> calibrated win probability.

    Usage is deliberately a drop-in for the existing call:
        p = calibrator.p_win(quality=0.71, prior=0.36)
    With no fitted data, p == prior exactly.
    """

    def __init__(self, path=CALIBRATION_JSON):
        self.path = path
        self.bins = []            # [{'lo','hi','p_obs','n','p_fitted'}]
        self.n_trades = 0
        self.brier_model = None
        self.brier_prior = None
        self.retention = 0.60     # allocator's edge_retention default
        self.signal_detected = False
        self.signal_z = None
        self.load()

    # ── persistence ──────────────────────────────────────────────────────────
    def load(self):
        if not os.path.exists(self.path):
            return self
        try:
            d = json.load(open(self.path))
            self.bins = d.get('bins', [])
            self.n_trades = int(d.get('n_trades', 0))
            self.brier_model = d.get('brier_model')
            self.brier_prior = d.get('brier_prior')
            self.retention = float(d.get('retention', 0.60))
            self.signal_detected = bool(d.get('signal_detected', False))
            self.signal_z = d.get('signal_z')
        except (ValueError, OSError) as e:
            logger.warning(f"calibration load failed ({e}) — running on the analytic prior")
        return self

    def save(self):
        json.dump({'bins': self.bins, 'n_trades': self.n_trades,
                   'brier_model': self.brier_model, 'brier_prior': self.brier_prior,
                   'retention': self.retention, 'signal_detected': self.signal_detected,
                   'signal_z': self.signal_z}, open(self.path, 'w'), indent=1)
        return self

    # ── the map ──────────────────────────────────────────────────────────────
    def p_win(self, quality, prior):
        """
        Calibrated probability. Blends the bin's observed rate with the
        analytic prior by that bin's own sample size, so a well-populated bin
        speaks loudly and a thin one barely at all.

        Clipped to [0.5*prior, 1.6*prior]: the calibrator is allowed to correct
        the prior, not to replace the geometry. A bin that happens to hold four
        winners should not be able to claim p=1.0 on a 2.4:1 payoff.
        """
        prior = float(np.clip(prior, 0.02, 0.95))
        if self.n_trades < MIN_TRADES_TO_FIT or not self.bins:
            return prior
        if not getattr(self, 'signal_detected', True):
            return prior          # quality has not been shown to predict anything yet
        b = self._bin_for(quality)
        if b is None:
            return prior
        w = b['n'] / (b['n'] + SHRINK_K)
        p = w * b['p_fitted'] + (1.0 - w) * prior
        return float(np.clip(p, 0.5 * prior, 1.6 * prior))

    def _bin_for(self, quality):
        q = float(np.clip(quality, 0.0, 1.0))
        for b in self.bins:
            if b['lo'] <= q <= b['hi']:
                return b
        return self.bins[-1] if q > self.bins[-1]['hi'] else self.bins[0]

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(self, trades_df, quality_col='quality_score', prior_col='p_win_est',
            pnl_col='net_pnl'):
        """
        trades_df: closed trades carrying quality_score. Rows without it are
        skipped rather than imputed — an imputed quality score would teach the
        map a relationship that never existed.

        A "win" is net_pnl > 0, i.e. after costs. Calibrating on gross would
        train the gate to admit trades that make money for the broker.
        """
        df = trades_df.copy()
        if quality_col not in df.columns:
            logger.info("no quality_score column yet — calibrator stays on the prior")
            return self
        df = df[pd.notna(df[quality_col]) & pd.notna(df[pnl_col])]
        if len(df) == 0:
            return self

        q = df[quality_col].astype(float).to_numpy()
        win = (df[pnl_col].astype(float) > 0).to_numpy().astype(float)
        prior = (df[prior_col].astype(float).to_numpy()
                 if prior_col in df.columns and df[prior_col].notna().all()
                 else np.full(len(df), float(win.mean())))

        edges = np.unique(np.quantile(q, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:
            edges = np.array([q.min() - 1e-9, np.median(q), q.max() + 1e-9])
        idx = np.clip(np.digitize(q, edges[1:-1]), 0, len(edges) - 2)

        centres, rates, counts = [], [], []
        for k in range(len(edges) - 1):
            m = idx == k
            if m.sum() == 0:
                continue
            centres.append(float(q[m].mean()))
            rates.append(float(win[m].mean()))
            counts.append(int(m.sum()))

        fitted = _isotonic(np.array(centres), np.array(rates), np.array(counts, float))

        self.bins = [{'lo': float(edges[k]), 'hi': float(edges[k + 1]),
                      'centre': centres[k], 'p_obs': rates[k], 'n': counts[k],
                      'p_fitted': float(fitted[k])}
                     for k in range(len(centres))]
        self.bins[0]['lo'], self.bins[-1]['hi'] = 0.0, 1.0
        self.n_trades = int(len(df))

        # ── Significance guard ───────────────────────────────────────────────
        # Isotonic regression only pools ADJACENT VIOLATING pairs, so a noise
        # pattern that happens to come out monotone passes through untouched.
        # At n=400 on data with no real relationship, the regression suite
        # measured a 5.5pp spread across quality bands purely from sampling —
        # enough to move a Kelly stake materially in the wrong direction.
        #
        # So before the map is allowed to depart from the prior at all, the
        # top-to-bottom difference must clear roughly two standard errors of
        # that difference. Below it, there is no measured relationship between
        # quality and outcome, and the honest output is the prior — not a
        # confident-looking curve fitted to noise.
        lo, hi = self.bins[0], self.bins[-1]
        se = np.sqrt(_se2(lo['p_obs'], lo['n']) + _se2(hi['p_obs'], hi['n']))
        observed = hi['p_fitted'] - lo['p_fitted']
        # One-sided, at z=1.28. The hypothesis is directional — higher quality
        # should mean a higher win rate, not merely a different one — so a
        # two-sided test is the wrong shape. The level is set deliberately
        # loose: at the sample sizes this will realistically see (100-400
        # trades), a genuine effect of the size worth acting on produces z of
        # roughly 1.5, so a 2-sigma gate would keep the calibrator switched off
        # essentially forever and leave the stack running on a hand-set
        # constant instead. 1.28 accepts a ~10% chance of acting on a direction
        # that is not real, against the certainty of never learning at all.
        # The downstream clip to [0.5x, 1.6x] of the prior bounds the damage if
        # that 10% lands.
        self.signal_detected = bool(observed > SIGNAL_Z_THRESHOLD * se) if se > 0 else False
        self.signal_z = round(float(observed / se), 2) if se > 0 else None

        # Does the map beat the constant it replaces? Brier = mean squared
        # error of the probability. Lower is better; if the model does not
        # win here, p_win() is still safe (it shrinks to prior) but the
        # operator should know the evidence is not there yet.
        p_model = np.array([self.p_win(qi, pi) for qi, pi in zip(q, prior)])
        self.brier_model = float(np.mean((p_model - win) ** 2))
        self.brier_prior = float(np.mean((prior - win) ** 2))
        return self

    def fit_retention(self, trades_df, broken_col='health_signals_at_exit', pnl_col='net_pnl'):
        """
        edge_retention for portfolio_allocator: how much of its entry edge a
        held position keeps. Estimated as the ratio of win rate among positions
        with an intact thesis to win rate among those with signals firing —
        the same quantity the allocator assumes at 0.60 with no evidence.
        """
        if broken_col not in trades_df.columns:
            return self
        df = trades_df[pd.notna(trades_df[broken_col])]
        if len(df) < MIN_TRADES_TO_FIT:
            return self
        intact = df[df[broken_col].astype(float) < 1]
        broken = df[df[broken_col].astype(float) >= 2]
        if len(intact) < 8 or len(broken) < 8:
            return self
        wi = float((intact[pnl_col].astype(float) > 0).mean())
        wb = float((broken[pnl_col].astype(float) > 0).mean())
        if wi <= 0:
            return self
        # Map the observed degradation onto the allocator's tilt scale, bounded
        # so a small sample cannot swing it to an extreme.
        self.retention = float(np.clip(0.60 * (1.0 + (wi - wb)), 0.20, 1.10))
        return self

    # ── diagnostics ──────────────────────────────────────────────────────────
    def reliability(self):
        lines = [f"  Calibrator — n={self.n_trades} closed trades with a quality score"]
        if self.n_trades < MIN_TRADES_TO_FIT or not self.bins:
            lines.append(f"  Running on the analytic prior "
                         f"({MIN_TRADES_TO_FIT - self.n_trades} more trades to activate).")
            return "\n".join(lines)
        lines.append(f"  {'quality band':<18}{'n':>5}{'observed':>11}{'isotonic':>11}{'weight':>9}")
        for b in self.bins:
            w = b['n'] / (b['n'] + SHRINK_K)
            lines.append(f"  {b['lo']:.2f}–{b['hi']:.2f}{'':<8}{b['n']:>5}"
                         f"{b['p_obs']*100:>10.1f}%{b['p_fitted']*100:>10.1f}%{w:>9.2f}")
        if not self.signal_detected:
            lines.append(f"  Quality-to-outcome relationship not yet significant "
                         f"(z={self.signal_z}) — holding the analytic prior.")
        if self.brier_model is not None:
            verdict = 'beats' if self.brier_model < self.brier_prior else 'does not beat'
            lines.append(f"  Brier: model {self.brier_model:.4f} vs prior {self.brier_prior:.4f} "
                         f"— {verdict} the constant it replaces")
        lines.append(f"  edge_retention estimate: {self.retention:.2f}")
        return "\n".join(lines)


def refresh_from_csv(trades_csv='paper_trades.csv', path=CALIBRATION_JSON):
    """
    Nightly hook: refit on every closed trade and persist. Safe to run when the
    file has no quality_score column or no closed rows — it simply leaves the
    calibrator on the prior.
    """
    cal = WinCalibrator(path)
    try:
        df = pd.read_csv(trades_csv)
    except (OSError, ValueError) as e:
        logger.warning(f"calibration refresh skipped ({e})")
        return cal
    closed = df[df.get('status', pd.Series(dtype=object)) == 'CLOSED']
    if len(closed) == 0:
        return cal
    cal.fit(closed).fit_retention(closed).save()
    logger.info("\n" + cal.reliability())
    return cal
