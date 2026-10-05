# meta_model.py  ── ACCURACY ENGINE (META-LABELLING)  v1
# ═════════════════════════════════════════════════════════════════════════════
# signal_generator answers "is this a trade". This answers the strictly
# different question: GIVEN that it said yes, will this one win?
#
# That separation is the whole idea. A primary model tuned to find setups and a
# secondary model tuned to grade them are solving different problems, and
# collapsing them into one scorer forces a single threshold to serve both. Keep
# them apart and the primary can stay permissive — more candidates, higher
# recall — while the secondary decides which of them deserve capital. It is
# also the only place in this stack where the FULL feature vector of a signal
# is compared against what actually happened to it.
#
# ── Why this and not "add more indicators" ──────────────────────────────────
# The stack already computes ~20 numbers per signal: six quality components,
# stop distance in sigma, realised R:R, efficiency ratio, extension, DI spread,
# cost in bps, the pattern, the regime, the breadth of the tape that day. Every
# one of them is currently used through a hand-written rule with a hand-set
# threshold. Nothing has ever checked which of them actually separate winners
# from losers, or whether some are pulling in the opposite direction to their
# assumed sign. This checks.
#
# ── What it will and will not do ────────────────────────────────────────────
# It will NOT activate on thin data, and it will not activate on data where it
# cannot beat the prior OUT OF SAMPLE. Both guards are non-negotiable, because
# a model fitted to 40 trades will always look excellent in-sample and will
# always be noise. Until both clear, p_win() returns exactly what the existing
# calibrated prior returns, so installing this changes nothing until it has
# earned the right to change something.
#
# ── Implementation notes ────────────────────────────────────────────────────
# L2-regularised logistic regression by Newton-IRLS, numpy only — no sklearn,
# no scipy, nothing to add to a GitHub Actions runner. Logistic rather than a
# tree ensemble on purpose: at a few hundred samples a boosted forest will
# memorise, its feature importances will be unstable run to run, and the
# coefficients here are readable, which matters when the output is allowed to
# size a position.
#
# Validation is PURGED walk-forward: folds are time-ordered, and trades whose
# holding period overlaps the test window are dropped from training. Without
# purging, a trade opened three days before the split and closed inside it
# leaks its outcome backwards, and the out-of-sample score becomes fiction.
# ═════════════════════════════════════════════════════════════════════════════

import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = 'meta_model.json'

MIN_TRADES_TO_FIT = 80      # below this, any fit is memorisation
MIN_TRADES_PER_FOLD = 20
N_FOLDS = 4
EMBARGO_BARS = 3            # sessions purged either side of a fold boundary
L2_LAMBDA = 1.0             # ridge penalty; deliberately strong for small n
MAX_NEWTON_ITERS = 60
BLEND_CEILING = 0.65        # the model never fully replaces the prior

# Features read off a signal's details dict. Missing ones are imputed to the
# training median and flagged, rather than zero-filled — zero is a meaningful
# value for several of these and imputing it invents a signal.
NUMERIC_FEATURES = [
    'quality_score', 'p_win_est', 'risk_pct_of_price', 'stop_sigma_mult',
    'risk_reward_ratio', 'sigma_daily_pct', 'efficiency_ratio', 'time_exit_bars',
    'confidence', 'band_usage', 'gap_down_p90',
]
QUALITY_PARTS = ['trend', 'dmi', 'efficiency', 'volume', 'non_extension', 'rel_strength']
INDICATOR_FEATURES = ['rsi', 'adx', 'plus_di', 'minus_di', 'cmf', 'extension_atr', 'stoch_k']
CATEGORICAL = {
    'entry_type': ['pullback', 'breakout', 'cmf_accum', 'stoch_cross', 'ema_cross',
                   'momentum_burst', 'bb_squeeze', 'engulfing', 'rsi_divergence'],
    'market_state_at_entry': ['RISK_ON', 'NEUTRAL', 'RISK_OFF', 'DEFENSIVE'],
}


# ═════════════════════════════════════════════════════════════════════════════
# FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def feature_names():
    names = list(NUMERIC_FEATURES)
    names += [f'q_{p}' for p in QUALITY_PARTS]
    names += [f'ind_{i}' for i in INDICATOR_FEATURES]
    names += ['di_spread', 'cost_bps', 'notional_log']
    for col, levels in CATEGORICAL.items():
        names += [f'{col}={lv}' for lv in levels[:-1]]     # drop-one encoding
    return names


def extract_features(row):
    """
    One feature vector from either a live signal `details` dict or a closed
    trade row. Both shapes are accepted because the model must be trainable on
    the trade log and callable on a fresh signal, and forcing one shape on the
    other is how a training/serving skew gets introduced.
    """
    get = row.get if hasattr(row, 'get') else (lambda k, d=None: row[k] if k in row else d)
    ind = get('indicators') or {}
    parts = get('quality_breakdown') or {}
    econ = get('economics') or {}
    if not isinstance(ind, dict):
        ind = {}
    if not isinstance(parts, dict):
        parts = {}
    if not isinstance(econ, dict):
        econ = {}

    vals = [_num(get(f)) for f in NUMERIC_FEATURES]
    vals += [_num(parts.get(p)) for p in QUALITY_PARTS]
    vals += [_num(ind.get(i, get(f'ind_{i}'))) for i in INDICATOR_FEATURES]

    pdi, mdi = _num(ind.get('plus_di')), _num(ind.get('minus_di'))
    vals.append(pdi - mdi if (pdi is not None and mdi is not None) else None)
    vals.append(_num(econ.get('cost_bps', get('cost_bps'))))
    notional = _num(econ.get('notional'))
    if notional is None:
        e, s = _num(get('entry_price')), _num(get('position_size'))
        notional = e * s if (e and s) else None
    vals.append(float(np.log(notional)) if notional and notional > 0 else None)

    for col, levels in CATEGORICAL.items():
        actual = str(get(col) or '')
        vals += [1.0 if actual == lv else 0.0 for lv in levels[:-1]]
    return vals


def _num(v):
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


# ═════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION (Newton-IRLS, ridge-penalised)
# ═════════════════════════════════════════════════════════════════════════════
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _fit_logistic(X, y, lam=L2_LAMBDA):
    """
    Newton-IRLS with an L2 penalty that is NOT applied to the intercept —
    penalising the intercept would bias the fitted base rate toward 0.5, which
    on a 45%-win-rate problem is a systematic distortion of exactly the
    quantity being estimated.

    Returns the coefficient vector, or None when the problem is degenerate
    (single class, singular Hessian). Degenerate is common early and must
    return None rather than a fitted-looking answer.
    """
    n, p = X.shape
    if n == 0 or len(np.unique(y)) < 2:
        return None
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    penalty = np.eye(p + 1) * lam
    penalty[0, 0] = 0.0

    for _ in range(MAX_NEWTON_ITERS):
        eta = Xb @ beta
        mu = _sigmoid(eta)
        W = np.clip(mu * (1 - mu), 1e-6, None)
        grad = Xb.T @ (y - mu) - penalty @ beta
        H = -(Xb.T * W) @ Xb - penalty
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        beta_new = beta - step
        if not np.all(np.isfinite(beta_new)):
            return None
        if np.max(np.abs(beta_new - beta)) < 1e-7:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _predict(beta, X):
    return _sigmoid(np.hstack([np.ones((X.shape[0], 1)), X]) @ beta)


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _auc(p, y):
    """Rank-based AUC. Ties share their average rank, so a constant predictor
    scores exactly 0.5 rather than an accidental 1.0."""
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(np.asarray(p, dtype=float))
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    sorted_p = np.asarray(p, dtype=float)[order]
    i = 0
    while i < len(sorted_p):                       # average ranks within ties
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ═════════════════════════════════════════════════════════════════════════════
# THE ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class MetaModel:
    """
    p_win(details, prior) -> calibrated probability.

    Inert until it has both enough data and demonstrated out-of-sample skill.
    Until then it returns `prior` unchanged, so it is safe to wire in on day
    one and it costs nothing to leave installed while it waits.
    """

    def __init__(self, path=MODEL_PATH):
        self.path = path
        self.beta = None
        self.mu = None          # feature means, for standardisation
        self.sd = None
        self.median = None      # training medians, for imputing missing features
        self.keep_mask = None   # which features survived the coverage filter
        self.n_trades = 0
        self.active = False
        self.oos_auc = None
        self.oos_brier = None
        self.prior_brier = None
        self.blend_weight = 0.0
        self.load()

    # ── persistence ──────────────────────────────────────────────────────────
    def load(self):
        if not os.path.exists(self.path):
            return self
        try:
            d = json.load(open(self.path))
            self.beta = np.array(d['beta']) if d.get('beta') else None
            self.mu = np.array(d['mu']) if d.get('mu') else None
            self.sd = np.array(d['sd']) if d.get('sd') else None
            self.median = np.array(d['median']) if d.get('median') else None
            self.keep_mask = np.array(d['keep_mask'], dtype=bool) if d.get('keep_mask') else None
            self.n_trades = int(d.get('n_trades', 0))
            self.active = bool(d.get('active', False))
            self.oos_auc = d.get('oos_auc')
            self.oos_brier = d.get('oos_brier')
            self.prior_brier = d.get('prior_brier')
            self.blend_weight = float(d.get('blend_weight', 0.0))
        except (ValueError, OSError, KeyError) as e:
            logger.warning(f"meta model unreadable ({e}) — running on the prior")
        return self

    def save(self):
        try:
            json.dump({'beta': None if self.beta is None else self.beta.tolist(),
                       'mu': None if self.mu is None else self.mu.tolist(),
                       'sd': None if self.sd is None else self.sd.tolist(),
                       'median': None if self.median is None else self.median.tolist(),
                       'keep_mask': None if self.keep_mask is None else [bool(b) for b in self.keep_mask],
                       'n_trades': self.n_trades, 'active': self.active,
                       'oos_auc': self.oos_auc, 'oos_brier': self.oos_brier,
                       'prior_brier': self.prior_brier,
                       'blend_weight': self.blend_weight},
                      open(self.path, 'w'), indent=1)
        except OSError as e:
            logger.warning(f"could not save meta model: {e}")
        return self

    # ── inference ────────────────────────────────────────────────────────────
    def p_win(self, details, prior):
        """
        Blended probability. The blend weight is capped at BLEND_CEILING, so
        even a model with real skill never wholly replaces the geometry-derived
        prior — the barrier arithmetic is a hard constraint on what is
        achievable, and a statistical model that disagrees with it by a wide
        margin is more likely wrong than right.
        """
        prior = float(np.clip(prior, 0.02, 0.95))
        if not self.active or self.beta is None:
            return prior
        try:
            x = self._vector(extract_features(details))
            p_model = float(_predict(self.beta, x.reshape(1, -1))[0])
        except Exception:
            return prior
        w = self.blend_weight
        p = w * p_model + (1.0 - w) * prior
        # Same containment as the calibrator: correct the prior, do not replace
        # the geometry that produced it.
        return float(np.clip(p, 0.5 * prior, 1.6 * prior))

    def _vector(self, raw):
        x = np.array([np.nan if v is None else float(v) for v in raw], dtype=float)
        if self.keep_mask is not None and len(self.keep_mask) == len(x):
            x = x[self.keep_mask]
        if self.median is not None:
            miss = ~np.isfinite(x)
            x[miss] = self.median[miss]
        x = np.where(np.isfinite(x), x, 0.0)
        if self.mu is not None and self.sd is not None:
            x = (x - self.mu) / self.sd
        return x

    # ── training ─────────────────────────────────────────────────────────────
    def fit(self, trades_df, label_col='net_pnl', date_col='entry_date'):
        """
        Fits on CLOSED trades and validates with purged walk-forward folds.

        The label is net_pnl > 0 — after costs. Training on gross would teach
        the model to prefer trades that make money for the broker, which is
        precisely the failure mode this project has already paid for once.
        """
        if trades_df is None or len(trades_df) == 0:
            return self
        df = trades_df.copy()
        if 'status' in df.columns:
            df = df[df['status'] == 'CLOSED']
        df = df[pd.notna(df.get(label_col))]
        if len(df) < MIN_TRADES_TO_FIT:
            self.n_trades = len(df)
            self.active = False
            logger.info(f"  Meta model: {len(df)}/{MIN_TRADES_TO_FIT} trades — holding the prior")
            return self.save()

        if date_col in df.columns:
            df = df.sort_values(date_col)
        X_raw = np.array([extract_features(r) for _, r in df.iterrows()], dtype=object)
        X = np.array([[np.nan if v is None else float(v) for v in row] for row in X_raw],
                     dtype=float)
        y = (df[label_col].astype(float) > 0).to_numpy().astype(float)
        prior = (df['p_win_est'].astype(float).to_numpy()
                 if 'p_win_est' in df.columns and df['p_win_est'].notna().all()
                 else np.full(len(df), float(y.mean())))

        # Drop features that are missing almost everywhere — an all-NaN column
        # imputes to a constant and contributes nothing but a coefficient to
        # overfit with.
        keep = np.mean(np.isfinite(X), axis=0) > 0.5
        X = X[:, keep]
        # Persisted, and applied identically at inference. A keep-mask that
        # exists only during training is a textbook training/serving skew: the
        # served vector would be a different length or, worse, silently
        # misaligned column-for-column against the fitted coefficients.
        self.keep_mask = keep

        self.median = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        X = np.where(np.isfinite(X), X, self.median)
        self.mu = X.mean(axis=0)
        self.sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Xs = (X - self.mu) / self.sd

        oos_p, oos_y, oos_prior = self._walk_forward(Xs, y, prior, df, date_col)

        if len(oos_y) >= MIN_TRADES_PER_FOLD:
            self.oos_auc = round(_auc(oos_p, oos_y), 3)
            self.prior_brier = round(_brier(oos_prior, oos_y), 4)

            # Score the BLEND, not the raw model — the blend is what actually
            # gets used, and the two are different objects. A logistic fit on a
            # few hundred samples routinely has real DISCRIMINATION (it ranks
            # winners above losers) and poor CALIBRATION (its absolute
            # probabilities are off), while the analytic prior has the opposite
            # profile. Judging the raw model on Brier therefore rejects models
            # that would improve the system, because it measures the one
            # property the prior already supplies.
            #
            # The weight is chosen on pooled out-of-sample data — one parameter
            # over one pooled set, which is a mild selection effect and is
            # declared rather than hidden. It is then capped by sample size and
            # by BLEND_CEILING, so a weight chosen on 90 trades cannot act like
            # one earned on 900.
            size_cap = min(BLEND_CEILING, len(df) / (len(df) + 150.0))
            best_w, best_brier = 0.0, self.prior_brier
            for w in np.arange(0.05, size_cap + 1e-9, 0.05):
                blended = w * oos_p + (1.0 - w) * oos_prior
                b = _brier(blended, oos_y)
                if b < best_brier:
                    best_w, best_brier = float(w), b
            self.oos_brier = round(best_brier, 4)
            self.blend_weight = round(best_w, 3)
            self.active = bool(best_w > 0 and self.oos_auc > 0.53
                               and best_brier < self.prior_brier)
        else:
            self.active = False

        self.beta = _fit_logistic(Xs, y)
        if self.beta is None:
            self.active = False
        self.n_trades = int(len(df))

        logger.info(f"  Meta model: n={self.n_trades} | OOS AUC {self.oos_auc} | "
                    f"Brier {self.oos_brier} vs prior {self.prior_brier} | "
                    f"{'ACTIVE w=' + str(self.blend_weight) if self.active else 'inactive'}")
        return self.save()

    def _walk_forward(self, X, y, prior, df, date_col):
        """
        Anchored walk-forward with purging. Fold k trains on everything before
        the fold and tests on the fold, and any training trade still OPEN when
        the test window begins is dropped — its outcome is contemporaneous with
        the test period and would leak.
        """
        n = len(y)
        bounds = np.linspace(n // (N_FOLDS + 1), n, N_FOLDS + 1).astype(int)
        oos_p, oos_y, oos_prior = [], [], []
        exits = (pd.to_datetime(df['exit_date'], errors='coerce').to_numpy()
                 if 'exit_date' in df.columns else None)
        entries = (pd.to_datetime(df[date_col], errors='coerce').to_numpy()
                   if date_col in df.columns else None)

        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo < MIN_TRADES_PER_FOLD // 2:
                continue
            train_idx = np.arange(0, lo)
            if exits is not None and entries is not None and lo < n:
                test_start = entries[lo]
                overlap = exits[train_idx] >= test_start
                train_idx = train_idx[~overlap]
            if len(train_idx) < MIN_TRADES_PER_FOLD or len(np.unique(y[train_idx])) < 2:
                continue
            beta = _fit_logistic(X[train_idx], y[train_idx])
            if beta is None:
                continue
            test_idx = np.arange(lo, hi)
            oos_p.extend(_predict(beta, X[test_idx]).tolist())
            oos_y.extend(y[test_idx].tolist())
            oos_prior.extend(prior[test_idx].tolist())
        return np.array(oos_p), np.array(oos_y), np.array(oos_prior)

    # ── reporting ────────────────────────────────────────────────────────────
    def report(self):
        lines = [f"  Meta model — {self.n_trades} closed trades"]
        if not self.active:
            lines.append(f"  Inactive: {'insufficient data' if self.n_trades < MIN_TRADES_TO_FIT else 'no out-of-sample skill demonstrated'}"
                         f" — p_win falls through to the calibrated prior.")
            if self.oos_auc is not None:
                lines.append(f"  Last check: OOS AUC {self.oos_auc}, Brier {self.oos_brier} "
                             f"vs prior {self.prior_brier}")
            return "\n".join(lines)
        lines.append(f"  ACTIVE  OOS AUC {self.oos_auc}  Brier {self.oos_brier} "
                     f"vs prior {self.prior_brier}  blend weight {self.blend_weight}")
        if self.beta is not None and self.sd is not None:
            names = [n for n, k in zip(feature_names(), (self.keep_mask if self.keep_mask is not None
                                                                 else [True] * len(feature_names()))) if k]
            coefs = self.beta[1:]
            if len(names) == len(coefs):
                top = sorted(zip(names, coefs), key=lambda kv: -abs(kv[1]))[:8]
                lines.append("  Strongest standardised coefficients "
                             "(sign = direction on P(win)):")
                for name, c in top:
                    lines.append(f"    {name:<28} {c:+.3f}")
        return "\n".join(lines)


def refresh_meta_model(trades_csv='paper_trades.csv', path=MODEL_PATH):
    """Nightly hook. Safe when the file is missing, empty, or has no
    quality_score column — it simply leaves the model inactive."""
    m = MetaModel(path)
    try:
        df = pd.read_csv(trades_csv)
    except (OSError, ValueError):
        return m
    return m.fit(df)
