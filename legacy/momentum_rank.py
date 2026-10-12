# momentum_rank.py  ── CROSS-SECTIONAL SELECTION  v1
# ═════════════════════════════════════════════════════════════════════════════
# The honest framing first: edge is not a setting. Every module built so far
# improves how a trade is sized, exited, costed or timed — none of them changes
# whether the SETUPS THEMSELVES predict anything. The Monte Carlo made that
# stark: at 0.00σ/day drift the account is down 31% of the time whatever the
# risk settings, and at 0.12σ/day it doubles 40% of the time. Everything else
# is second order.
#
# So this module attacks the only thing that moves that number: WHICH stocks
# get traded at all.
#
# ── What the current selection actually does ────────────────────────────────
# signal_generator evaluates each symbol in isolation — is THIS chart in an
# uptrend, is THIS ADX high enough, is THIS pattern present. Relative strength
# is one of six quality components at 18% weight, and it compares the stock to
# the index. Nothing anywhere compares candidates TO EACH OTHER. A tape where
# every name is trending produces a hundred passing signals, and the bot takes
# whichever three happen to clear first.
#
# That is leaving the most durable documented equity anomaly on the table.
# Cross-sectional momentum — buying the strongest names relative to the rest of
# the universe rather than any name that is merely rising — has held up across
# decades, markets and asset classes since Jegadeesh and Titman. It is also
# free to compute from the OHLCV this scanner already fetches every morning.
#
# ── The four components, and why each is here ───────────────────────────────
#
#   12-1 MOMENTUM. Twelve-month return with the most recent month EXCLUDED.
#     The skip is not decoration: the most recent month exhibits short-term
#     REVERSAL, so including it systematically dilutes the signal. This is the
#     single most-replicated formulation.
#
#   RESIDUAL MOMENTUM. The same return, with the market component regressed
#     out, scaled by the residual's own volatility. A high-beta stock in a
#     rising market has high raw momentum and no idiosyncratic strength at all
#     — it is an index proxy. Residual momentum isolates the part that is about
#     the company, and it has historically produced a higher information ratio
#     than raw momentum with markedly smaller crash risk.
#
#   INFORMATION DISCRETENESS. Whether the move arrived gradually or in a few
#     jumps, measured as the sign of the total return times the share of days
#     that were negative. Gradual information diffuses under the radar and
#     continues; discrete jumps are news that is already priced. Continuation
#     is materially stronger for gradual movers.
#
#   CONSISTENCY. The share of the last six months that were positive. A stock
#     up 60% on one gap and flat otherwise ranks identically to a steady
#     compounder on raw momentum, and behaves nothing like it going forward.
#
# ── How it is used ──────────────────────────────────────────────────────────
# As a GATE, not a tiebreak. A symbol outside the top tier of the universe is
# not traded no matter how good its individual chart looks, because "good chart
# in absolute terms" is exactly the criterion that was already producing a 36%
# win rate. The tilt in the allocator is secondary.
#
# This is a hypothesis with strong published support, not a guarantee. What it
# is NOT is a promise of 0.08σ/day — that number has to be measured on your own
# data by run_backtest.py, and the stratified output there is what will say
# whether this earned its place.
# ═════════════════════════════════════════════════════════════════════════════

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOOKBACK_LONG = 252      # ~12 months
LOOKBACK_MID = 126       # ~6 months
SKIP_RECENT = 21         # ~1 month, excluded for short-term reversal
MIN_BARS = LOOKBACK_MID + SKIP_RECENT + 10
MIN_UNIVERSE = 15        # below this, a percentile is not a measurement

WEIGHTS = {'mom_12_1': 0.30, 'residual': 0.35, 'discreteness': 0.15, 'consistency': 0.20}

# Only the top tier of the universe is eligible. 70 means the strongest 30%.
DEFAULT_GATE_PERCENTILE = 70.0


def _safe_ret(close, start, end):
    """Return between two backward offsets, guarding short history."""
    if len(close) <= start:
        return None
    a = float(close.iloc[-start])
    b = float(close.iloc[-end]) if end > 0 else float(close.iloc[-1])
    if a <= 0 or b <= 0:
        return None
    return float(np.log(b / a))


def compute_factors(df, index_df=None):
    """
    Raw (unranked) momentum factors for one symbol. Returns None when the
    history is too short — omitted from the ranking rather than imputed to the
    median, because an imputed rank would let a newly listed name inherit the
    universe's average strength without having earned it.
    """
    if df is None or len(df) < MIN_BARS or 'close' not in df:
        return None
    close = df['close'].astype(float)
    out = {}

    long_start = min(LOOKBACK_LONG, len(close) - 1)
    out['mom_12_1'] = _safe_ret(close, long_start, SKIP_RECENT)
    mid_start = min(LOOKBACK_MID, len(close) - 1)
    out['mom_6_1'] = _safe_ret(close, mid_start, SKIP_RECENT)
    if out['mom_12_1'] is None:
        return None

    # ── Residual momentum ────────────────────────────────────────────────────
    out['residual'] = None
    if index_df is not None and 'close' in index_df and len(index_df) >= MIN_BARS:
        n = min(len(close), len(index_df), LOOKBACK_LONG)
        r_s = np.diff(np.log(close.tail(n).to_numpy()))
        r_m = np.diff(np.log(index_df['close'].astype(float).tail(n).to_numpy()))
        k = min(len(r_s), len(r_m))
        r_s, r_m = r_s[-k:], r_m[-k:]
        if k > 40 and np.std(r_m) > 1e-9:
            beta = float(np.cov(r_s, r_m)[0, 1] / np.var(r_m))
            resid = r_s - beta * r_m
            window = resid[:-SKIP_RECENT] if k > SKIP_RECENT + 20 else resid
            sd = float(np.std(window))
            if sd > 1e-9:
                # Cumulative idiosyncratic return, scaled by its own noise — a
                # t-statistic, so a steady small alpha outranks a large noisy one.
                out['residual'] = float(window.sum() / (sd * np.sqrt(len(window))))
                out['beta'] = round(beta, 2)

    # ── Information discreteness ─────────────────────────────────────────────
    # sign(total return) x (share of negative days) - (share of positive days).
    # More NEGATIVE = more gradual = better continuation, so the sign is
    # flipped at the end to keep "higher is better" consistent across factors.
    window = np.diff(np.log(close.tail(LOOKBACK_MID + 1).to_numpy()))
    if len(window) > 20:
        pos = float((window > 0).mean())
        neg = float((window < 0).mean())
        total = float(window.sum())
        out['discreteness'] = -float(np.sign(total) * (neg - pos))
    else:
        out['discreteness'] = None

    # ── Consistency ──────────────────────────────────────────────────────────
    monthly = close.tail(LOOKBACK_MID).to_numpy()
    if len(monthly) >= 120:
        chunks = np.array_split(monthly, 6)
        ups = sum(1 for c in chunks if len(c) > 1 and c[-1] > c[0])
        out['consistency'] = ups / 6.0
    else:
        out['consistency'] = None
    return out


def rank_universe(universe_dfs, index_df=None, weights=None):
    """
    Cross-sectional percentile rank of every symbol with usable history.

    Each factor is ranked WITHIN the universe and then weighted, rather than
    z-scored on raw values. Momentum distributions are fat-tailed and skewed,
    so a z-score lets one runaway name dominate the composite; a rank cannot.

    Returns {symbol: {'percentile', 'composite', factors...}}.
    """
    weights = weights or WEIGHTS
    raw = {}
    for symbol, df in (universe_dfs or {}).items():
        f = compute_factors(df, index_df)
        if f is not None:
            raw[symbol] = f
    if len(raw) < MIN_UNIVERSE:
        logger.info(f"  Momentum rank: only {len(raw)} symbols with usable history "
                    f"(need {MIN_UNIVERSE}) — ranking withheld")
        return {}

    frame = pd.DataFrame(raw).T
    composite = pd.Series(0.0, index=frame.index)
    used_weight = pd.Series(0.0, index=frame.index)
    for factor, w in weights.items():
        if factor not in frame:
            continue
        col = pd.to_numeric(frame[factor], errors='coerce')
        if col.notna().sum() < MIN_UNIVERSE:
            continue
        pct = col.rank(pct=True)                       # NaN stays NaN
        composite = composite.add(pct.fillna(0.0) * w, fill_value=0.0)
        used_weight = used_weight.add(col.notna().astype(float) * w, fill_value=0.0)

    # Renormalise per symbol over the factors it actually has, so a name
    # missing residual momentum is not penalised as though it scored zero.
    composite = composite / used_weight.replace(0, np.nan)
    percentile = composite.rank(pct=True) * 100.0

    out = {}
    for symbol in frame.index:
        if pd.isna(composite.get(symbol)):
            continue
        entry = {k: (None if pd.isna(v) else round(float(v), 4))
                 for k, v in frame.loc[symbol].items() if k != 'beta'}
        entry.update({'composite': round(float(composite[symbol]), 4),
                      'percentile': round(float(percentile[symbol]), 1)})
        out[symbol] = entry
    top = sorted(out.items(), key=lambda kv: -kv[1]['percentile'])[:5]
    logger.info(f"  Momentum rank: {len(out)} symbols | leaders "
                f"{[(s, v['percentile']) for s, v in top]}")
    return out


def gate(ranks, symbol, threshold=DEFAULT_GATE_PERCENTILE):
    """
    (eligible, reason). A symbol with no rank is ELIGIBLE — an absent
    measurement must not become a silent exclusion, which would quietly shrink
    the universe to whatever happens to have long history.
    """
    entry = ranks.get(symbol) if ranks else None
    if not entry:
        return True, None
    if entry['percentile'] < threshold:
        return False, (f"momentum percentile {entry['percentile']:.0f} below the "
                       f"top {100 - threshold:.0f}% of the universe")
    return True, None
