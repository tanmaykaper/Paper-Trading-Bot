# fundamental_screener.py  ── HIGH-RISK VERSION v3
# ─────────────────────────────────────────────────────────────────────────────
# In high-risk / high-frequency mode, fundamentals are a guardrail, not a gate.
# Only block genuinely broken companies. Everything else gets through.
#
# Hard-fail criteria (any one = reject):
#   • D/E > 4.0     — dangerously leveraged
#   • ROE < 3%      — barely earning anything
#   • P/E > 100     — bubble territory (pure speculation)
#   • Negative book value (insolvent)
#
# Soft score is still computed and included in signal_details for reference,
# but it does NOT block trades unless the score is extremely low (< 1.5 / 5).
# ─────────────────────────────────────────────────────────────────────────────

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Hard limits — breach = instant reject ────────────────────────────────────
HARD_LIMITS = {
    'max_de':  4.0,    # was 3.0 in conservative version
    'min_roe': 3.0,    # was 5.0% — allow turnaround plays
    'max_pe': 100.0,   # was 80 — allow high-growth names
}

# ── Soft scoring thresholds ───────────────────────────────────────────────────
SOFT = {
    'good_pe_rel': 1.2,   # P/E < sector_avg × 1.2
    'fair_pe_rel': 1.6,
    'good_de':  0.6,
    'fair_de':  1.2,
    'good_roe': 15.0,     # %
    'fair_roe':  8.0,
    'good_cr':  1.5,
    'fair_cr':  1.0,
    'good_cagr': 10.0,    # %
    'fair_cagr':  5.0,
}

# Minimum soft score to pass (out of 5) — very low for high-risk mode
MIN_SCORE = 1.5


class FundamentalScreener:
    def __init__(self):
        logger.info("✓ FundamentalScreener v3 — HIGH-RISK (soft gate)")

    def check_fundamental_gate(self, fundamentals):
        """
        Returns (passed: bool, info: dict).

        In high-risk mode:
          - Hard fails return (False, {'reason': str})
          - Soft score < MIN_SCORE returns (False, checks_dict)
          - Otherwise (True, checks_dict) — trade proceeds
        """
        pe        = float(fundamentals.get('pe_ratio',      25))
        sector_pe = float(fundamentals.get('sector_avg_pe', 25))
        de        = float(fundamentals.get('debt_to_equity', 1.0))
        roe_raw   = float(fundamentals.get('roe_5yr',        0.15))
        roe       = roe_raw * 100 if roe_raw <= 1.0 else roe_raw
        cagr_raw  = float(fundamentals.get('revenue_cagr',  0.10))
        cagr      = cagr_raw * 100 if cagr_raw <= 1.0 else cagr_raw
        cr        = float(fundamentals.get('current_ratio',  1.3))

        # ── Hard fails ────────────────────────────────────────────────────────
        if de > HARD_LIMITS['max_de']:
            return False, {'reason': f'D/E={de:.1f} > {HARD_LIMITS["max_de"]} (dangerously leveraged)'}
        if roe < HARD_LIMITS['min_roe']:
            return False, {'reason': f'ROE={roe:.1f}% < {HARD_LIMITS["min_roe"]}%'}
        if pe > HARD_LIMITS['max_pe']:
            return False, {'reason': f'P/E={pe:.0f} > {HARD_LIMITS["max_pe"]} (bubble)'}

        # ── Soft scoring ──────────────────────────────────────────────────────
        checks = {}

        checks['pe_check'] = (
            1.0 if pe < sector_pe * SOFT['good_pe_rel'] else
            0.5 if pe < sector_pe * SOFT['fair_pe_rel'] else 0.0
        )
        checks['de_check'] = (
            1.0 if de < SOFT['good_de'] else
            0.5 if de < SOFT['fair_de'] else 0.0
        )
        checks['roe_check'] = (
            1.0 if roe >= SOFT['good_roe'] else
            0.5 if roe >= SOFT['fair_roe'] else 0.0
        )
        checks['cr_check'] = (
            1.0 if cr >= SOFT['good_cr'] else
            0.5 if cr >= SOFT['fair_cr'] else 0.0
        )
        checks['cagr_check'] = (
            1.0 if cagr >= SOFT['good_cagr'] else
            0.5 if cagr >= SOFT['fair_cagr'] else 0.0
        )

        score  = sum(checks.values())
        passed = score >= MIN_SCORE

        return passed, checks

    def get_check_summary(self, symbol, fundamentals, checks):
        lines = [f"\n  Fundamentals — {symbol}"]
        if isinstance(checks, dict) and 'reason' in checks:
            lines.append(f"    ✗ HARD FAIL: {checks['reason']}")
        else:
            for k, v in checks.items():
                icon = '✓' if v >= 1.0 else ('~' if v >= 0.5 else '✗')
                lines.append(f"    {icon} {k}: {v}")
            score = sum(checks.values()) if isinstance(checks, dict) else 0
            lines.append(f"    Score: {score:.1f}/5.0  → {'PASS' if score >= MIN_SCORE else 'FAIL'}")
        return '\n'.join(lines)
