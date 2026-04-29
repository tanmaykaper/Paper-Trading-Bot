# fundamental_screener.py  ── v4  LARGECAP + MIDCAP DUAL-GATE
# ─────────────────────────────────────────────────────────────────────────────
# Largecap gate  — same high-risk thresholds as v3
# Midcap gate    — different hard limits reflecting midcap characteristics:
#     • Higher D/E tolerance (midcaps often carry more leverage for growth)
#     • Lower minimum ROE (growth-stage companies sacrifice near-term ROE)
#     • Higher PE ceiling (midcap growth names trade at premiums)
#     • Revenue CAGR is checked as an additional hard gate for midcaps
#       (a midcap with declining revenue and high leverage is a value trap)
# Soft scoring thresholds are also differentiated per cap type.
# ─────────────────────────────────────────────────────────────────────────────

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Largecap hard limits ──────────────────────────────────────────────────────
LC_HARD = {
    'max_de':  4.0,
    'min_roe': 3.0,    # %
    'max_pe': 100.0,
}

# ── Midcap hard limits ────────────────────────────────────────────────────────
# Midcaps can carry more leverage and temporarily lower ROE during expansion,
# but they MUST show revenue growth to justify the risk premium.
MC_HARD = {
    'max_de':       5.0,    # higher tolerance — growth capex is common
    'min_roe':      2.0,    # lower floor — some are in investment phase
    'max_pe':      120.0,   # growth midcaps legitimately trade at premiums
    'min_rev_cagr': 0.05,   # 5% revenue CAGR minimum — no shrinking businesses
}

# ── Largecap soft thresholds ──────────────────────────────────────────────────
LC_SOFT = {
    'good_pe_rel': 1.2,
    'fair_pe_rel': 1.6,
    'good_de':  0.6,
    'fair_de':  1.2,
    'good_roe': 15.0,
    'fair_roe':  8.0,
    'good_cr':  1.5,
    'fair_cr':  1.0,
    'good_cagr': 10.0,
    'fair_cagr':  5.0,
}
LC_MIN_SCORE = 1.5   # out of 5.0

# ── Midcap soft thresholds ────────────────────────────────────────────────────
# Midcaps are scored more generously on PE (growth premium acceptable),
# more strictly on revenue CAGR (must show growth to justify the risk).
MC_SOFT = {
    'good_pe_rel': 1.4,    # wider PE band — growth premium is normal
    'fair_pe_rel': 2.0,
    'good_de':  0.8,       # slightly more leverage tolerated
    'fair_de':  1.8,
    'good_roe': 12.0,      # lower bar for good ROE
    'fair_roe':  6.0,
    'good_cr':  1.3,
    'fair_cr':  0.9,
    'good_cagr': 15.0,     # midcaps need stronger growth to earn the score
    'fair_cagr':  8.0,
}
MC_MIN_SCORE = 1.5


class FundamentalScreener:
    def __init__(self):
        logger.info("✓ FundamentalScreener v4 — LARGECAP + MIDCAP dual-gate")

    def check_fundamental_gate(self, fundamentals: dict, is_midcap: bool = False):
        """
        Returns (passed: bool, info: dict).

        Hard fail  → (False, {'reason': str})
        Score fail → (False, checks_dict)
        Pass       → (True,  checks_dict)
        """
        pe        = float(fundamentals.get('pe_ratio',      25))
        sector_pe = float(fundamentals.get('sector_avg_pe', 25))
        de        = float(fundamentals.get('debt_to_equity', 1.0))
        roe_raw   = float(fundamentals.get('roe_5yr',        0.15))
        roe       = roe_raw * 100 if roe_raw <= 1.0 else roe_raw
        cagr_raw  = float(fundamentals.get('revenue_cagr',  0.10))
        cagr      = cagr_raw * 100 if cagr_raw <= 1.0 else cagr_raw
        cr        = float(fundamentals.get('current_ratio',  1.3))

        if is_midcap:
            return self._midcap_gate(pe, sector_pe, de, roe, cagr, cr, cagr_raw)
        else:
            return self._largecap_gate(pe, sector_pe, de, roe, cagr, cr)

    # ── Largecap gate ─────────────────────────────────────────────────────────
    def _largecap_gate(self, pe, sector_pe, de, roe, cagr, cr):
        if de  > LC_HARD['max_de']:
            return False, {'reason': f'D/E={de:.1f} > {LC_HARD["max_de"]}'}
        if roe < LC_HARD['min_roe']:
            return False, {'reason': f'ROE={roe:.1f}% < {LC_HARD["min_roe"]}%'}
        if pe  > LC_HARD['max_pe']:
            return False, {'reason': f'P/E={pe:.0f} > {LC_HARD["max_pe"]} (bubble)'}

        S = LC_SOFT
        checks = {
            'pe_check':   1.0 if pe   < sector_pe * S['good_pe_rel'] else 0.5 if pe   < sector_pe * S['fair_pe_rel'] else 0.0,
            'de_check':   1.0 if de   < S['good_de']   else 0.5 if de   < S['fair_de']   else 0.0,
            'roe_check':  1.0 if roe  >= S['good_roe']  else 0.5 if roe  >= S['fair_roe']  else 0.0,
            'cr_check':   1.0 if cr   >= S['good_cr']   else 0.5 if cr   >= S['fair_cr']   else 0.0,
            'cagr_check': 1.0 if cagr >= S['good_cagr'] else 0.5 if cagr >= S['fair_cagr'] else 0.0,
        }
        return sum(checks.values()) >= LC_MIN_SCORE, checks

    # ── Midcap gate ───────────────────────────────────────────────────────────
    def _midcap_gate(self, pe, sector_pe, de, roe, cagr, cr, cagr_raw):
        if de   > MC_HARD['max_de']:
            return False, {'reason': f'D/E={de:.1f} > {MC_HARD["max_de"]} (too leveraged for midcap)'}
        if roe  < MC_HARD['min_roe']:
            return False, {'reason': f'ROE={roe:.1f}% < {MC_HARD["min_roe"]}%'}
        if pe   > MC_HARD['max_pe']:
            return False, {'reason': f'P/E={pe:.0f} > {MC_HARD["max_pe"]} (midcap bubble)'}
        # Midcap-specific: revenue shrinkage = hard fail (value trap risk)
        if cagr_raw < MC_HARD['min_rev_cagr']:
            return False, {'reason': f'Rev CAGR={cagr:.1f}% < {MC_HARD["min_rev_cagr"]*100:.0f}% (shrinking)'}

        S = MC_SOFT
        checks = {
            'pe_check':   1.0 if pe   < sector_pe * S['good_pe_rel'] else 0.5 if pe   < sector_pe * S['fair_pe_rel'] else 0.0,
            'de_check':   1.0 if de   < S['good_de']   else 0.5 if de   < S['fair_de']   else 0.0,
            'roe_check':  1.0 if roe  >= S['good_roe']  else 0.5 if roe  >= S['fair_roe']  else 0.0,
            'cr_check':   1.0 if cr   >= S['good_cr']   else 0.5 if cr   >= S['fair_cr']   else 0.0,
            'cagr_check': 1.0 if cagr >= S['good_cagr'] else 0.5 if cagr >= S['fair_cagr'] else 0.0,
        }
        return sum(checks.values()) >= MC_MIN_SCORE, checks

    def get_check_summary(self, symbol, fundamentals, checks, is_midcap=False):
        lines = [f"\n  Fundamentals — {symbol} ({'MIDCAP' if is_midcap else 'LARGECAP'})"]
        if isinstance(checks, dict) and 'reason' in checks:
            lines.append(f"    ✗ HARD FAIL: {checks['reason']}")
        else:
            for k, v in checks.items():
                icon = '✓' if v >= 1.0 else ('~' if v >= 0.5 else '✗')
                lines.append(f"    {icon} {k}: {v}")
            score    = sum(checks.values()) if isinstance(checks, dict) else 0
            min_s    = MC_MIN_SCORE if is_midcap else LC_MIN_SCORE
            lines.append(f"    Score: {score:.1f}/5.0  → {'PASS' if score >= min_s else 'FAIL'}")
        return '\n'.join(lines)
