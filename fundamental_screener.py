# fundamental_screener.py  ── v4  (MEASURED DATA, CROSS-SECTIONAL COMPARISON)
# ═════════════════════════════════════════════════════════════════════════════
# v3's logic was sound and was scoring nothing. data_fetcher_free v1 parsed
# `<td>` cells that screener.in does not render, so every symbol arrived here
# carrying _default_fundamentals() — P/E 25, D/E 0.70, ROE 18%, current ratio
# 1.3 — and this module has been evaluating the same synthetic company, once
# per stock, every day. It passed everything, because the defaults are
# deliberately healthy. Now that the fetcher returns real numbers, three
# defects in v3 become live and have to be fixed before the gate does damage:
#
# 1. LOSS-MAKING COMPANIES PASSED THE P/E GATE.
#    `if pe > max_pe: reject` catches the bubble end and nothing else. A
#    company losing money has a NEGATIVE P/E, comfortably below 100, so it
#    sailed through the one ratio meant to price earnings. On real data that is
#    not a rounding error — it is the most common way a genuinely broken
#    balance sheet enters a momentum universe.
#
# 2. SECTOR P/E WAS THE CONSTANT 25.
#    pe_check compares against sector_avg_pe, which _default_fundamentals sets
#    to 25 and nothing ever overwrote. An IT name on 30x was "expensive" and a
#    PSU bank on 9x "cheap" against the same yardstick. v4 computes each
#    sector's median P/E FROM THE UNIVERSE BEING SCANNED, so the comparison is
#    against the peer group. Median rather than mean: one 140x outlier drags a
#    mean far enough to make its whole sector look reasonably priced.
#
# 3. A FAILED SCRAPE LOOKED IDENTICAL TO A HEALTHY COMPANY.
#    Both produced the defaults and both passed. v4 reads the
#    `fundamentals_measured` flag the fetcher now sets and treats unmeasured
#    data as an absence of evidence. It does not reject on it — that would
#    delete any symbol screener.in happens to rate-limit — but it pins the
#    score to the neutral band and reports `measured: False`, so the caller can
#    distinguish "checked and fine" from "never checked".
#
# The high-risk posture from v3 is retained: fundamentals are a guardrail
# against broken companies, not a value screen. Only genuinely damaged balance
# sheets are rejected; everything else is scored and handed to the technical
# layer, which is where this system's edge is meant to come from.
# ═════════════════════════════════════════════════════════════════════════════

import logging
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Hard limits — breach = instant reject ────────────────────────────────────
HARD_LIMITS = {
    'max_de':      4.0,     # dangerously leveraged
    'min_roe':     3.0,     # % — barely earning anything
    'max_pe':    100.0,     # bubble territory
    'min_pe':      0.0,     # NEW: negative P/E means the company is losing
                            # money. v3 had no floor, so every loss-maker
                            # passed the ceiling test by sitting under it.
    'min_market_cap_cr': 300.0,   # ₹300cr — below this, reported fundamentals
                                  # are thin, and the scrip is usually too
                                  # illiquid to have cleared signal_generator's
                                  # turnover floor anyway
}

# ── Soft scoring thresholds ──────────────────────────────────────────────────
SOFT = {
    'good_pe_rel': 1.2,     # P/E below sector median x 1.2
    'fair_pe_rel': 1.6,
    'good_de':  0.6,   'fair_de':   1.2,
    'good_roe': 15.0,  'fair_roe':  8.0,
    'good_cr':  1.5,   'fair_cr':   1.0,
    'good_cagr': 10.0, 'fair_cagr': 5.0,
}

MIN_SCORE = 1.5             # out of 5 — very low, by design
UNMEASURED_SCORE = 2.5      # neutral band when nothing could be read


class FundamentalScreener:

    def __init__(self):
        self.sector_pe = {}          # sector -> median P/E, from the live universe
        self.universe_pe = None      # fallback when a sector has too few members
        logger.info("✓ FundamentalScreener v4 — measured data, cross-sectional P/E")

    # ─────────────────────────────────────────────────────────────────────────
    def calibrate_sector_pe(self, universe_fundamentals, sector_map=None, min_members=4):
        """
        Build the sector P/E medians this run compares against. Call once per
        scan, before screening, with {symbol: fundamentals}.

        Only MEASURED, positive, non-absurd P/Es contribute. Feeding the
        defaults back in would reconstruct the constant 25 this replaces, and a
        loss-maker's negative P/E would drag a sector median downward while
        saying nothing about what the sector is worth.

        Sectors with fewer than `min_members` usable readings fall back to the
        universe median — a two-stock "sector" median is a coin flip presented
        as a benchmark.
        """
        sector_map = sector_map or {}
        buckets, everything = {}, []
        for symbol, f in (universe_fundamentals or {}).items():
            if not f or not f.get('fundamentals_measured'):
                continue
            pe = _num(f.get('pe_ratio'))
            if pe is None or pe <= 0 or pe > 300:
                continue
            everything.append(pe)
            buckets.setdefault(sector_map.get(symbol, 'UNKNOWN'), []).append(pe)

        self.universe_pe = statistics.median(everything) if everything else None
        self.sector_pe = {s: statistics.median(v) for s, v in buckets.items()
                          if len(v) >= min_members}

        if self.universe_pe is None:
            logger.warning("  Sector P/E: nothing measurable in the universe — "
                           "pe_check falls back to the supplied sector_avg_pe")
        else:
            logger.info(f"  Sector P/E medians from {len(everything)} measured symbols: "
                        f"{ {k: round(v, 1) for k, v in sorted(self.sector_pe.items())} } "
                        f"| universe {self.universe_pe:.1f}")
        return self.sector_pe

    def _benchmark_pe(self, fundamentals, sector=None):
        """Sector median, then universe median, then whatever the caller
        supplied, then 25. Each step down is a weaker comparison, used only
        because the stronger one was unavailable."""
        # Callers that cannot pass a sector positionally (signal_generator
        # invokes the gate without one) can instead stamp it onto the
        # fundamentals dict, which keeps every existing signature intact.
        sector = sector or fundamentals.get('sector')
        if sector and sector in self.sector_pe:
            return self.sector_pe[sector]
        if self.universe_pe:
            return self.universe_pe
        return _num(fundamentals.get('sector_avg_pe')) or 25.0

    # ─────────────────────────────────────────────────────────────────────────
    def check_fundamental_gate(self, fundamentals, sector=None):
        """
        Returns (passed: bool, info: dict).

        info carries 'reason' on a hard fail, otherwise per-check scores plus
        'score', 'measured' and 'benchmark_pe'. The signature and the hard-fail
        contract are unchanged, so signal_generator's existing call works
        untouched; `sector` is optional and sharpens the comparison when given.
        """
        fundamentals = fundamentals or {}
        measured = bool(fundamentals.get('fundamentals_measured'))

        pe = _num(fundamentals.get('pe_ratio'), 25.0)
        de = _num(fundamentals.get('debt_to_equity'), 1.0)
        roe = _pct(fundamentals.get('roe_5yr'), 0.15)
        cagr = _pct(fundamentals.get('revenue_cagr'), 0.10)
        cr = _num(fundamentals.get('current_ratio'), 1.3)
        mcap_cr = _market_cap_cr(fundamentals.get('market_cap'))
        bench_pe = self._benchmark_pe(fundamentals, sector)

        # ── Hard fails ────────────────────────────────────────────────────────
        # Applied only to data actually read. Rejecting on a default value would
        # mean rejecting a symbol for a scraping failure, which is a fact about
        # the network rather than about the company.
        if measured:
            if de > HARD_LIMITS['max_de']:
                return False, {'reason': f'D/E={de:.1f} > {HARD_LIMITS["max_de"]} (dangerously leveraged)',
                               'measured': True}
            if pe <= HARD_LIMITS['min_pe']:
                return False, {'reason': f'P/E={pe:.1f} — loss-making', 'measured': True}
            if roe < HARD_LIMITS['min_roe']:
                return False, {'reason': f'ROE={roe:.1f}% < {HARD_LIMITS["min_roe"]}%', 'measured': True}
            if pe > HARD_LIMITS['max_pe']:
                return False, {'reason': f'P/E={pe:.0f} > {HARD_LIMITS["max_pe"]} (bubble)',
                               'measured': True}
            if mcap_cr is not None and mcap_cr < HARD_LIMITS['min_market_cap_cr']:
                return False, {'reason': f'Market cap ₹{mcap_cr:,.0f}cr below the '
                                         f'₹{HARD_LIMITS["min_market_cap_cr"]:,.0f}cr floor',
                               'measured': True}

        # ── Soft scoring ──────────────────────────────────────────────────────
        checks = {
            'pe_check': (1.0 if pe < bench_pe * SOFT['good_pe_rel'] else
                         0.5 if pe < bench_pe * SOFT['fair_pe_rel'] else 0.0),
            'de_check': (1.0 if de < SOFT['good_de'] else 0.5 if de < SOFT['fair_de'] else 0.0),
            'roe_check': (1.0 if roe >= SOFT['good_roe'] else 0.5 if roe >= SOFT['fair_roe'] else 0.0),
            'cr_check': (1.0 if cr >= SOFT['good_cr'] else 0.5 if cr >= SOFT['fair_cr'] else 0.0),
            'cagr_check': (1.0 if cagr >= SOFT['good_cagr'] else
                           0.5 if cagr >= SOFT['fair_cagr'] else 0.0),
        }
        score = sum(checks.values())

        if not measured:
            # Unmeasured data cannot earn a good score or a bad one. Pinning it
            # to the neutral band stops the defaults — which happen to describe
            # a flattering profile — from being read as evidence of quality.
            score = UNMEASURED_SCORE
            checks = {k: None for k in checks}

        checks.update({'score': round(score, 2), 'measured': measured,
                       'benchmark_pe': round(bench_pe, 1)})
        return score >= MIN_SCORE, checks

    # ─────────────────────────────────────────────────────────────────────────
    def get_check_summary(self, symbol, fundamentals, checks):
        lines = [f"\n  Fundamentals — {symbol}"]
        if isinstance(checks, dict) and 'reason' in checks:
            lines.append(f"    ✗ HARD FAIL: {checks['reason']}")
            return '\n'.join(lines)
        if not checks.get('measured', False):
            lines.append(f"    ~ not measured — neutral {UNMEASURED_SCORE:.1f}/5.0, "
                         f"passed without evidence")
            return '\n'.join(lines)
        for k, v in checks.items():
            if k in ('score', 'measured', 'benchmark_pe') or v is None:
                continue
            icon = '✓' if v >= 1.0 else ('~' if v >= 0.5 else '✗')
            lines.append(f"    {icon} {k}: {v}")
        lines.append(f"    Score: {checks['score']:.1f}/5.0  (sector P/E benchmark "
                     f"{checks['benchmark_pe']})  → "
                     f"{'PASS' if checks['score'] >= MIN_SCORE else 'FAIL'}")
        return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
def _num(value, default=None):
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value, default):
    """Accepts either convention: 0.215 and 21.5 both mean 21.5%."""
    v = _num(value, default)
    if v is None:
        return default * 100 if default <= 1.0 else default
    return v * 100.0 if abs(v) <= 1.5 else v


def _market_cap_cr(value):
    """Screener reports market cap in ₹ crore, the same unit as the legacy
    default (500000), so no conversion is applied. An implausibly large value
    is treated as unknown rather than silently passing a floor it never cleared."""
    v = _num(value)
    if v is None or v <= 0 or v > 5_000_000:
        return None
    return v
