# sentiment_engine.py
#
# ─────────────────────────────────────────────────────────────────────────────
# NEWS SENTIMENT ENGINE — Loughran-McDonald-inspired, free-data, regime-aware
# ─────────────────────────────────────────────────────────────────────────────
#
# STANDALONE MODULE, built and tested against synthetic/constructed headline
# data (see test_sentiment_engine.py), then wired into run_paper_trading.py
# and alpha_engine.py — same two-phase pattern this project already used for
# alpha_engine.py itself. This sandbox cannot reach finance.yahoo.com or
# news.google.com (no live network access here), so — exactly like
# alpha_engine.py's own documented caveat about no live NSE data access —
# the MECHANICS below are verified against known, deliberately-constructed
# inputs, not against real headlines fetched live. Run
# test_sentiment_engine.py's live smoke test once on a machine/GitHub Actions
# runner with real internet access to sanity-check the two fetchers before
# trusting this in production; see the bottom of this file.
#
# ── Why Loughran-McDonald instead of a generic sentiment lexicon ───────────
# Generic sentiment word lists (Harvard-IV/General Inquirer, or naive tools
# like TextBlob/VADER trained on product reviews and social media) are a
# documented poor fit for financial text. Loughran & McDonald's foundational
# finding (2011, Journal of Finance, "When Is a Liability Not a Liability?
# Textual Analysis, Dictionaries, and 10-Ks") was that roughly three-quarters
# of the words a generic dictionary tags "negative" are NOT actually negative
# in a financial/business context — ordinary words like "tax", "cost",
# "liability", "capital", "vice" (as in "vice president"), or "president"
# read as negative-flavoured in everyday English but are neutral, routine
# vocabulary in finance and business reporting. Their fix was a dictionary
# built FROM financial text (10-K filings) FOR financial text, with
# categories that map onto what actually moves financial outcomes:
# Negative, Positive, Uncertainty, Litigious, Constraining, and Modal
# strength — not just a blunt positive/negative split. This module builds
# category word lists in that same spirit and with many of the same actual
# category members (a representative, hand-curated subset — not a verbatim
# reproduction of Notre Dame's full ~4,000-word Master Dictionary file,
# which is a specific licensed research artifact this module doesn't
# reproduce), plus the dictionary's own documented negation refinement
# (checking a small word-window before a polarity word for a negator, so
# "not profitable" doesn't score as positive) applied symmetrically to both
# polarities.
#
# ── Where this actually helps profitability — and where it honestly doesn't ─
# Being straight about this, the same way alpha_engine.py was straight about
# "accuracy": a bag-of-words tone score computed from a handful of daily
# headlines is a WEAK, NOISY signal next to price/volume factors that have
# decades of published cross-sectional evidence behind them (see
# alpha_engine.py's own header). Two claims would be overfit nonsense:
#   ✗ "Positive headlines predict the stock will go up" — by the time
#     retail-readable headlines are unambiguously positive, price has
#     usually already moved; this is closer to lagging than leading.
#   ✗ "A sentiment score alone should size a bet" — a handful of headlines
#     is far too small and far too gameable (PR framing, clickbait framing)
#     a sample to carry that much weight on its own.
# Where a text signal genuinely adds something price/volume factors are
# structurally blind to:
#   ✓ TAIL-RISK DETECTION — fraud allegations, regulatory investigations,
#     accounting irregularities, and litigation are discrete, low-frequency
#     events that price only fully reflects AFTER the gap-down. A same-day
#     news read can flag "don't enter, or exit, this specific name" before
#     the technicals alone would ever show it — this is what the hard veto
#     gate below is for, and it's the higher-conviction use of this module.
#   ✓ A SMALL, VOLUME-AWARE CONVICTION NUDGE — used as a genuinely minority
#     factor (10% weight in alpha_engine.py's composite score, see
#     BASE_FACTOR_WEIGHTS there) alongside five much more established
#     price/volume factors, not in place of them.
# So this module is built to do the first job with real confidence and the
# second job cautiously — which is also why every design choice below
# (decay weighting, sample-size shrinkage, two independent free sources,
# cross-sectional percentile ranking rather than fixed thresholds) exists
# specifically to keep a genuinely noisy input from being trusted more than
# its actual information content justifies. This mirrors, and in a couple of
# places directly reuses, the same statistical devices alpha_engine.py's
# AdaptiveWeightCalibrator already established for exactly this reason.
#
# ── Architecture ─────────────────────────────────────────────────────────────
#   LM_* word lists         — the category lexicon (module level)
#   HeadlineScorer           — scores ONE headline/summary's text against the
#                               lexicon, with symmetric negation handling
#   NewsFetcher               — pulls recent headlines for a symbol from TWO
#                               independent free sources (Yahoo Finance via
#                               yfinance — already a project dependency — and
#                               Google News RSS), dedupes, filters by recency
#   SymbolSentimentAggregator — combines many scored headlines into one
#                               reading per symbol: recency-decay weighted,
#                               shrunk toward neutral for small samples
#   CrossSectionalSentimentRanker — converts raw tone into 0-100 percentile
#                               ranks against today's scan universe, same
#                               shape/philosophy as alpha_engine.py's
#                               CrossSectionalRanker
#   SentimentEngine            — orchestrates all of the above; exposes both
#                               the soft (0-100, feeds alpha_engine.py as an
#                               optional 6th factor) and hard (veto gate)
#                               outputs a caller needs
# ─────────────────────────────────────────────────────────────────────────────

import re
import time
import random
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. LM-INSPIRED LEXICON
# ═════════════════════════════════════════════════════════════════════════════
# Hand-curated, representative word lists per category. Deliberately kept as
# plain, inspectable Python sets — every word here is auditable, and the
# whole scoring path is traceable back to which words fired (see
# HeadlineScorer.score()'s 'hits' output) rather than being any kind of
# black box. Lowercase; matched against lowercased, tokenized text.

LM_NEGATIVE = {
    'abandon', 'abandoned', 'abandoning', 'abandonment', 'abnormal', 'abrupt',
    'abruptly', 'adverse', 'adversely', 'allegation', 'allegations', 'alleged',
    'allege', 'alleges', 'alleging', 'axe', 'axed', 'backlash', 'bankrupt',
    'bankruptcy', 'bearish', 'beaten', 'bleed', 'bleeding', 'breach',
    'breached', 'catastrophe', 'catastrophic', 'closure', 'closures',
    'collapse', 'collapsed', 'complaint', 'complaints', 'concern', 'concerns',
    'concerning', 'contraction', 'crackdown', 'crater', 'cratered', 'craters',
    'crisis', 'decline', 'declined', 'declines', 'declining', 'default',
    'defaulted', 'deficient', 'deficiency', 'delay', 'delayed', 'delays',
    'delinquent', 'delist', 'delisted', 'deteriorate', 'deteriorated',
    'deterioration', 'difficult', 'difficulty', 'disappointing',
    'disappointed', 'discontinue', 'discontinued', 'dive', 'dived', 'diving',
    'downgrade', 'downgraded', 'downside', 'downturn', 'drop', 'dropped',
    'erode', 'eroded', 'erosion', 'evasion', 'exodus', 'fail', 'failed',
    'failing', 'failure', 'fine', 'fined', 'fraud', 'fraudulent', 'halt',
    'halted', 'harm', 'harmful', 'headwind', 'headwinds', 'hurt', 'impair',
    'impaired', 'impairment', 'infringement', 'insolvency', 'insolvent',
    'investigate', 'investigated', 'investigation', 'jeopardy', 'lag',
    'lagging', 'layoff', 'layoffs', 'liquidation', 'loss', 'losses', 'lost',
    'meltdown', 'misconduct', 'mislead', 'misleading', 'mismanagement',
    'miss', 'missed', 'misses', 'mismanaged', 'negative', 'negatively',
    'nosedive', 'outage', 'penalty', 'penalties', 'plummet', 'plummeted',
    'plummets', 'plunge', 'plunged', 'plunges', 'probe', 'pullback', 'recall',
    'recalled', 'recession', 'reject', 'rejected', 'resign', 'resigned',
    'resignation', 'restatement', 'restructuring', 'rout', 'scandal',
    'scrapped', 'scrutiny', 'selloff', 'setback', 'shortfall', 'shrink',
    'shrinking', 'shutdown', 'sink', 'sank', 'sunk', 'slash', 'slashed',
    'slowdown', 'sluggish', 'slump', 'slumped', 'strain', 'strike',
    'struggling', 'suspend', 'suspended', 'terminate', 'terminated',
    'termination', 'tumble', 'tumbled', 'unable', 'underperform',
    'underperformed', 'underperforming', 'unfavorable', 'unfavourable',
    'volatile', 'volatility', 'warn', 'warns', 'warning', 'weak', 'weaken',
    'weakened', 'weakness', 'wipeout', 'wrongdoing', 'writedown',
    'write-off',
    # Deliberately EXCLUDED despite sounding negative-adjacent, because they
    # collide too often with routine, non-negative financial usage:
    #   'tank'/'tanked'  — "stock tanks" (negative) vs literal defense-sector
    #                      hardware ("wins tank contract") — a real collision
    #                      risk given this bot's own defense-sector exposure.
    #   'contract'/'contracted' — "signs a contract" (positive/neutral,
    #                      dominant sense) vs "economic contraction" (negative,
    #                      much rarer verb form) — kept only the unambiguous
    #                      noun form 'contraction'.
    #   'squeeze'/'squeezed' — "margin squeeze" (negative) vs "short squeeze"
    #                      (a sharp price INCREASE) — genuinely opposite
    #                      meanings depending on context a unigram can't see.
    #   'scrap' (bare)   — "scraps the deal" (negative) vs "scrap metal
    #                      prices" (routine commodity-desk vocabulary,
    #                      relevant to metals/mining names) — kept only the
    #                      past-tense 'scrapped', which rarely refers to
    #                      scrap metal.
    #   'trim'/'trimmed' — "trimmed guidance" (negative) vs a promoter
    #                      routinely "trimming stake" (often neutral
    #                      portfolio rebalancing, not distress). Handled via
    #                      the LM_PHRASES_NEGATIVE 'trimmed guidance' phrase
    #                      instead of the bare word.
}

LM_POSITIVE = {
    'achieve', 'achieved', 'achievement', 'achieves', 'advantage',
    'advantages', 'attain', 'attained', 'beat', 'beats', 'beneficial',
    'benefit', 'benefits', 'best', 'boom', 'booming', 'boost', 'boosted',
    'boosts', 'breakthrough', 'bullish', 'delight', 'delighted', 'effective',
    'efficient', 'efficiency', 'enhance', 'enhanced', 'enhancement',
    'excellent', 'exceed', 'exceeded', 'exceeds', 'exceptional', 'expand',
    'expanded', 'expanding', 'expansion', 'favorable', 'favourable', 'gain',
    'gained', 'gains', 'great', 'greater', 'grow', 'growing', 'growth',
    'grew', 'improve', 'improved', 'improvement', 'improving', 'innovation',
    'innovative', 'jump', 'jumped', 'leadership', 'lucrative', 'milestone',
    'momentum', 'opportunity', 'opportunities', 'outpace', 'outpaced',
    'outperform', 'outperformed', 'outperforming', 'positive', 'positively',
    'profit', 'profitable', 'profitability', 'progress', 'rally', 'rallied',
    'rebound', 'rebounded', 'record', 'recovery', 'resilient', 'rewarding',
    'rise', 'risen', 'rising', 'robust', 'soar', 'soared', 'spur',
    'stability', 'stable', 'strength', 'strengthen', 'strengthened',
    'strong', 'stronger', 'success', 'successful', 'successfully', 'surge',
    'surged', 'surges', 'sustainable', 'tailwind', 'tailwinds', 'thrive',
    'thriving', 'top', 'upbeat', 'upgrade', 'upgraded', 'upside', 'upturn',
    'win', 'winning', 'won',
    # Deliberately EXCLUDED: 'raise'/'raised' — "raised guidance" (positive)
    # vs a "capital raise" (often read cautiously/as dilutive, i.e. negative,
    # by momentum traders specifically) is too context-dependent for a
    # unigram. Handled via the 'raised guidance'/'raises guidance' phrases
    # instead. Similarly 'dividend' alone is not inherently positive news
    # (only a dividend HIKE is) — not included bare.
}

LM_UNCERTAINTY = {
    'ambiguity', 'ambiguous', 'anticipate', 'anticipated', 'approximate',
    'approximately', 'assume', 'assumption', 'contingency', 'contingent',
    'depend', 'depending', 'dependent', 'fluctuate', 'fluctuated',
    'fluctuation', 'indefinite', 'indefinitely', 'likely', 'possible',
    'possibly', 'predict', 'predicted', 'random', 'risk', 'risks', 'risky',
    'rumor', 'rumors', 'rumoured', 'speculate', 'speculation', 'speculative',
    'tentative', 'uncertain', 'uncertainty', 'unclear', 'unconfirmed',
    'unforeseen', 'unknown', 'unpredictable', 'unproven', 'variability',
    'volatile',
}

LM_LITIGIOUS = {
    'allegation', 'allegations', 'appeal', 'arbitration', 'attorney',
    'complaint', 'counsel', 'court', 'defendant', 'enforcement', 'fraud',
    'illegal', 'illegally', 'indictment', 'injunction', 'inquiry',
    'investigation', 'judgment', 'judgement', 'jurisdiction', 'lawsuit',
    'legislation', 'litigation', 'penalty', 'plaintiff', 'probe',
    'prosecute', 'prosecution', 'regulator', 'regulators', 'regulatory',
    'ruling', 'sebi', 'settlement', 'statute', 'sue', 'sued', 'suit',
    'summons', 'tribunal', 'verdict', 'violation', 'violations',
}

LM_CONSTRAINING = {
    'bound', 'commit', 'commitment', 'compel', 'compelled', 'compulsory',
    'condition', 'conditional', 'constrain', 'constrained', 'covenant',
    'mandate', 'mandated', 'mandatory', 'obligate', 'obligated',
    'obligation', 'prohibit', 'prohibited', 'require', 'required',
    'requirement', 'restrict', 'restricted', 'restriction',
}

# Negators for the negation-window check. Includes standalone words plus the
# common contracted forms a simple word-tokenizer will see as single tokens
# (e.g. "isn't" -> one token "isn't", not "is" + "not").
NEGATORS = {
    'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nor', 'never',
    'without', 'cannot', 'lack', 'lacking', "n't", "don't", "doesn't",
    "didn't", "isn't", "wasn't", "aren't", "weren't", "won't", "can't",
    "couldn't", "wouldn't", "shouldn't", "hasn't", "haven't", "hadn't",
}

NEGATION_WINDOW = 3   # tokens looked back from a polarity word for a negator

# Tokens that END a negation "scope" — a negator shouldn't be able to reach
# backward across one of these into an unrelated clause. Fixes a real,
# confirmed failure mode: "results were not impressive, margins weak" was
# incorrectly flipping 'weak' to positive, because 'not' sat within the
# 3-token window even though it modifies 'impressive' in an entirely
# different clause, not 'weak'. Splitting on these boundaries before running
# the negation check (see HeadlineScorer._tokenize_clauses) closes that gap.
_CLAUSE_BOUNDARY_RE = re.compile(r"[,.;:!?]|(?:\bbut\b)|(?:\bhowever\b)")

# Multi-word idioms common in financial headlines that either aren't
# reliably captured by single words (a unigram tokenizer breaks hyphenated/
# numeric phrases like "52-week low" apart) or would be too ambiguous as
# bare unigrams (see the exclusion notes above LM_POSITIVE/LM_NEGATIVE).
# Matched as case-insensitive substrings directly against the raw text
# (not the tokenized word list), each occurrence counted once.
LM_PHRASES_NEGATIVE = [
    'profit warning', 'guidance cut', 'cut guidance', 'cuts guidance',
    'lowered guidance', 'lowers guidance', 'slashed guidance',
    'trimmed guidance', 'trims guidance', 'earnings miss', 'missed estimates',
    'misses estimates', 'missed expectations', 'misses expectations',
    'missed street estimates', 'job cuts', 'credit rating downgrade',
    'rating downgrade', '52-week low', '52 week low', 'all-time low',
    'all time low', 'record low', 'going concern',
]
LM_PHRASES_POSITIVE = [
    'cost cutting', 'cost cuts', 'cost reduction', 'share buyback',
    'stock buyback', 'buyback programme', 'buyback program', 'record high',
    'all-time high', 'all time high', '52-week high', '52 week high',
    'beat estimates', 'beats estimates', 'beat expectations',
    'beats expectations', 'beat street estimates', 'raised guidance',
    'raises guidance', 'upgraded guidance', 'dividend hike', 'hikes dividend',
    'stake acquisition',
]


# ═════════════════════════════════════════════════════════════════════════════
# 2. HEADLINE SCORER
# ═════════════════════════════════════════════════════════════════════════════

class HeadlineScorer:
    """
    Scores ONE piece of text (a headline, or headline+summary) against the
    LM-inspired lexicon above. Deliberately simple tokenization (lowercase,
    strip punctuation to word/apostrophe tokens) rather than a heavier NLP
    dependency — headlines are short, mostly grammatical, single-sentence
    text where a lexicon-plus-negation approach is exactly what the original
    LM methodology itself uses, not a simplification of something better.
    """

    _TOKEN_RE = re.compile(r"[a-z']+")

    def _tokenize(self, text):
        if not text:
            return []
        return self._TOKEN_RE.findall(text.lower())

    def _tokenize_clauses(self, text):
        """Splits text into clauses at commas/periods/;:!?/but/however, then
        tokenizes each clause separately. Returns a list of token-lists. The
        negation-window check in score() looks backward only within the
        current clause's own tokens, so a negator in one clause can never
        flip a polarity word in a different clause."""
        if not text:
            return []
        clauses = _CLAUSE_BOUNDARY_RE.split(text.lower())
        return [self._TOKEN_RE.findall(c) for c in clauses if c.strip()]

    def score(self, text):
        """
        Returns a dict:
          n_words          — token count (denominator for density scores)
          pos, neg          — negation-adjusted counts (unigram + phrase)
          uncertainty, litigious, constraining — raw category counts
          net_tone          — (pos-neg)/(pos+neg), in [-1, 1]; 0.0 if no
                              polarity words fired at all (neutral, not
                              positive — a headline with zero sentiment
                              words is genuinely neutral, not "good news")
          hits              — {category: [matched tokens]} for auditability
        """
        clauses = self._tokenize_clauses(text)
        n = sum(len(c) for c in clauses)
        hits = {'positive': [], 'negative': [], 'uncertainty': [],
                'litigious': [], 'constraining': []}

        pos = neg = 0
        for tokens in clauses:
            for i, tok in enumerate(tokens):
                window = tokens[max(0, i - NEGATION_WINDOW):i]
                negated = any(w in NEGATORS for w in window)

                if tok in LM_POSITIVE:
                    if negated:
                        neg += 1
                        hits['negative'].append(f"not-{tok}")
                    else:
                        pos += 1
                        hits['positive'].append(tok)
                elif tok in LM_NEGATIVE:
                    if negated:
                        pos += 1
                        hits['positive'].append(f"not-{tok}")
                    else:
                        neg += 1
                        hits['negative'].append(tok)

                if tok in LM_UNCERTAINTY:
                    hits['uncertainty'].append(tok)
                if tok in LM_LITIGIOUS:
                    hits['litigious'].append(tok)
                if tok in LM_CONSTRAINING:
                    hits['constraining'].append(tok)

        # ── Phrase pass ──────────────────────────────────────────────────
        # Substring match on the raw lowercased text (not the tokenizer,
        # which would otherwise mangle "52-week low" into separate tokens).
        # A crude negation check looks at the ~15 characters immediately
        # before the phrase for a negator word — enough to catch "did not
        # beat estimates" without needing full clause-splitting for phrases.
        text_lower = (text or '').lower()
        for phrase in LM_PHRASES_NEGATIVE:
            idx = text_lower.find(phrase)
            if idx != -1:
                preceding = text_lower[max(0, idx - 15):idx]
                if any(f"{neg_w} " in preceding for neg_w in ('not', 'no', "n't")):
                    pos += 1
                    hits['positive'].append(f"not-{phrase}")
                else:
                    neg += 1
                    hits['negative'].append(phrase)
        for phrase in LM_PHRASES_POSITIVE:
            idx = text_lower.find(phrase)
            if idx != -1:
                preceding = text_lower[max(0, idx - 15):idx]
                if any(f"{neg_w} " in preceding for neg_w in ('not', 'no', "n't")):
                    neg += 1
                    hits['negative'].append(f"not-{phrase}")
                else:
                    pos += 1
                    hits['positive'].append(phrase)

        net_tone = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0

        return {
            'n_words': n,
            'pos': pos, 'neg': neg,
            'uncertainty': len(hits['uncertainty']),
            'litigious': len(hits['litigious']),
            'constraining': len(hits['constraining']),
            'net_tone': round(net_tone, 4),
            'hits': hits,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 3. NEWS FETCHER — two independent free sources, no API keys
# ═════════════════════════════════════════════════════════════════════════════

def _retry(fn, attempts=3, base_delay=1.2, what=""):
    """Same exponential-backoff retry shape as data_fetcher_free.py's _retry —
    reused here rather than reinvented, for one consistent resilience pattern
    project-wide."""
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                delay = base_delay * (2 ** i) + random.uniform(0, 0.4)
                logger.warning(f"⚠️ {what} attempt {i+1}/{attempts} failed ({e}) — retrying in {delay:.1f}s")
                time.sleep(delay)
    logger.error(f"✗ {what} failed after {attempts} attempts: {last_err}")
    return None


class NewsFetcher:
    """
    Pulls recent headlines for an NSE symbol from two independent, free,
    no-API-key sources:
      1. Yahoo Finance, via yfinance.Ticker.get_news() — already a project
         dependency, structured JSON, no scraping needed.
      2. Google News RSS search — a stable, long-standing public RSS
         endpoint, no key required. Parsed with the stdlib
         xml.etree.ElementTree (no new dependency added to requirements.txt).

    TWO sources on purpose, not one: this is the same "don't depend on a
    single fragile free source" lesson already learned the hard way
    elsewhere in this project (Sniper Outbound's multi-tier email
    verification waterfall, data_fetcher_free's bulk-then-individual-retry
    fallback). Yahoo's news API schema has changed shape before without
    notice — this fetcher defensively tries several plausible field-name
    layouts for exactly that reason (see _parse_yf_item) — and Google News
    coverage differs from Yahoo's, especially for smaller/midcap NSE names
    that get thinner Yahoo coverage. Either source failing independently
    still leaves usable data; both failing degrades to "no sentiment data
    for this symbol", which SentimentEngine treats as neutral/no-opinion,
    never as a penalty (same philosophy as alpha_engine.py's missing-factor
    handling).
    """

    def __init__(self, lookback_days=5, timeout=10):
        self.lookback_days = lookback_days
        self.timeout = timeout

    # ── Source 1: Yahoo Finance (yfinance) ──────────────────────────────────

    def _parse_yf_item(self, item):
        """
        Defensively extract (title, summary, published_dt, source, url) from
        one yfinance news item. Yahoo's underlying API has shipped at least
        two different shapes historically — a flat one (title/summary/link/
        providerPublishTime/publisher) and a newer nested one (content.title/
        content.summary/content.pubDate/content.provider.displayName/
        content.canonicalUrl.url). Tries both; returns None if neither shape
        yields a usable title, so a single unrecognised item is skipped
        rather than crashing the whole fetch.
        """
        try:
            content = item.get('content') if isinstance(item.get('content'), dict) else None

            if content:
                title   = content.get('title')
                summary = content.get('summary') or content.get('description') or ''
                pub_raw = content.get('pubDate') or content.get('displayTime')
                provider = content.get('provider') or {}
                source  = provider.get('displayName') if isinstance(provider, dict) else None
                url_obj = content.get('canonicalUrl') or content.get('clickThroughUrl') or {}
                url     = url_obj.get('url') if isinstance(url_obj, dict) else None
            else:
                title   = item.get('title')
                summary = item.get('summary') or ''
                pub_raw = item.get('providerPublishTime')
                source  = item.get('publisher')
                url     = item.get('link')

            if not title:
                return None

            published = self._parse_timestamp(pub_raw)
            return {
                'title': str(title), 'summary': str(summary or ''),
                'published': published, 'source': source or 'Yahoo Finance',
                'url': url, 'origin': 'yfinance',
            }
        except Exception:
            return None

    @staticmethod
    def _parse_timestamp(raw):
        """Accepts a unix epoch (int/float/numeric-string) OR an ISO-8601
        string; returns a tz-aware UTC datetime, or None if unparseable —
        an unparseable date just means the item can't be recency-filtered,
        not that the fetch should fail."""
        if raw is None:
            return None
        try:
            if isinstance(raw, (int, float)):
                return datetime.fromtimestamp(float(raw), tz=timezone.utc)
            if isinstance(raw, str):
                if raw.isdigit():
                    return datetime.fromtimestamp(float(raw), tz=timezone.utc)
                ts = pd.to_datetime(raw, utc=True, errors='coerce')
                if pd.isna(ts):
                    return None
                return ts.to_pydatetime()
        except Exception:
            return None
        return None

    def fetch_yfinance_news(self, symbol, max_items=15):
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not importable — skipping Yahoo Finance news source")
            return []

        yf_symbol = symbol if (symbol.startswith('^') or '.' in symbol) else f"{symbol}.NS"

        def _fetch():
            ticker = yf.Ticker(yf_symbol)
            raw = ticker.get_news(count=max_items, tab="news")
            if raw is None:
                raise ValueError("yfinance returned None for news")
            return raw

        raw_items = _retry(_fetch, attempts=2, what=f"yfinance news({symbol})")
        if not raw_items:
            return []

        parsed = [self._parse_yf_item(it) for it in raw_items]
        return [p for p in parsed if p is not None]

    # ── Source 2: Google News RSS ───────────────────────────────────────────

    _RSS_NS_STRIP = re.compile(r'<[^>]+>')

    def fetch_google_news_rss(self, query, max_items=15):
        """
        query: search string, e.g. 'RELIANCE NSE' or a company name. Uses
        the standard public Google News RSS search endpoint (no API key) —
        India-region/English results, matching this bot's NSE universe.
        """
        url = (
            "https://news.google.com/rss/search?q="
            f"{quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        def _fetch():
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.content

        content = _retry(_fetch, attempts=2, what=f"Google News RSS({query})")
        if not content:
            return []

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.warning(f"⚠️ Google News RSS: could not parse XML for '{query}': {e}")
            return []

        items = []
        for item_el in root.findall('.//item')[:max_items]:
            try:
                title = item_el.findtext('title') or ''
                if not title:
                    continue
                link = item_el.findtext('link')
                pub_date_raw = item_el.findtext('pubDate')
                published = self._parse_timestamp(pub_date_raw)
                source_el = item_el.find('source')
                source = source_el.text if source_el is not None else None
                description = item_el.findtext('description') or ''
                description = self._RSS_NS_STRIP.sub(' ', description).strip()

                items.append({
                    'title': title, 'summary': description,
                    'published': published, 'source': source or 'Google News',
                    'url': link, 'origin': 'google_news_rss',
                })
            except Exception:
                continue
        return items

    # ── Combined, deduped, recency-filtered ─────────────────────────────────

    @staticmethod
    def _dedupe(items):
        """Dedupe by a normalised title key — the same story frequently gets
        picked up by both sources, or syndicated across outlets with an
        identical headline. Keeps the first-seen copy (yfinance items are
        fetched first in fetch_all, so a Yahoo-sourced duplicate wins,
        matching the more structured/reliable source when both agree)."""
        seen = set()
        out = []
        for it in items:
            key = re.sub(r'[^a-z0-9]+', '', it['title'].lower())[:80]
            if key and key not in seen:
                seen.add(key)
                out.append(it)
        return out

    def fetch_all(self, symbol, company_name=None, max_items=20):
        """
        Merges both sources, dedupes, and filters to the last
        self.lookback_days days. Items with no parseable timestamp are KEPT
        (assumed recent — the RSS/API feeds themselves are near-real-time,
        so an item with an unparseable date is far more likely a parsing
        gap than genuinely stale news) but flagged via 'published': None,
        which SymbolSentimentAggregator treats as "no decay discount, but
        also not usable for the veto gate's recency requirement" (see
        below) — a conservative choice: undated items can nudge the soft
        score but can't, on their own, trigger a hard block.
        """
        yf_items  = self.fetch_yfinance_news(symbol, max_items=max_items)
        gnews_query = company_name or symbol
        rss_items = self.fetch_google_news_rss(f"{gnews_query} NSE", max_items=max_items)

        combined = self._dedupe(yf_items + rss_items)

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        filtered = [
            it for it in combined
            if it['published'] is None or it['published'] >= cutoff
        ]

        logger.info(
            f"  📰 {symbol}: {len(yf_items)} Yahoo + {len(rss_items)} Google News → "
            f"{len(combined)} unique → {len(filtered)} within {self.lookback_days}d window"
        )
        return filtered


# ═════════════════════════════════════════════════════════════════════════════
# 4. SYMBOL SENTIMENT AGGREGATOR
# ═════════════════════════════════════════════════════════════════════════════

class SymbolSentimentAggregator:
    """
    Combines many scored headlines for ONE symbol into a single, honestly-
    uncertain reading. Two statistical safeguards, both deliberate:

      1. RECENCY DECAY — today's headline should outweigh a five-day-old
         one, but not to zero; an ongoing story (e.g. a multi-day
         regulatory saga) shouldn't vanish from the read the moment it's
         not brand new. Exponential half-life weighting, same functional
         form as compound decay anywhere else in finance.

      2. SAMPLE-SIZE SHRINKAGE TOWARD NEUTRAL — a symbol with ONE headline
         (however strongly worded) should NOT produce as confident a
         reading as one with a dozen corroborating headlines. This reuses
         the exact same n/(n+K) empirical-Bayes shrinkage device
         alpha_engine.py's AdaptiveWeightCalibrator already established for
         pattern-weight recalibration — same statistical justification
         (don't let a small, noisy sample swing a decision as hard as a
         large, corroborated one), applied here to news volume instead of
         trade count. The shrinkage target is neutral (0.0), not a
         population mean — unlike trade expectancy, there's no principled
         non-zero prior for "what should headline tone average", so neutral
         is the correct uninformative prior.
    """

    HALF_LIFE_DAYS = 2.0     # today counts full weight; 2-day-old counts half
    SHRINKAGE_K     = 3.0    # decay-weighted headline count for 50% trust in raw tone

    def __init__(self, scorer=None):
        self.scorer = scorer or HeadlineScorer()

    @staticmethod
    def _ensure_aware(dt):
        """Defensive normalisation: NewsFetcher._parse_timestamp always
        returns tz-aware UTC datetimes (or None) in the paths this module
        controls, so this should be a no-op in practice — but a single
        naive datetime slipping through (e.g. a future upstream schema
        change) would otherwise raise TypeError on comparison/subtraction
        against an aware 'as_of' and take out sentiment scoring for the
        WHOLE batch of candidates that run, not just the one bad item.
        Cheap enough to always apply."""
        if dt is not None and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _decay_weight(self, published, as_of):
        published = self._ensure_aware(published)
        if published is None:
            return 0.5   # undated item: assumed recent-ish (see NewsFetcher.fetch_all)
                          # but discounted relative to a confirmed-fresh item
        age_days = max(0.0, (as_of - published).total_seconds() / 86400.0)
        return 0.5 ** (age_days / self.HALF_LIFE_DAYS)

    def score_symbol(self, symbol, headlines, as_of=None):
        """
        headlines: list of {title, summary, published, source, url, origin}
                   as produced by NewsFetcher.fetch_all().
        as_of: reference datetime for recency decay (defaults to now, UTC).

        Returns a SentimentReading dict — see fields below. Always returns a
        dict, never raises; an empty/failed headlines list produces a
        neutral, zero-confidence reading rather than an error, so a caller
        can always merge this into a factor_ranks-style structure without
        special-casing failures (same "missing/failed = no opinion, not a
        penalty" philosophy as the rest of this project's scoring layers).
        """
        as_of = as_of or datetime.now(timezone.utc)

        if not headlines:
            return self._empty_reading(symbol, reason='no_headlines')

        per_headline = []
        for h in headlines:
            text = f"{h.get('title', '')}. {h.get('summary', '')}".strip()
            s = self.scorer.score(text)
            weight = self._decay_weight(h.get('published'), as_of)
            per_headline.append({**s, 'weight': weight, 'headline': h})

        total_weight = sum(p['weight'] for p in per_headline)
        if total_weight <= 1e-9:
            return self._empty_reading(symbol, reason='zero_weight')

        raw_tone = sum(p['weight'] * p['net_tone'] for p in per_headline) / total_weight

        total_words = sum(p['n_words'] for p in per_headline) or 1
        uncertainty_density = sum(p['weight'] * p['uncertainty'] for p in per_headline) / total_weight
        litigious_density   = sum(p['weight'] * p['litigious']   for p in per_headline) / total_weight

        # ── Shrinkage toward neutral, gated on decay-weighted sample size ──
        effective_n   = total_weight
        shrink_factor = effective_n / (effective_n + self.SHRINKAGE_K)
        shrunk_tone   = shrink_factor * raw_tone   # prior mean is 0.0 (neutral)
        confidence    = round(shrink_factor, 3)

        # ── Litigious/negative article counts, for the hard-veto gate ─────
        # Counted on RECENT, CONFIRMED-DATED items only (published is not
        # None and within RECENCY_FOR_VETO_DAYS) — the veto gate exists to
        # catch fresh, corroborated bad news, not an undated item that might
        # actually be old, or a single stale story still inside the fetch
        # window. See SentimentEngine.check_veto() for how these are used.
        recent_cutoff = as_of - timedelta(days=3)
        litigious_articles = sum(
            1 for p in per_headline
            if p['litigious'] > 0
            and self._ensure_aware(p['headline'].get('published')) is not None
            and self._ensure_aware(p['headline']['published']) >= recent_cutoff
        )
        strong_negative_articles = sum(
            1 for p in per_headline
            if p['net_tone'] <= -0.5 and (p['pos'] + p['neg']) >= 2
            and self._ensure_aware(p['headline'].get('published')) is not None
            and self._ensure_aware(p['headline']['published']) >= recent_cutoff
        )

        most_negative = min(per_headline, key=lambda p: p['net_tone'])

        return {
            'symbol': symbol,
            'net_tone': round(shrunk_tone, 4),
            'raw_tone': round(raw_tone, 4),
            'uncertainty_score': round(uncertainty_density, 4),
            'litigious_score': round(litigious_density, 4),
            'n_headlines': len(headlines),
            'effective_n': round(effective_n, 2),
            'confidence': confidence,
            'litigious_articles_recent': litigious_articles,
            'strong_negative_articles_recent': strong_negative_articles,
            'most_negative_headline': most_negative['headline'].get('title'),
            'most_negative_tone': most_negative['net_tone'],
            'as_of': as_of,
            'reason': None,
        }

    def _empty_reading(self, symbol, reason):
        return {
            'symbol': symbol, 'net_tone': 0.0, 'raw_tone': 0.0,
            'uncertainty_score': 0.0, 'litigious_score': 0.0,
            'n_headlines': 0, 'effective_n': 0.0, 'confidence': 0.0,
            'litigious_articles_recent': 0, 'strong_negative_articles_recent': 0,
            'most_negative_headline': None, 'most_negative_tone': 0.0,
            'as_of': None, 'reason': reason,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 5. CROSS-SECTIONAL SENTIMENT RANKER
# ═════════════════════════════════════════════════════════════════════════════

class CrossSectionalSentimentRanker:
    """
    Converts raw net_tone across TODAY's scan universe into 0-100 percentile
    ranks — same design choice, and same reasoning, as
    alpha_engine.CrossSectionalRanker: a fixed absolute threshold ("tone >
    0.2 is good") doesn't mean the same thing on a day the whole market's
    news flow skews one way (e.g. a broad risk-off day where almost every
    headline reads a little negative) versus an ordinary day. Ranking
    relative to the current universe self-calibrates to whatever the
    prevailing news environment actually looks like today.
    """

    def rank_universe(self, readings_by_symbol):
        """
        readings_by_symbol: {symbol: SentimentReading} from
        SymbolSentimentAggregator.score_symbol().

        Returns {symbol: percentile_0_100}. Symbols with zero confidence
        (no usable headlines) are EXCLUDED from the ranking, not defaulted
        to 50 — consistent with "missing factor = no opinion" elsewhere;
        the caller (alpha_engine.py's score_symbol) already handles a
        missing 'sentiment' key correctly.
        """
        usable = {s: r['net_tone'] for s, r in readings_by_symbol.items()
                  if r.get('confidence', 0) > 0}
        if len(usable) < 2:
            return {}

        series = pd.Series(usable)
        pct = series.rank(pct=True) * 100
        return {s: round(float(v), 1) for s, v in pct.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 6. SENTIMENT ENGINE — orchestrator
# ═════════════════════════════════════════════════════════════════════════════

class SentimentEngine:
    """
    Ties NewsFetcher + SymbolSentimentAggregator + CrossSectionalSentimentRanker
    together, and adds the hard veto gate. Two outputs a caller uses
    differently:

      • rank_universe(...)'s 0-100 percentiles feed alpha_engine.py as the
        OPTIONAL 6th 'sentiment' factor (10% weight — see
        CompositeAlphaScore.BASE_FACTOR_WEIGHTS) — the SOFT use.

      • check_veto(reading) is an independent, absolute (not percentile)
        check for a hard block — the HARD use. This is deliberately NOT
        folded into the composite score: a genuinely severe, corroborated,
        fresh negative-news cluster should be able to block a trade
        regardless of how strong the OTHER five factors look, the same way
        signal_generator.py's own hard fundamental-fail gate
        (check_fundamental_gate) already works independently of its soft
        score. A percentile rank can't do this job on its own — "worst
        sentiment in today's universe" could still just mean "mildly
        negative on an otherwise-good news day", which is why the veto uses
        absolute thresholds instead.
    """

    # Two independent trigger conditions — either alone is sufficient. See
    # SymbolSentimentAggregator.score_symbol() for how the underlying counts
    # are computed (recent + dated items only).
    VETO_TONE_THRESHOLD     = -0.35   # shrunk net_tone at/below this...
    VETO_TONE_MIN_CONFIDENCE = 0.35   # ...AND confidence at/above this
    VETO_MIN_LITIGIOUS_ARTICLES = 2   # >=2 distinct recent litigious-flagged
                                       # headlines — a single stray "regulator"
                                       # mention (e.g. routine compliance
                                       # filing news) shouldn't alone block a
                                       # trade; a genuine cluster should
    VETO_MIN_NEGATIVE_ARTICLES  = 3   # or a pile-up of clearly-negative
                                       # headlines even without litigious terms

    def __init__(self, lookback_days=5, timeout=10):
        self.fetcher    = NewsFetcher(lookback_days=lookback_days, timeout=timeout)
        self.scorer     = HeadlineScorer()
        self.aggregator = SymbolSentimentAggregator(self.scorer)
        self.ranker     = CrossSectionalSentimentRanker()

    def score_symbol(self, symbol, company_name=None, as_of=None):
        """Fetch + score ONE symbol. Never raises — returns a neutral,
        zero-confidence reading on total fetch failure."""
        try:
            headlines = self.fetcher.fetch_all(symbol, company_name=company_name)
        except Exception as e:
            logger.error(f"  ✗ Sentiment fetch failed for {symbol}: {e}")
            headlines = []
        return self.aggregator.score_symbol(symbol, headlines, as_of=as_of)

    def score_universe(self, symbols, company_names=None, as_of=None):
        """
        symbols: list of symbols to fetch+score.
        company_names: optional {symbol: company_name} for better Google
                       News query relevance (falls back to the symbol
                       itself, which works fine for well-known tickers but
                       less well for a bare, ambiguous symbol string).

        Returns {symbol: SentimentReading}. Each symbol is independently
        try/excepted internally by score_symbol — one symbol's fetch
        failure never blocks the rest of the universe.
        """
        company_names = company_names or {}
        readings = {}
        for symbol in symbols:
            readings[symbol] = self.score_symbol(
                symbol, company_name=company_names.get(symbol), as_of=as_of,
            )
        return readings

    def rank_universe(self, readings_by_symbol):
        """Percentile ranks for the soft factor — see
        CrossSectionalSentimentRanker for the reasoning."""
        return self.ranker.rank_universe(readings_by_symbol)

    def check_veto(self, reading):
        """
        Returns (blocked: bool, reason: str|None).

        reading: one SentimentReading, e.g. from score_symbol()/
                 score_universe(). Absolute-threshold, not percentile-based
                 — see class docstring for why.
        """
        if reading.get('reason') is not None:
            return False, None   # no usable data — no opinion, not a block

        tone_trigger = (
            reading['net_tone'] <= self.VETO_TONE_THRESHOLD and
            reading['confidence'] >= self.VETO_TONE_MIN_CONFIDENCE
        )
        if tone_trigger:
            return True, (
                f"sentiment veto: net_tone={reading['net_tone']} "
                f"(confidence={reading['confidence']}) — "
                f"'{reading.get('most_negative_headline')}'"
            )

        litigious_trigger = (
            reading['litigious_articles_recent'] >= self.VETO_MIN_LITIGIOUS_ARTICLES
        )
        if litigious_trigger:
            return True, (
                f"sentiment veto: {reading['litigious_articles_recent']} recent "
                f"litigious/regulatory-flagged headlines — "
                f"'{reading.get('most_negative_headline')}'"
            )

        negative_pileup_trigger = (
            reading['strong_negative_articles_recent'] >= self.VETO_MIN_NEGATIVE_ARTICLES
        )
        if negative_pileup_trigger:
            return True, (
                f"sentiment veto: {reading['strong_negative_articles_recent']} recent "
                f"strongly-negative headlines — "
                f"'{reading.get('most_negative_headline')}'"
            )

        return False, None

    @staticmethod
    def tier(percentile):
        """Simple label for logging/display — mirrors alpha_engine's tier
        style but is NOT used for gating (percentiles never gate; only
        check_veto's absolute thresholds do)."""
        if percentile is None:
            return 'No data'
        if percentile >= 80:
            return 'Strongly positive'
        if percentile >= 60:
            return 'Positive'
        if percentile >= 40:
            return 'Neutral'
        if percentile >= 20:
            return 'Negative'
        return 'Strongly negative'


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Lightweight smoke test / usage demo against CONSTRUCTED headlines (no
    # network) — proves the MECHANICS (lexicon scoring, negation, decay,
    # shrinkage, veto triggers, ranking) are correct. See
    # test_sentiment_engine.py for the full, assertion-based test suite this
    # was actually validated against, and for the separate opt-in live-
    # network smoke test.
    now = datetime.now(timezone.utc)

    def h(title, days_ago=0, summary=''):
        return {'title': title, 'summary': summary,
                'published': now - timedelta(days=days_ago),
                'source': 'Test', 'url': None, 'origin': 'synthetic'}

    scenarios = {
        'STRONG_POSITIVE': [
            h("Company reports record profit and strong revenue growth", 0),
            h("Analysts upgrade stock after robust quarterly results", 1),
            h("Firm announces expansion, hiring surge as demand booms", 1),
        ],
        'FRAUD_CLUSTER': [
            h("Regulator opens investigation into accounting irregularities", 0),
            h("SEBI probe widens as company faces fraud allegations", 0),
            h("Company shares fall amid lawsuit and litigation concerns", 1),
        ],
        'NEGATED_POSITIVE': [
            h("Company says growth is not sustainable, warns of slowdown", 0),
            h("Quarterly results were not impressive, margins weak", 0),
        ],
        'SINGLE_DRAMATIC_HEADLINE': [
            h("Stock craters on catastrophic guidance cut", 0),
        ],
        'STALE_BAD_NEWS': [
            h("Company faced backlash over layoffs and restructuring", 6),
        ],
        'THIN_NEUTRAL_NEWS': [
            h("Company announces date of annual general meeting", 1),
        ],
    }

    engine = SentimentEngine()
    readings = {name: engine.aggregator.score_symbol(name, hl, as_of=now)
                for name, hl in scenarios.items()}
    ranks = engine.rank_universe(readings)

    print(f"\n{'Scenario':<26} {'tone':>7} {'conf':>6} {'pctl':>6}  veto")
    print('-' * 70)
    for name, r in readings.items():
        blocked, reason = engine.check_veto(r)
        pctl = ranks.get(name)
        veto_str = f"YES — {reason}" if blocked else "no"
        pctl_str = f"{pctl:.0f}" if pctl is not None else "  -"
        print(f"{name:<26} {r['net_tone']:>7.3f} {r['confidence']:>6.2f} {pctl_str:>6}  {veto_str}")
