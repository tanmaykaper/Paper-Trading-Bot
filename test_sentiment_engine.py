# test_sentiment_engine.py
#
# Assertion-based tests against CONSTRUCTED/synthetic headlines (this
# sandbox has no live network access to Yahoo Finance or Google News — same
# constraint alpha_engine.py's own tests are built under). These prove the
# MECHANICS are correct: lexicon scoring, negation scoping, sector-specific
# word-sense collisions, recency decay, small-sample shrinkage, the veto
# gate's two independent triggers, and cross-sectional ranking.
#
# A separate, OPT-IN live smoke test at the bottom hits the real network
# (only runs if you pass --live) — run that once on a machine or GitHub
# Actions runner with real internet access before trusting this in
# production, to confirm Yahoo/Google's actual response schema still
# matches what NewsFetcher expects (see NewsFetcher's docstring on why this
# is defensively multi-schema in the first place).
#
# Run: python3 test_sentiment_engine.py           (offline tests only)
#      python3 test_sentiment_engine.py --live     (+ live fetch smoke test)

import sys
from datetime import datetime, timezone, timedelta

from sentiment_engine import (
    HeadlineScorer, NewsFetcher, SymbolSentimentAggregator,
    CrossSectionalSentimentRanker, SentimentEngine,
)

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}  {detail}")


def h(title, days_ago=0, summary=''):
    now = datetime.now(timezone.utc)
    return {'title': title, 'summary': summary,
            'published': now - timedelta(days=days_ago),
            'source': 'Test', 'url': None, 'origin': 'synthetic'}


# ═════════════════════════════════════════════════════════════════════════════
print("\n[1] HeadlineScorer — basic lexicon scoring")
# ═════════════════════════════════════════════════════════════════════════════
scorer = HeadlineScorer()

r = scorer.score("Company reports record profit and strong revenue growth")
check("clear positive headline scores net_tone > 0", r['net_tone'] > 0, r)

r = scorer.score("Company reports heavy losses amid weak demand and slowdown")
check("clear negative headline scores net_tone < 0", r['net_tone'] < 0, r)

r = scorer.score("Company announces date of annual general meeting")
check("neutral factual headline scores exactly 0.0", r['net_tone'] == 0.0, r)
check("neutral headline has zero pos+neg hits", (r['pos'] + r['neg']) == 0, r)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[2] HeadlineScorer — negation handling")
# ═════════════════════════════════════════════════════════════════════════════

r = scorer.score("Results were not weak this quarter")
check("'not weak' flips negative word to positive contribution",
      r['pos'] >= 1 and r['neg'] == 0, r)

r = scorer.score("Growth was not strong enough to offset costs")
check("'not strong' flips positive word to negative contribution",
      r['neg'] >= 1, r)

# The confirmed, fixed bug: a negator in one clause must NOT reach across a
# comma into an unrelated clause and flip a word it doesn't actually modify.
r = scorer.score("Quarterly results were not impressive, margins weak")
check("negation does not cross a clause boundary (comma)",
      'not-weak' not in r['hits']['negative'] and 'weak' in r['hits']['negative'],
      r['hits'])

r = scorer.score("Company did not beat estimates this quarter")
check("phrase-level negation flips a positive phrase to negative",
      r['net_tone'] < 0, r)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[3] HeadlineScorer — sector-specific word-sense collisions")
# ═════════════════════════════════════════════════════════════════════════════
# These are the exact kind of context-dependent words Loughran & McDonald's
# own methodology is built around avoiding — checked here specifically
# against sectors this bot actually trades (defense, metals).

r = scorer.score("Company wins defense tank contract worth Rs 500 crore")
check("'tank' (defense hardware) does not falsely trigger negative sentiment",
      r['neg'] == 0, r)

r = scorer.score("Steel scrap prices rise on strong demand")
check("'scrap' (commodity, metals sector) does not falsely trigger negative sentiment",
      r['neg'] == 0, r)

r = scorer.score("Promoter trims stake in routine portfolio move")
check("'trims' (routine stake rebalancing) does not falsely trigger negative sentiment",
      r['neg'] == 0, r)

r = scorer.score("Firm settles litigation, shares recover on relief rally")
check("'litigation' alone does not force a negative reading when the actual news is positive",
      r['net_tone'] > 0, r)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[4] HeadlineScorer — litigious/uncertainty categories are independent of tone")
# ═════════════════════════════════════════════════════════════════════════════

r = scorer.score("SEBI opens probe into accounting irregularities at company")
check("litigious-flagged headline correctly counts litigious hits", r['litigious'] > 0, r)
check("litigious-flagged headline is also negative in tone here", r['net_tone'] < 0, r)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[5] SymbolSentimentAggregator — recency decay")
# ═════════════════════════════════════════════════════════════════════════════
agg = SymbolSentimentAggregator(scorer)
now = datetime.now(timezone.utc)

fresh_bad = agg.score_symbol('TEST', [h("Company reports catastrophic losses", days_ago=0)], as_of=now)
stale_bad = agg.score_symbol('TEST', [h("Company reports catastrophic losses", days_ago=6)], as_of=now)
check("fresh negative headline scores more negative than an identical stale one",
      fresh_bad['net_tone'] < stale_bad['net_tone'],
      f"fresh={fresh_bad['net_tone']} stale={stale_bad['net_tone']}")
check("stale (6d) bad headline is heavily decayed toward neutral",
      stale_bad['net_tone'] > -0.3, stale_bad)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[6] SymbolSentimentAggregator — small-sample shrinkage toward neutral")
# ═════════════════════════════════════════════════════════════════════════════

one_headline = agg.score_symbol(
    'TEST', [h("Stock craters on catastrophic guidance cut", days_ago=0)], as_of=now)
many_headlines = agg.score_symbol('TEST', [
    h("Stock craters on catastrophic guidance cut", days_ago=0),
    h("Company slashes guidance, warns of steep decline", days_ago=0),
    h("Analysts downgrade stock after disastrous quarter", days_ago=0),
    h("Shares plunge as company reports heavy losses", days_ago=0),
], as_of=now)

check("a single dramatic headline is shrunk toward neutral (weaker than raw)",
      abs(one_headline['net_tone']) < abs(one_headline['raw_tone']),
      one_headline)
check("a corroborated cluster of similar headlines produces a stronger (higher-confidence) reading than one headline alone",
      many_headlines['confidence'] > one_headline['confidence'],
      f"one={one_headline['confidence']} many={many_headlines['confidence']}")
check("a corroborated cluster's net_tone is more negative than a single shrunk headline",
      many_headlines['net_tone'] < one_headline['net_tone'],
      f"one={one_headline['net_tone']} many={many_headlines['net_tone']}")


# ═════════════════════════════════════════════════════════════════════════════
print("\n[7] SymbolSentimentAggregator — graceful handling of no data")
# ═════════════════════════════════════════════════════════════════════════════

empty = agg.score_symbol('TEST', [], as_of=now)
check("empty headline list returns neutral reading, not an error", empty['net_tone'] == 0.0, empty)
check("empty headline list returns zero confidence", empty['confidence'] == 0.0, empty)
check("empty headline list is flagged with a reason", empty['reason'] == 'no_headlines', empty)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[8] CrossSectionalSentimentRanker — percentile ranking")
# ═════════════════════════════════════════════════════════════════════════════
ranker = CrossSectionalSentimentRanker()

readings = {
    'GOOD':    agg.score_symbol('GOOD', [h("Company posts record profit and strong growth", 0)] * 3, as_of=now),
    'BAD':     agg.score_symbol('BAD', [h("Company reports catastrophic losses and steep decline", 0)] * 3, as_of=now),
    'NEUTRAL': agg.score_symbol('NEUTRAL', [h("Company announces AGM date", 0)], as_of=now),
    'NODATA':  agg.score_symbol('NODATA', [], as_of=now),
}
ranks = ranker.rank_universe(readings)

check("best-tone symbol ranks highest", ranks['GOOD'] == max(ranks.values()), ranks)
check("worst-tone symbol ranks lowest", ranks['BAD'] == min(ranks.values()), ranks)
check("zero-confidence (no data) symbol is excluded from ranking, not defaulted to 50",
      'NODATA' not in ranks, ranks)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[9] SentimentEngine — hard veto gate")
# ═════════════════════════════════════════════════════════════════════════════
engine = SentimentEngine()

fraud_reading = engine.aggregator.score_symbol('TEST', [
    h("Regulator opens investigation into accounting irregularities", 0),
    h("SEBI probe widens as company faces fraud allegations", 0),
    h("Company shares fall amid lawsuit and litigation concerns", 1),
], as_of=now)
blocked, reason = engine.check_veto(fraud_reading)
check("fresh, corroborated fraud/litigation cluster triggers the veto", blocked, reason)

mild_reading = engine.aggregator.score_symbol('TEST', [
    h("Company margins dip slightly amid input cost pressure", 0),
], as_of=now)
blocked2, _ = engine.check_veto(mild_reading)
check("a single mild, non-litigious negative headline does NOT trigger the veto",
      not blocked2, mild_reading)

stale_fraud = engine.aggregator.score_symbol('TEST', [
    h("Regulator opens investigation into accounting irregularities", days_ago=10),
], as_of=now)
blocked3, _ = engine.check_veto(stale_fraud)
check("an old (10d) single litigious headline outside the recency window does NOT trigger the veto",
      not blocked3, stale_fraud)

single_litigious = engine.aggregator.score_symbol('TEST', [
    h("Company receives routine regulatory filing acknowledgement from SEBI", 0),
], as_of=now)
blocked4, _ = engine.check_veto(single_litigious)
check("a single litigious-adjacent headline alone (not a cluster) does NOT trigger the veto",
      not blocked4, single_litigious)

no_data_reading = engine.aggregator.score_symbol('TEST', [], as_of=now)
blocked5, _ = engine.check_veto(no_data_reading)
check("no news data never triggers the veto (no opinion, not a block)", not blocked5, no_data_reading)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[10] SentimentEngine — resilience (never raises on bad input)")
# ═════════════════════════════════════════════════════════════════════════════
try:
    r = agg.score_symbol('TEST', [{'title': None, 'summary': None, 'published': None}], as_of=now)
    check("malformed headline (None title/summary) does not raise", True)
except Exception as e:
    check("malformed headline (None title/summary) does not raise", False, str(e))

try:
    blocked, reason = engine.check_veto(agg._empty_reading('TEST', 'no_headlines'))
    check("check_veto on an empty reading does not raise", True)
except Exception as e:
    check("check_veto on an empty reading does not raise", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
print("\n[11] NewsFetcher — defensive parsing of both known yfinance schema shapes")
# ═════════════════════════════════════════════════════════════════════════════
fetcher = NewsFetcher()

flat_item = {
    'title': 'Company posts strong results', 'summary': 'Details here',
    'link': 'https://example.com/a', 'providerPublishTime': int(now.timestamp()),
    'publisher': 'Example Wire',
}
parsed_flat = fetcher._parse_yf_item(flat_item)
check("legacy flat yfinance schema parses correctly",
      parsed_flat is not None and parsed_flat['title'] == 'Company posts strong results',
      parsed_flat)

nested_item = {
    'content': {
        'title': 'Company posts strong results', 'summary': 'Details here',
        'pubDate': now.isoformat(),
        'provider': {'displayName': 'Example Wire'},
        'canonicalUrl': {'url': 'https://example.com/b'},
    }
}
parsed_nested = fetcher._parse_yf_item(nested_item)
check("newer nested yfinance schema (content.*) parses correctly",
      parsed_nested is not None and parsed_nested['title'] == 'Company posts strong results',
      parsed_nested)

garbage_item = {'unexpected': 'shape', 'no_title_field': True}
parsed_garbage = fetcher._parse_yf_item(garbage_item)
check("unrecognised item shape is skipped (returns None), not a crash",
      parsed_garbage is None, parsed_garbage)


# ═════════════════════════════════════════════════════════════════════════════
print("\n[12] NewsFetcher — dedup across sources")
# ═════════════════════════════════════════════════════════════════════════════
dup_items = [
    {'title': 'Company Posts Strong Q1 Results!', 'summary': '', 'published': now,
     'source': 'Yahoo', 'url': 'a', 'origin': 'yfinance'},
    {'title': 'company posts strong q1 results', 'summary': '', 'published': now,
     'source': 'Google', 'url': 'b', 'origin': 'google_news_rss'},
    {'title': 'A totally different headline', 'summary': '', 'published': now,
     'source': 'Google', 'url': 'c', 'origin': 'google_news_rss'},
]
deduped = fetcher._dedupe(dup_items)
check("near-identical titles across sources are deduped to one", len(deduped) == 2, deduped)
check("the first-seen (Yahoo) copy is kept on a duplicate", deduped[0]['source'] == 'Yahoo', deduped)


# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}\n{PASS} passed, {FAIL} failed\n{'='*70}")
if FAIL:
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# OPT-IN LIVE SMOKE TEST — only runs with --live, hits the real network.
# Not part of the pass/fail count above; this is a schema sanity-check to
# eyeball once (e.g. on GitHub Actions or a machine with real internet),
# not an automated assertion (real headlines are unpredictable by nature).
# ═════════════════════════════════════════════════════════════════════════════
if '--live' in sys.argv:
    print("\n[LIVE] Fetching real news for RELIANCE and TCS — eyeball-check the output below.\n")
    live_engine = SentimentEngine()
    for sym, name in [('RELIANCE', 'Reliance Industries'), ('TCS', 'Tata Consultancy Services')]:
        reading = live_engine.score_symbol(sym, company_name=name)
        print(f"{sym}: net_tone={reading['net_tone']} confidence={reading['confidence']} "
              f"n_headlines={reading['n_headlines']} reason={reading['reason']}")
        blocked, reason = live_engine.check_veto(reading)
        if blocked:
            print(f"  VETO: {reason}")
        print(f"  most negative headline seen: {reading['most_negative_headline']}\n")
