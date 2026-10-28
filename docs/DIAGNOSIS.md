# Diagnosis — Oct 2026: why the book is down, and what to fix first

Written 28 Oct 2026, after 16 live paper sessions (6–28 Oct; intraday from 9 Oct). Sources:

- the bots' own ledgers (`state/*/trades.csv`, `equity.csv`, `state.json`);
- their GitHub Actions logs;
- a code audit;
- a real-data study written for this note: `nsebot/research/diagnosis.py`, run 40735738159. Every NSE-listed stock (1,779 ever eligible), Jul 2018 – 28 Oct 2026, Zerodha costs, all five rebalance timings.

Everything that compares strategy variants is post-hoc: it was chosen after the losses were seen. §4 says which conclusions are firm and which are only effect sizes.

---

## 1. The short answer

**₹1,39,551 from ₹1,50,000 (−6.97%).** The ₹10,449 loss has five sources:

| Source | ₹ | Share | What it is |
|---|---|---|---|
| Momentum: BHAGYANGR booked at −89% | −4,466 | 43% | **Accounting.** A demerger the paper ledger can't represent (§2.1) |
| Intraday: Zerodha charges | −2,437 | 23% | Costs larger than intraday's trading loss |
| Intraday: trading | −1,661 | 16% | A strategy with no measurable edge (§2.2) |
| Momentum: everything else | −2,802 | 27% | Real price moves in 11 small-cap holdings |
| Swing | +1,143 | — | 3 trades, 2 winners; 89% of its money idle (§2.3) |
| Cost of selling the open positions | −227 | 2% | |

Against the market from the 5 Oct close to the 28 Oct close (study §1):

| | Return |
|---|---|
| Momentum sleeve, as booked | −14.5% |
| **Momentum sleeve, demerger value kept** | **−4.6%** |
| Equal-weight hold of every eligible NSE stock (971 names, no costs) | −5.1% |
| Nifty Midcap 50 | −2.4% |
| Nifty 500 | −1.1% |
| Nifty 50 | +0.6% |

**The momentum strategy did not fail.** Once the demerger is counted properly, it did what its market did: small and mid caps fell while the Nifty 50 rose.

- **A −4.6% stretch is common for this strategy.** 19% of all 16-session stretches since 2018 were that bad or worse.
- **The −14.5% as booked happened in only 4% of stretches**, and it is mostly the accounting error.

**The real problems are elsewhere:**

1. A ledger that can't handle a demerger.
2. An intraday sleeve with no edge that pays more in charges than it loses trading.
3. A swing sleeve that leaves its money idle.
4. No alerts reaching you.

---

## 2. Findings

### 2.1 A demerger booked as an 89% loss — bug, certain

| Date | Event |
|---|---|
| 25 Sep | Bhagyanagar India's board fixes 8 Oct as the record date for demerging its copper business into Tieramet Ltd: 1 Tieramet share for every share held |
| 5 Oct | The first rebalance ranks BHAGYANGR #5 on 12-1 momentum (+318%). Part of that run was the market pricing the demerger |
| 6 Oct | Bought 12 @ ₹418.15 |
| 8 Oct | Ex-date. Yahoo's bar opens and closes about 90% lower: the copper business has left the price. No warning is raised |
| 12 Oct | The rebalance finds the ~₹45 shell "no longer eligible" (under the ₹50 price floor) |
| 13 Oct | Sold @ ₹47.82, net −₹4,465.85. The 12 Tieramet shares a real holder owns don't exist in the ledger |

**Why the guards missed it** (`nsebot/engine/momentum.py`):

- **A demerger is not a split.** `_rescale_for_splits` only recognises clean small-number ratios that Yahoo has back-adjusted. Yahoo doesn't adjust demergers.
- **The split anchor moves to the latest bar every run** (`_reanchor`). A split Yahoo adjusts a day or more late is therefore compared against an already-adjusted bar and is missed too.
- **Nothing alerts on a one-day collapse.** Under NSE's 2–20% circuit bands, a session that opens *and* closes 30%+ below the previous close is a corporate action or a data fault in all but a handful of F&O names, never an ordinary move.

**How common it is** (study §2): 44 such sessions across 41 stocks since 2017, and rising fast:

| Year | Sessions |
|---|---|
| 2023 | 3 |
| 2024 | 9 |
| 2025 | 16 |
| 2026 to date | 12 |

India's listed companies are demerging at a record pace. B4's 2018–26 backtest (at the rebalance timing checked) held none of the 44, so this was bad luck in week one, not a hidden drag in the research numbers. Rerunning B4 with value kept across every gap changes its CAGR by less than 2 points.

**Predicted outcome of the fix:**

- **Accounting accuracy.** About ₹5.0k comes back on the momentum sleeve: −14.5% becomes −4.6% for the sleeve, and −6.97% becomes about −3.6% for the whole book, once the Tieramet entitlement is credited.
- **No forced sales.** Future gaps get flagged instead of being sold as "ineligible" shells.
- **No effect on expected return.**

### 2.2 Intraday has no edge, and its costs exceed its trading loss — strategy, high confidence

**55 trades over 13 sessions:**

| | Value |
|---|---|
| Win rate after costs | 38% |
| Average winner | +0.97R |
| Average loser | −1.09R |
| Gross R per trade | **−0.11R** (t −0.75; 95% interval −0.42R to +0.19R) |
| Cost per trade | **0.19R**: 10.6 bps of notional against stops 0.3–1.0% wide |
| Net R per trade | **−0.30R** (95% interval −0.60R to 0.00R) |
| Stopped out within 10 minutes | 12 of 55 |
| Daily loss limit hit | 21 Oct, 25 minutes after the first entry |

- **Even the top of the 95% interval only breaks even after costs.** That matches the research before it went live: every ORB variant was net-negative after MIS costs (rounds 1–2).
- **The 0.5% risk floor keeps the bleed small but steady.** That means ₹250 at risk and ₹44k of notional per trade, at 4.2 trades a session.

**Predicted outcome of switching it off.**

- **Expected saving:** about **₹315 per session (about ₹1,500 a week)**, 0.2% of the whole book each session.
- **Remaining damage if left on:** the sleeve's 25% latch is ₹7.4k of further losses away, about 5 weeks at the current pace.

**Fix.** Stop placing intraday trades. If the data is still wanted, run the scan in shadow mode: log the signals with no capital and no costs. Reconsider only if 100+ shadow trades show gross R comfortably above the 0.19R cost. Tuning on these 55 trades isn't worth doing; 09:50–10:30 entries made money, but on 10 trades.

### 2.3 Swing earns less than its idle cash would — structural

- **Capital use.** Average invested: **11%** of the sleeve over 16 sessions.
- **Backtest.** The full-engine backtest, rerun today on current data (run 40734076709, Jun 2024 – Oct 2026), returned **+2.9% a year**, with a −19% max drawdown, over 188 trades.
- **Where it loses.** The 7-day time stop is the loss bucket: 40 trades, −₹20.6k.
- **Untouched data.** The S4b signal failed on 2018–20 (round 5).

A liquid fund pays more, with no drawdown.

**Predicted outcome.**

- **Idle cash in a liquid ETF** (LIQUIDBEES / LIQUIDCASE, about 6.5% a year): about +₹230 a month on the paper book, no new risk.
- **Retiring swing and moving its ₹50k to momentum:** changes that money's expectation from about +3% a year to momentum's (§3), with momentum's drawdowns. That is your call, not a fix.

### 2.4 Momentum: the design holds up — the stretched stocks are where the returns come from

The first ten picks had a median 12-1 return of +290% (CUPID +532%, STLTECH +480%), mostly small caps, and 5 of the 9 current holdings are in series BE (trade-for-trade surveillance). This looked like a design flaw. The textbook suggestions are:

- volatility-adjusting the score (Barroso & Santa-Clara 2015; NSE's own momentum indices rank this way);
- preferring smooth winners ("frog in the pan", Da, Gurun & Warachka 2014);
- avoiding extreme daily jumpers (the MAX effect, Bali, Cakici & Whitelaw 2011);
- 52-week-high proximity (George & Hwang 2004).

**On NSE data, most of them don't hold up** (study §3, 109 non-overlapping months):

1. **The most stretched stocks earn the most.** Forward one-month excess return rises steadily with 12-1 momentum. Stocks with a negative 12-1 return trail the average by 0.50% a month (t −6.7); those up 100–200% beat it by 0.87% (t 4.8); those up 200%+ beat it by 1.35% (t 3.6).
2. **Fewer than half of the stretched stocks beat the market.** Only 46% of the 200%+ group did: the average is carried by a few large winners.
3. **Raw 12-1 is the best stock picker tested.** Its top 10 beat the eligible average by **1.9% a month** on the untouched 2018–23 window (t 2.2) and 2.6% since 2023.
4. **Every alternative picks a worse 10 from B4's own top 30.** Volatility-adjusted, NSE-style composite, smoothness, 52-week high, low volatility and low max-gain all came out between −0.4% and −1.1% a month behind.

**One idea survives: don't buy names that jumped 15%+ in a single day in the last 63 sessions.** This is V4, a buy filter only; holdings still leave the normal way. Results across the five rebalance timings (study §4):

| Design | CAGR 2018–23 (untouched) | CAGR 2018–26 | CAGR Jun 2024–26 | Worst max DD | Sharpe | Timings beating B4 |
|---|---|---|---|---|---|---|
| **B4 (deployed)** | +30.5% [23.3 – 31.1] | +31.1% [26.5 – 35.1] | +24.8% [14.9 – 29.6] | −53.6% | 1.14 | — |
| V4, 15% jump filter | +30.0% [25.1 – 35.0] | **+36.7%** [32.7 – 39.8] | **+34.8%** [30.2 – 47.2] | −50.4% | 1.30 | 4 / 5 / 5 |
| V4 at 10% | +28.1% [24.8 – 31.6] | +34.4% [32.2 – 38.9] | +36.2% [25.5 – 38.7] | −53.7% | 1.27 | 3 / 4 / 4 |
| V4 at 20% | +30.1% [21.8 – 30.3] | +31.3% [25.2 – 34.2] | +25.1% [15.4 – 28.5] | −53.2% | 1.15 | 1 / 1 / 3 |
| V1 12-1 ÷ volatility | +23.7% | +30.8% | +32.7% | −49.8% | 1.19 | 1 / 2 / 4 |
| V2 NSE-style 6m+12m composite | +31.7% | +34.8% | +18.2% | −53.2% | 1.26 | 4 / 3 / 2 |
| V3 smooth winners only | +21.2% | +29.3% | +30.9% | −56.9% | 1.13 | 0 / 1 / 5 |
| V5 near the 52-week high | +31.0% | +31.7% | +20.2% | −54.0% | 1.15 | 3 / 3 / 2 |
| V6 monthly rebalance | +23.4% | +30.2% | +24.6% | **−63.9%** | 1.11 | 1 / 2 / 2 |
| V7 top 20 | +28.9% | +30.3% | +22.4% | **−48.4%** | 1.20 | 3 / 3 / 1 |
| V8 = V1 + V4 | +26.3% | +36.3% | +37.3% | −51.1% | 1.37 | 1 / 4 / 5 |

_Median across the five timings, [min – max]. All rows use prices adjusted for corporate-action gaps. "Timings beating B4" is untouched / full / recent._

**How far to trust V4:**

- **Where it helps.** It beat B4 at 4 of 5 timings on the untouched window, but its median there is level with B4's. Almost all of its gain is in 2023–26, the stretch this project already researched.
- **Threshold.** It still helps at 10% but fades at 20%. That fits the idea: a 20% day is usually a circuit-limit run, which the 15% cut already catches.
- **Conflict with §3.** Re-ranking the top 30 by smallest daily gain did *worse*, so this works only as a filter on extreme jumpers.
- **Selection.** It was the best of 11 designs tried after the fact.

Expected effect: somewhere between nothing and +5 points of CAGR, and about 3 points less drawdown. That justifies a **pre-registered shadow test**, not a switch. Run V4 as a second paper book alongside B4, decision rules written down first, verdict after 12 months.

**Don't change:**

- **The score.** Volatility-adjusted, NSE composite and smoothness all lose on the untouched window or the recent one.
- **The rebalance frequency.** Monthly has the worst drawdown in the table.
- **The size of the book, unless you want less risk.** Top 20 gives up a little return for a 5-point shallower drawdown. That is a risk preference, not an improvement.
- **The BE series.** The research universe always included it. Excluding it is untested and would likely cut the strongest names, since surveillance is triggered by exactly the price runs momentum buys.
- **The BZ series (non-compliant companies).** Excluding it is a reasonable risk rule, but it hasn't been tested.

### 2.5 Operations

| Finding | Effect | Fix |
|---|---|---|
| The `EMAIL_SENDER`, `EMAIL_PASSWORD` and `EMAIL_RECIPIENT` secrets are empty in every run log | No one was told about the 89% overnight drop, the daily loss limit, or anything else | Add the three secrets (docs/SCHEDULING.md, about 5 minutes) |
| 12 Oct: an intraday run dispatched at 08:26 IST hit GitHub's 6-hour cap at 14:27, before the 15:10 square-off. The follow-on run's state commit then failed on a rebase conflict in `state/intraday/state.json` | JSWSTEEL (₹1.26 lakh notional) was carried overnight in the ledger and closed next morning at −₹426. Zerodha would have squared it off at 15:20 | `commit_state.sh`: on a conflict inside the bot's own `state/<mode>/`, keep the newer local copy. Refuse to start an intraday session that can't reach 15:12 inside 6 hours |
| `_reanchor` moves the split anchor every run | A split Yahoo adjusts late is never detected | Keep the entry-date anchor as well (§2.1) |
| The 8 Oct momentum run marked BHAGYANGR at a pre-ex price; the gap only appeared on 9 Oct | Yahoo's NSE data can lag a corporate action by a session | Covered by the gap guard (§2.1) |

---

## 3. What owning B4 is actually like (study §5)

The goal set for this bot is "high accuracy, consistent, high growth". The evidence says those pull against each other. Here is what the only design with an out-of-sample record looks like over 2018–26, after Zerodha costs:

| Measure | Value |
|---|---|
| Round trips since Jul 2018 | 285 |
| Trades that made money | **53%** |
| Average winner / average loser | +37.9% / −14.7% (payoff 2.6) |
| Median holding | 50 sessions |
| Best 10% of trades | **All of the profit.** They sum to +3,953% of per-trade return; the other 90% together sum to −195% |
| 16-session stretches: 5th / 10th / median / 90th percentile | −13.1% / −9.0% / +2.3% / +13.1% |
| 252-session windows that lost money | 28% (9% lost 20%+) |
| Share of sessions spent 20%+ below the previous peak | **48%** (25% spent 30%+ below) |
| Worst drawdown across the five timings | −53.6% |

The growth comes from holding a few names that go up several times over, not from being right often:

- **Cutting losers early cuts the winners too.** Momentum only works by letting the few big winners run, so a tighter stop or a smarter entry filter would remove the trades that pay.
- **"High accuracy" can't be engineered in.** The model is right about half the time and makes its money on the size of the wins. §2.4's "smarter" filters all made it worse.
- **"Consistent" isn't on offer either.** Half the time this book is 20%+ under its high, and more than one year in four loses money.
- **Three weeks says nothing.** The first meaningful read on the live sleeve is about 12 months out.

**For real money, tax matters.** With a 50-session median hold, nearly every gain is short-term, taxed at 20.8% when realised. A smallcap fund investor pays 12.5% long-term gains tax, and only on redemption. At a ~30% pre-tax CAGR, that costs the bot roughly 5–6 points a year against a fund. The README's "in your pocket" figure taxes the paper book once at the end; the backtests don't tax it at all.

---

## 4. What to fix, in order, with predicted outcomes

| # | Fix | Kind | Effort | Predicted outcome | Confidence |
|---|---|---|---|---|---|
| 1 | **Stop intraday trading** (shadow-log the signals instead) | Config | Minutes | Saves an expected ₹315 a session (about ₹1,500 a week). Book: −6.97% today, and no further intraday bleed | **High**: 55 live trades plus research rounds 1–2, all net-negative |
| 2 | **Corporate-action guard + demerger credit**: 30%+ open-and-close gap → freeze, alert, no forced sale; an `adjust` command to credit the new company's shares; an entry-date split anchor | Code | ~1 day | ₹5.0k back on the paper book now (momentum −14.5% → −4.6%; book about −3.6%). No phantom −10% days or forced shell sales later. No change to expected return | **High** (accounting) |
| 3 | **Turn on the email alerts** | Setup | 5 minutes | You hear about the next anomaly the same evening | High |
| 4 | **Fix the state commit and the intraday 6-hour guard** | Code | Hours | No more lost state or phantom overnight MIS positions | High |
| 5 | **Swing: idle cash to a liquid ETF**, or retire the sleeve | Config / your call | Hours | +₹230 a month on idle cash. Swing's own backtest (+2.9% a year) trails that yield | Medium |
| 6 | **V4 shadow test**: pre-register, then run beside B4 for 12 months | Research | ~1 day | If the history holds: +0–5 points of CAGR and ~3 points less drawdown. Decide on the forward record only | Low–medium (post-hoc, one of 11 designs) |
| 7 | **Leave B4's score, rebalance and size alone** | — | — | Every textbook "improvement" tested did worse on untouched data or recent data (§2.4) | Medium |
| 8 | **Exclude BZ (non-compliant) names** | Config | Minutes | Small; removes a governance tail risk. Untested | Low |

**What all of this does to the outcome.**

- **Fixes 1–5 change the book's results.** They don't touch the one strategy with evidence. They stop the paper book losing money to an edge that doesn't exist, to idle cash and to its own accounting.
- **They don't change momentum's expected return or volatility.** That remains about 25–35% a year at the median, with drawdowns near 50%.
- **Whether to move more money into momentum is a decision, not a fix.** Moving the intraday and swing ₹1L into it would make the whole book behave like §3's table.

---

## 5. How to reproduce

```bash
python -m nsebot.research.diagnosis --out research_results     # or: research workflow, study=diagnosis
```

The study's report is `research_results/DIAGNOSIS.md`, with per-timing rows in `diagnosis_variants.csv`. The live-trade figures in §1–2 come from `state/` as committed on 28 Oct 2026.
