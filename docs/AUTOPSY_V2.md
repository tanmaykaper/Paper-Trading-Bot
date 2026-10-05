# V2 Autopsy — why the bot has opened zero trades since 14 Sep 2026

**Verdict:** V2 isn't crashing. It is locked by three things stacked on top of each other, and any one of them is enough to stop all entries:

| # | Lock | Type | Status today | Blocks forever? |
|---|------|------|--------------|-----------------|
| A | `MarketState` DEFENSIVE clamp sets exposure to 0x and slots to 0 | Regime gate | **Firing every run** since 15 Sep | No, it releases in a bull tape |
| B | `entry_requires_risk_on` vetoes NEUTRAL even when it has granted a slot | Regime gate (commit `027aa1f`) | Fired on 21 Sep (1 slot granted, entries still suspended) | No, but only RISK_ON passes |
| C | The T+1 pending-order book lives in `orchestrator_state.json`, which the GitHub workflow never commits | **State machine lock** | Hidden behind A and B | **Yes.** Even in a perfect bull market, every plan is lost before it can fill |

Behind those three sit a 27-gate signal conjunction (D) and execution filters strict enough that even a plan that survived would usually be abandoned (E).

Each claim below is reproduced offline in `diagnostics/test_v2_paralysis.py` (10/10 pass on the V2 code as committed), or quoted from the bot's own GitHub Actions logs.

---

## 0. Timeline (from the repo's own artifacts)

| Date | Event | Source |
|------|-------|--------|
| 09 Sep | Last entry ever: MOTHERSON @ ₹163.85, booked **immediately at the signal close** by the v10 pipeline | Actions run 133 log: `✅ Trade OPENED (tranched): MOTHERSON @ ₹163.85` |
| 13–15 Sep | V2 lands: `orchestrator.py`, `market_state.py`, `entry_execution.py`, `portfolio_allocator.py`, `momentum_rank.py`, and the 'growth' profile in 6 modules | `git log` |
| 14 Sep | First V2 run (run 136) | Actions |
| 15 Sep | `MARKET STATE: DEFENSIVE ... exposure 0.00x slots 0` with 7 positions still open | run 137 log |
| 17 Sep | The last V1-era positions stop out. Book goes flat | `daily_equity.csv` |
| 21 Sep | `MARKET STATE: NEUTRAL ... exposure 0.13x slots 1`, then `Entries: suspended — NEUTRAL — entries require a confirmed risk-on tape` | run 142 log |
| 28 Sep | DEFENSIVE, breadth 26% | run 146 log |
| 02 Oct | DEFENSIVE, breadth 9%, `Candidates: 0` | run 150 log |

The V2 scanner **has not evaluated a single symbol** on any day since it was deployed. `_scan()` is never reached, so the funnel is empty and the logs give no hint of what the signal layer would have done.

---

## 1. The Quant Trap: condition conjunction

For one symbol to become a filled position under V2, **every** layer below has to pass on the same day. Each one is plausible on its own. Stacked together, the chance of a trade goes to roughly zero:

```
Market state  ─ DEFENSIVE (any of 6 triggers) → 0 slots
              ─ NEUTRAL → suspended (Lock B)
              ─ RISK_ON needs risk_score ≥ 0.54 with no trigger
Safety layer  ─ kill switch / ≥70% fetch / index staleness / 30% drawdown
Momentum gate ─ top 30% of 245 symbols only (73 survive)
Data quality  ─ stale / halted / circuit-frozen
Signal gen    ─ 27 sequential `return 'HOLD'` exits:
                bars, regime, debounce, fundamentals, min price, turnover,
                circuit headroom, EMA trend, ADX, DI spread, RSI band,
                efficiency ratio, extension, MACD, pattern fired, retired
                pattern, entry quality, earnings ×2, geometry, stop band,
                reachable R:R ≥ 1.70, circuit-stop cover, sizing, economics
Quality add   ─ + up to 0.12 more quality in a weak tape
Enricher      ─ earnings blackout, sentiment hard veto
Allocator     ─ min edge bps/day, correlation block, sector cap,
                heat cap (14% × exposure), slot count (3), cash
T+1 fill      ─ open must be ≤ +0.5% AND ≤ +0.25σ above signal close
Re-anchor     ─ R:R at the actual fill must still be ≥ 1.70
Economics     ─ re-check after re-anchor shrinks the size
```

The math behind Lock A (proven in `test_lock_a_risk_on_is_unreachable_while_index_trend_is_zero`):

- The risk score has four weighted parts: breadth 0.34, index trend 0.26, volatility 0.22, drawdown 0.18.
- In every V2 log since 15 Sep, `index_trend = 0.00`.
- With trend at zero, the best possible score is 0.74. Reaching the 0.54 RISK_ON bar needs the other three parts to average **73% of their maximum at the same time**.
- That is a confirmed bull tape. The bot only trades long, so V2 sits in cash for the whole of any correction, which is the whole period since it went live.

V1 (v10) had **one** regime input (Nifty EMA-50 vs SMA-200, about 4 flips a year) and allowed longs in BEAR. It traded through the same tape.

---

## 2. State machine locks

### Lock C: the pending-order book is wiped by every GitHub Actions run *(critical, latent)*

`orchestrator.py` changed entries from **open now** to **plan today, fill tomorrow**:

```python
# orchestrator.py:322-329  — plans are placed, not traded
self.pending.place(entry['symbol'], d, route(d))
...
# orchestrator.py:626-632  — the book is saved to a local file
json.dump(payload, open(self.state_path, 'w'), ...)   # orchestrator_state.json
```

```yaml
# .github/workflows/main.yml — only two files survive the runner
git add paper_trades.csv daily_equity.csv
```

Each Actions run is a fresh checkout. The next run calls `_load_state()`, finds no file, and starts with an empty book. **Every plan ever placed is thrown away before it can fill.** Locks A and B are hiding this today. The day the market turns RISK_ON, V2 will place plans, lose them overnight, and still report zero fills, with no error in the log.

The same missing file also resets `MarketState`'s hysteresis (`clamp_active`, `cooldown_left`, `last_exposure`) every morning. The "slow restore" behaviour that `market_state.py` describes as *"the single most important line in the module"* has never actually run in production.

### `in_position`-style flags

There are no boolean position flags. Open/closed state comes from `paper_trades.csv` `status`, which `close_position()` resets correctly. Seven positions were closed and freed between 15 and 17 Sep. **This is not the cause.**

---

## 3. Market structure validation

| Check | Finding | Severity |
|-------|---------|----------|
| **Tick size** | `NSEMicrostructure.TICK = 0.05` is hard-coded. NSE now uses price-banded tick sizes (for example, a ₹1,000+ stock is not on a ₹0.05 grid). `round_to_tick(1443.07)` returns ₹1,443.05, which a live order would get rejected for if the instrument's `tick_size` is 0.10. Paper mode books it silently. **Kite's `instruments()` dump (`tick_size` per token) must be the only source of truth.** | Landmine for live trading |
| **Runner tranche target** | `entry + 1000 × risk` (₹5,066 on a ₹176 GAIL, ₹30,673 on a ₹1,313 RELIANCE). A GTT or limit at that price is far outside the circuit band and would be rejected by the exchange. | Landmine for live trading |
| **F&O lot sizes** | Not applicable. The bot trades cash equity only. | — |
| **Margin shortfall** | Cash/CNC with no leverage. `open_trade()` logs and shrinks size to fit cash rather than failing silently. | OK |
| **Swallowed exceptions** | `orchestrator._value()` returns a default on *any* exception (for example, `free_cash` falls back to 0.0, which means no cash and nothing gets allocated, with no log line). `run_safety_checks`, the candidate enricher, the alpha engine and earnings lookups are all wrapped in bare `except`. None of these is firing today, but each would turn a bug into a quiet zero-trade day. | Medium |
| **Fundamentals parser** | Every one of 236 "measured" symbols reports `D/E 0.7`, a parser default and not real data. The fundamental gate is working from fiction, and the scrape takes about 4 of the 6.5 minutes of each run. | Medium (wasted run time and false confidence) |
| **Equity-curve risk scalar** | `profit_engine.risk_scalar` reads the `equity` column, which has been empty since April (real values are in `total_portfolio_value`). The **0.71x Kelly haircut** in every brief is calculated from stale ₹10k/₹50k placeholders. | Medium (silently undersizes) |
| **Daily brief** | Prints `equity ₹0` every day because it reads the wrong key. | Cosmetic |

---

## 4. Reality check on V1, from its own trade log

The brief for this rebuild says V1 *"generated solid returns"*. The repository's own records don't support that, and the sizing work in Phase 3 depends on getting this number right:

| Metric (₹50k era, 30 Jul – 17 Sep) | Value |
|---|---|
| Trade decisions / CSV rows | 25 / 53 |
| Net P&L | **−₹1,307** (−2.6%) |
| Gross P&L before costs | −₹396 |
| Costs paid | ₹911, which is **70% of the net loss** |
| Row win rate / decision win rate | 30.2% / 24% |
| Payoff ratio (avg win ÷ avg loss) | 1.38 |
| **Full-Kelly fraction on realised stats** | **−0.20**, meaning a negative measured edge |
| Peak | about +₹3.6k unrealised on 25 Aug, then given back |

Where the money came from, and where it went:

| Holding period | Decisions | Net |
|---|---|---|
| ≤ 2 days | 3 | −₹420 (all full stops) |
| 5–10 days | 8 | **−₹2,132** |
| 10–20 days | 14 | **+₹1,245** |

| Exit | Rows | Net |
|---|---|---|
| Stop-loss (incl. gap) | 33 | −₹3,106 |
| Time exit (14–17 days) | 18 | **+₹1,557** |
| Target | 2 | +₹242 |

**What V1 was actually exploiting:** buying names already in motion (stoch_cross +₹733, cmf_accum +₹617) and **holding them about two weeks** in a rising tape (17–25 Aug). It lost money on pullback-buying (−₹1,508 over 30 rows) and on breakouts. The September correction stopped out every long within 5–10 days, all together. That was correlated beta, not stock-picking failure.

**Implication for the rebuild:**
1. V1 did trade, and it showed a real **payoff-from-holding** asymmetry. But the sample is 25 decisions, which is far too few to prove an edge, and the measured Kelly is negative.
2. Costs turned a −₹396 gross into −₹1,307 net. Three tranches pay the flat DP charge three times. **Untranched positions and intraday (MIS, no DP charge) both attack the largest measured leak directly.**
3. Any Kelly sizer fed the realised stats will size to zero. Phase 3 has to use a shrunk prior with a hard minimum risk budget and say so openly, not pretend the edge is proven.

---

## 5. What the fix must do (handover to Phases 2–4)

1. **One regime input, used as a size dial and not an on/off switch.** Trade smaller in weak tapes instead of not trading at all.
2. **Execute on the same run that decides.** Live: place the order immediately via Kite. Paper: book at the actual quote or close with modelled slippage. No overnight book that depends on a file nobody saves.
3. **Persist every piece of state the runner depends on**, or don't keep any.
4. **About 5 gates, not 40**: liquidity, trend, trigger, risk, cost.
5. **Tick size and lot size from Kite's instrument master. No hard-coded constants.**
6. **No bare `except` on the order path.** Fail loudly, alert, halt entries.
