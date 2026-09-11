# data_fetcher_free.py
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import warnings
import time
import random
import json
import re

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Symbols that must NOT have .NS appended
_INDEX_SYMBOLS = {'^NSEI', '^BSESN', '^NSEBANK', '^NSMIDCP'}


def _retry(fn, attempts=3, base_delay=1.5, what=""):
    """
    Run fn() with exponential backoff + jitter. Returns fn()'s result, or None
    if every attempt fails. This exists because yfinance calls from shared/
    datacenter IPs (e.g. GitHub Actions runners) get transiently rate-limited
    fairly often — a single failed request should NOT mean "no data forever".
    """
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                delay = base_delay * (2 ** i) + random.uniform(0, 0.5)
                logger.warning(f"⚠️ {what} attempt {i+1}/{attempts} failed ({e}) — retrying in {delay:.1f}s")
                time.sleep(delay)
    logger.error(f"✗ {what} failed after {attempts} attempts: {last_err}")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# DATA INTEGRITY
# ═════════════════════════════════════════════════════════════════════════════
# Everything downstream — ATR, Yang-Zhang volatility, efficiency ratio, stop
# distance, position size, the cost floor — is computed from this frame. A
# single corrupted bar does not produce a visibly wrong answer; it produces a
# plausible-looking one, which is worse, because nothing rejects it.
#
# The specific exposure here is CORPORATE ACTIONS. yfinance is called with
# auto_adjust=False, so a 1:10 split prints as a -90% close-to-close move that
# never happened. Consequences, in order of how much they cost:
#   • sigma explodes, so the stop is placed absurdly wide and position size
#     collapses to a fraction of what it should be;
#   • or the split sits just outside the volatility window while the price
#     level shifts inside it, so ATR is measured on one price regime and the
#     stop applied to another;
#   • efficiency ratio, RSI and ADX all read the artefact as a real move.
#
# A split is distinguishable from a genuine crash by the SHAPE of the bar, not
# its size. A real -35% day has an intraday range to match — the stock traded
# down through it. A split gaps the whole series to a new level and the bar
# itself looks utterly ordinary. That is the test applied below, and it is why
# a legitimate limit-down day is left alone rather than "corrected".

_SPLIT_RATIOS = [1/20, 1/10, 1/5, 1/4, 1/3, 1/2, 2/3, 3/2, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
_RATIO_TOLERANCE = 0.04        # 4% — corporate-action ratios are exact, prices are not
_JUMP_THRESHOLD = 0.22         # close-to-close move that triggers inspection
_ORDINARY_RANGE = 0.08         # a bar whose own high-low range is under this did not
                               # trade through the move it appears to have made


def _sanitize_ohlcv(df, symbol):
    """
    Returns (clean_df, notes). Never raises: a frame that cannot be repaired is
    returned with its problems described, and the caller decides.

    Four passes, cheapest first:
      1. structural  — non-positive prices, high/low inconsistent with open/close
      2. duplicates  — repeated or out-of-order dates
      3. corporate actions — detected and BACK-ADJUSTED rather than dropped, so
         the symbol stays tradeable with a continuous price series
      4. residual    — any remaining implausible jump is reported, not silently
         accepted
    """
    notes = []
    d = df.copy()

    # ── 1. Structural ────────────────────────────────────────────────────────
    bad_price = (d[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
    if bad_price.any():
        notes.append(f"dropped {int(bad_price.sum())} bar(s) with non-positive prices")
        d = d[~bad_price]

    if len(d) == 0:
        return d, notes

    body_hi = d[['open', 'close']].max(axis=1)
    body_lo = d[['open', 'close']].min(axis=1)
    # Clamp rather than drop: a high printed below the close is a feed glitch on
    # one field, and discarding the whole bar would punch a hole in every
    # rolling window that spans it.
    hi_bad = d['high'] < body_hi
    lo_bad = d['low'] > body_lo
    if hi_bad.any() or lo_bad.any():
        notes.append(f"clamped {int(hi_bad.sum() + lo_bad.sum())} inconsistent high/low value(s)")
        d.loc[hi_bad, 'high'] = body_hi[hi_bad]
        d.loc[lo_bad, 'low'] = body_lo[lo_bad]

    # ── 2. Duplicates and ordering ───────────────────────────────────────────
    if 'datetime' in d.columns:
        dupes = d['datetime'].duplicated(keep='last')
        if dupes.any():
            notes.append(f"dropped {int(dupes.sum())} duplicate date(s)")
            d = d[~dupes]
        if not d['datetime'].is_monotonic_increasing:
            notes.append("re-sorted out-of-order dates")
            d = d.sort_values('datetime')
    d = d.reset_index(drop=True)

    if len(d) < 3:
        return d, notes

    # ── 3. Corporate actions ─────────────────────────────────────────────────
    close = d['close'].astype(float)
    prev = close.shift()
    ratio = close / prev
    move = (ratio - 1.0).abs()
    bar_range = (d['high'].astype(float) - d['low'].astype(float)) / close

    suspects = move.index[(move > _JUMP_THRESHOLD) & (bar_range < _ORDINARY_RANGE)]
    for i in suspects:
        r = float(ratio.iloc[i])
        match = next((cand for cand in _SPLIT_RATIOS
                      if abs(r - cand) / cand <= _RATIO_TOLERANCE), None)
        if match is None:
            continue
        # Back-adjust everything BEFORE the event onto the post-event scale, so
        # the series is continuous and the most recent bars — the ones every
        # signal is computed from — keep their true traded prices.
        for col in ('open', 'high', 'low', 'close'):
            d.loc[:i - 1, col] = d.loc[:i - 1, col].astype(float) * match
        if 'volume' in d.columns and match > 0:
            d.loc[:i - 1, 'volume'] = d.loc[:i - 1, 'volume'].astype(float) / match
        notes.append(f"back-adjusted a {1/match:.4g}:1 corporate action at bar {i}")

    # ── 4. Residual ──────────────────────────────────────────────────────────
    close = d['close'].astype(float)
    residual = (close / close.shift() - 1.0).abs()
    remaining = int((residual > _JUMP_THRESHOLD).sum())
    if remaining:
        notes.append(f"{remaining} unexplained move(s) over {_JUMP_THRESHOLD*100:.0f}% remain "
                     f"— genuine limit moves, or an action with an unrecognised ratio")
    return d, notes


def _coerce_date(value):
    """Timestamps, datetimes, date strings and numpy datetimes all appear here
    depending on yfinance version; anything unparseable becomes None rather
    than an exception inside a calendar lookup."""
    if value is None:
        return None
    try:
        if hasattr(value, 'date') and not isinstance(value, str):
            return value.date()
        return pd.to_datetime(str(value)).date()
    except Exception:
        return None


class DataFetcherFree:
    def __init__(self):
        logger.info("✓ DataFetcherFree initialized (no API needed)")

    def _to_yf_symbol(self, symbol: str) -> str:
        """Return the correct yfinance ticker string for a given NSE symbol."""
        if symbol in _INDEX_SYMBOLS or symbol.startswith('^'):
            return symbol          # index — use as-is
        return f"{symbol}.NS"      # equity — append .NS

    def get_historical_data(self, symbol, days=200, min_bars=50):
        """
        Fetch historical daily OHLCV data.

        Args:
            symbol  : NSE symbol (e.g. 'RELIANCE') or index (e.g. '^NSEI')
            days    : Number of trading days wanted
            min_bars: Minimum acceptable rows (default 50); pass 1 to skip check
        """
        try:
            yf_symbol  = self._to_yf_symbol(symbol)
            end_date   = datetime.now()
            start_date = end_date - timedelta(days=days + 60)   # buffer for weekends/holidays

            logger.info(f"📥 Fetching {yf_symbol} from {start_date.date()} to {end_date.date()}")

            df = None
            for attempt in range(3):
                try:
                    candidate = yf.download(
                        yf_symbol,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        progress=False,
                        auto_adjust=False,
                    )
                    if candidate is not None and len(candidate) > 0:
                        df = candidate
                        break
                    # Empty result is often silent rate-limiting, not "no data" —
                    # worth a retry rather than trusting it immediately.
                    raise ValueError("empty dataframe returned")
                except Exception as e:
                    if attempt < 2:
                        delay = 1.5 * (2 ** attempt) + random.uniform(0, 0.5)
                        logger.warning(f"⚠️ {symbol} history fetch attempt {attempt+1}/3 failed ({e}) — retrying in {delay:.1f}s")
                        time.sleep(delay)

            if df is None or len(df) == 0:
                logger.error(f"✗ No data returned for {symbol} after retries")
                return None

            # Flatten MultiIndex columns (yfinance quirk with single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            df.reset_index(inplace=True)

            # Normalise column names
            df.columns = [str(c).lower().strip() for c in df.columns]

            # Accept 'date' or 'datetime'
            if 'date' in df.columns and 'datetime' not in df.columns:
                df = df.rename(columns={'date': 'datetime'})

            required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing  = [c for c in required if c not in df.columns]
            if missing:
                logger.error(f"✗ Missing columns {missing} for {symbol}. Got: {list(df.columns)}")
                return None

            df = df[required].dropna().copy()

            # Sanitise BEFORE the tail cut. A split sitting just outside the
            # retained window still shifts the price level inside it, so the
            # check has to see the whole fetched history, not the slice that
            # survives.
            df, notes = _sanitize_ohlcv(df, symbol)
            for note in notes:
                logger.warning(f"  🧹 {symbol}: {note}")

            if len(df) > days:
                df = df.tail(days).copy()

            df.reset_index(drop=True, inplace=True)

            if len(df) < min_bars:
                logger.warning(f"⚠️ Only {len(df)} bars for {symbol} (min={min_bars})")
                return None

            logger.info(
                f"✓ {symbol}: {len(df)} bars | "
                f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()} | "
                f"close ₹{df['close'].iloc[-1]:.2f}"
            )
            return df

        except Exception as e:
            logger.error(f"✗ Error fetching {symbol}: {e}")
            return None

    def get_ltp(self, symbol, attempts=3):
        """
        Return the last traded / most-recent close price.
        Does NOT apply the min_bars validation — always returns a number or None.
        Retries with backoff — a single failed request should not mean "no price".
        """
        yf_symbol = self._to_yf_symbol(symbol)

        def _fetch():
            ticker = yf.Ticker(yf_symbol)
            data   = ticker.history(period='5d')   # 5d gives at least 1 trading session
            if data is None or len(data) == 0:
                raise ValueError("empty history")
            return float(data['Close'].iloc[-1])

        ltp = _retry(_fetch, attempts=attempts, what=f"get_ltp({symbol})")
        if ltp is not None:
            logger.info(f"✓ {symbol} LTP: ₹{ltp:.2f}")
        else:
            logger.error(f"✗ No LTP data for {symbol}")
        return ltp

    def get_ltp_bulk(self, symbols, attempts=3, chunk_size=40):
        """
        Fetch last-traded prices for MANY symbols in as few HTTP requests as
        possible, using yf.download's multi-ticker support instead of one
        yf.Ticker(...).history() call per symbol.

        Why this matters: looping get_ltp() over N symbols makes N separate
        requests, which is exactly the pattern that gets rate-limited/blocked
        by Yahoo Finance when run from a shared IP (e.g. GitHub Actions). A
        single batched request is far more reliable and much faster.

        Returns: dict {symbol: price} — only symbols that resolved successfully
        are included. Any symbols missing from the result should be treated by
        the caller as "price unknown", not "price is zero".
        """
        symbols = list(dict.fromkeys(symbols))  # de-dupe, preserve order
        results = {}

        for i in range(0, len(symbols), chunk_size):
            chunk        = symbols[i:i + chunk_size]
            yf_to_orig   = {self._to_yf_symbol(s): s for s in chunk}
            yf_symbols   = list(yf_to_orig.keys())

            data = None
            for attempt in range(attempts):
                try:
                    candidate = yf.download(
                        yf_symbols, period='5d', group_by='ticker',
                        progress=False, auto_adjust=False, threads=True,
                    )
                    if candidate is not None and len(candidate) > 0:
                        data = candidate
                        break
                    raise ValueError("empty bulk dataframe")
                except Exception as e:
                    if attempt < attempts - 1:
                        delay = 1.5 * (2 ** attempt) + random.uniform(0, 0.5)
                        logger.warning(
                            f"⚠️ bulk LTP fetch attempt {attempt+1}/{attempts} "
                            f"failed for chunk of {len(chunk)} ({e}) — retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)

            if data is None:
                logger.error(f"✗ Bulk LTP fetch failed entirely for {len(chunk)} symbols — will retry individually")
                continue

            for yf_sym, orig_sym in yf_to_orig.items():
                try:
                    if len(yf_symbols) == 1:
                        # yf.download with a single ticker doesn't use a MultiIndex
                        closes = data['Close'].dropna()
                    else:
                        closes = data[yf_sym]['Close'].dropna()
                    if len(closes) > 0:
                        results[orig_sym] = float(closes.iloc[-1])
                except Exception:
                    continue  # this symbol just wasn't in the bulk result — handled below

        # Anything the bulk call didn't resolve gets a small number of
        # individual retries — worth the extra requests since it's normally
        # just a handful of symbols (e.g. your open positions) at this point.
        missing = [s for s in symbols if s not in results]
        if missing:
            logger.warning(f"  {len(missing)} symbols missing from bulk fetch — retrying individually: {missing}")
            for s in missing:
                price = self.get_ltp(s, attempts=2)
                if price is not None:
                    results[s] = price

        logger.info(f"✓ Bulk LTP: {len(results)}/{len(symbols)} symbols resolved")
        return results

    # ── Fundamentals ─────────────────────────────────────────────────────────
    # v2. The v1 parser looked for `soup.find('td', string=label)`. Screener.in
    # does not render its headline ratios in table cells — they live in a
    # `#top-ratios` list of `<li><span class="name">P/E</span><span
    # class="value">…</span></li>`. So every lookup returned None, every symbol
    # fell through to _default_fundamentals(), and the entire fundamental gate
    # has been scoring the SAME synthetic company (P/E 25, D/E 0.70, ROE 18%)
    # for every stock in the universe. It has never rejected anything, and a
    # genuinely broken balance sheet looked identical to a healthy one.
    #
    # Three changes beyond fixing the selector:
    #   • Parsing is multi-strategy. The top-ratios list first, then a
    #     document-wide label scan, then the ratio tables. Screener's markup
    #     shifts periodically and a single-strategy parser fails silently when
    #     it does — which is exactly the failure being repaired here.
    #   • The result says whether it was MEASURED. `fundamentals_measured`
    #     distinguishes "this company passed" from "we never looked", which the
    #     screener can then treat differently instead of waving both through.
    #   • Results are cached to disk for CACHE_DAYS. Fundamentals change
    #     quarterly; refetching 100+ symbols daily is pure rate-limit exposure
    #     for data that has not moved, and screener.in will throttle a scraper
    #     that does it.

    CACHE_PATH = '.fundamentals_cache.json'
    CACHE_DAYS = 7

    _LABELS = {
        'pe_ratio':        ('stock p/e', 'p/e'),
        'debt_to_equity':  ('debt to equity',),
        'roe_5yr':         ('roe', 'return on equity'),
        'current_ratio':   ('current ratio',),
        'revenue_cagr':    ('sales growth', 'compounded sales growth'),
        'market_cap':      ('market cap',),
        'book_value':      ('book value',),
    }

    @staticmethod
    def _to_float(text):
        """Screener renders '1,234.56 %', '₹ 1,234 Cr.' and '' — all of which
        must become a number or None, never an exception."""
        if not text:
            return None
        cleaned = re.sub(r'[^0-9.\-]', '', str(text).replace(',', ''))
        if cleaned in ('', '-', '.', '-.'):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _load_cache(self):
        try:
            with open(self.CACHE_PATH) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save_cache(self, cache):
        try:
            with open(self.CACHE_PATH, 'w') as fh:
                json.dump(cache, fh)
        except OSError:
            pass

    def get_fundamentals(self, symbol, force=False):
        cache = self._load_cache()
        hit = cache.get(symbol)
        if hit and not force:
            age = (datetime.now() - datetime.fromisoformat(hit['fetched'])).days
            if age < self.CACHE_DAYS:
                return hit['data']

        parsed = _retry(lambda: self._scrape_screener(symbol), attempts=2,
                        what=f"fundamentals({symbol})") or {}

        data = self._default_fundamentals()
        data.update({k: v for k, v in parsed.items() if v is not None})
        data['fundamentals_measured'] = bool(parsed)
        data['fundamentals_fields'] = sorted(parsed.keys())

        cache[symbol] = {'fetched': datetime.now().isoformat(), 'data': data}
        self._save_cache(cache)

        if parsed:
            logger.info(f"✓ Fundamentals {symbol}: {len(parsed)} fields "
                        f"(P/E {data.get('pe_ratio')}, D/E {data.get('debt_to_equity')}, "
                        f"ROE {data.get('roe_5yr')})")
        else:
            logger.warning(f"⚠ Fundamentals {symbol}: nothing parsed — defaults in use, "
                           f"flagged unmeasured")
        return data

    def _scrape_screener(self, symbol):
        """Returns only the fields actually found. An empty dict means the page
        was unreadable, which the caller reports rather than disguises."""
        headers = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36')}
        soup = None
        for url in (f"https://www.screener.in/company/{symbol}/consolidated/",
                    f"https://www.screener.in/company/{symbol}/"):
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                break
        if soup is None:
            return {}

        found = {}

        def record(label_text, value_text):
            label = (label_text or '').strip().lower().rstrip(':')
            value = self._to_float(value_text)
            if value is None:
                return
            for field, aliases in self._LABELS.items():
                if field in found:
                    continue
                if any(label == a or label.startswith(a) for a in aliases):
                    found[field] = value

        # Strategy 1 — the headline ratio list.
        top = soup.find(id='top-ratios')
        if top:
            for li in top.find_all('li'):
                name = li.find('span', class_='name')
                val = li.find('span', class_='value') or li.find('span', class_='number')
                if name is not None:
                    record(name.get_text(), (val.get_text() if val else
                                             li.get_text().replace(name.get_text(), '')))

        # Strategy 2 — document-wide label/value pairing, for markup changes.
        if len(found) < 3:
            for span in soup.find_all('span', class_='name'):
                sib = span.find_next('span')
                record(span.get_text(), sib.get_text() if sib else '')

        # Strategy 3 — the ratio tables, which do use <td>.
        if len(found) < 3:
            for td in soup.find_all('td'):
                nxt = td.find_next('td')
                if nxt is not None:
                    record(td.get_text(), nxt.get_text())

        # Screener reports ROE and growth as whole percentages; the screener
        # module accepts either convention but the fraction form is what the
        # rest of this codebase passes around.
        for pct_field in ('roe_5yr', 'revenue_cagr'):
            if pct_field in found and found[pct_field] > 1.5:
                found[pct_field] = found[pct_field] / 100.0
        return found

    # ── Earnings calendar ────────────────────────────────────────────────────
    # The single largest source of overnight gap risk a swing position carries.
    # This system places EOD stops at roughly 2.3 sigma — about 4-5% of price on
    # a typical name — while an Indian results announcement routinely moves a
    # midcap 6-12% at the open. Holding through results therefore replaces a
    # controlled, measured risk with an uncontrolled one on which the strategy
    # has no edge whatsoever: nothing in the technical stack forecasts an
    # earnings surprise, so the position is a coin flip sized as though it were
    # a 2:1 setup.
    #
    # yfinance exposes this free. Coverage for NSE names is good but not
    # complete and the dates shift, so every failure path returns None, and
    # None means "no constraint" rather than "no earnings" — an absent calendar
    # must never silently become a blackout that empties the universe.

    EARNINGS_CACHE_PATH = '.earnings_cache.json'
    EARNINGS_CACHE_DAYS = 3

    def get_next_earnings_date(self, symbol, force=False):
        """
        Returns the next scheduled results date as a datetime.date, or None.

        Cached for a few days rather than a week: unlike ratios, this value is
        actively approaching, and a stale entry is worse than no entry — it
        would clear a blackout that is in fact still in force.
        """
        cache = {}
        try:
            with open(self.EARNINGS_CACHE_PATH) as fh:
                cache = json.load(fh)
        except (OSError, ValueError):
            pass

        hit = cache.get(symbol)
        if hit and not force:
            try:
                age = (datetime.now() - datetime.fromisoformat(hit['fetched'])).days
                if age < self.EARNINGS_CACHE_DAYS:
                    return datetime.fromisoformat(hit['date']).date() if hit['date'] else None
            except (ValueError, KeyError):
                pass

        result = _retry(lambda: self._fetch_earnings_date(symbol), attempts=2,
                        what=f"earnings({symbol})")

        cache[symbol] = {'fetched': datetime.now().isoformat(),
                         'date': result.isoformat() if result else None}
        try:
            with open(self.EARNINGS_CACHE_PATH, 'w') as fh:
                json.dump(cache, fh)
        except OSError:
            pass
        return result

    def _fetch_earnings_date(self, symbol):
        """
        Two sources, because yfinance exposes this inconsistently across
        versions and tickers: the calendar dict first, then the earnings-dates
        frame. Only FUTURE dates count — get_earnings_dates() returns past
        announcements too, and a results date from last quarter would clear
        every blackout by being comfortably in the past.
        """
        ticker = yf.Ticker(self._to_yf_symbol(symbol))
        today = datetime.now().date()
        candidates = []

        try:
            cal = ticker.calendar
            if isinstance(cal, dict):
                for key in ('Earnings Date', 'earningsDate'):
                    val = cal.get(key)
                    for item in (val if isinstance(val, (list, tuple)) else [val]):
                        d = _coerce_date(item)
                        if d and d >= today:
                            candidates.append(d)
            elif cal is not None and hasattr(cal, 'loc'):
                for key in ('Earnings Date', 'earningsDate'):
                    if key in getattr(cal, 'index', []):
                        d = _coerce_date(cal.loc[key].iloc[0])
                        if d and d >= today:
                            candidates.append(d)
        except Exception:
            pass

        try:
            frame = ticker.get_earnings_dates(limit=8)
            if frame is not None and len(frame):
                for idx in frame.index:
                    d = _coerce_date(idx)
                    if d and d >= today:
                        candidates.append(d)
        except Exception:
            pass

        return min(candidates) if candidates else None


    def _default_fundamentals(self):
        return {
            'pe_ratio':        25.0,
            'sector_avg_pe':   25.0,
            'debt_to_equity':   0.70,
            'roe_5yr':          0.18,
            'revenue_cagr':     0.12,
            'current_ratio':    1.3,
            'market_cap':     500000,
            'fundamentals_measured': False,
            'fundamentals_fields': [],
        }
