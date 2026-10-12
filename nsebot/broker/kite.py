"""Zerodha Kite Connect adapter — live execution and (paid) market data.

Off by default. Activates only when KITE_API_KEY and an access token are
configured; the free personal account does not include the Connect API, and
everything else in nsebot runs without it.

What this handles natively:

TOKENS      Kite access tokens expire daily (~06:00 IST). The token is loaded
            from KITE_ACCESS_TOKEN or .kite_token.json (written by
            `python -m nsebot.kite_login`), its expiry stamp is checked before
            use, and any TokenException mid-session raises TokenExpired —
            never retried, never swallowed. The runner halts new entries and
            alerts. Open positions stay protected because every position gets
            an EXCHANGE-SIDE stop (GTT for CNC, SL-M for MIS) at entry, which
            keeps working when the bot's token does not.

RATE LIMITS Every call goes through KiteRateLimiter (per-endpoint pacing +
            order caps) before it leaves the process.

RETRIES     Classified by exception type. Network/5xx-style errors back off and
            retry; input, order, permission and token errors do not (retrying
            a rejected order just gets it rejected again). Order placement is
            NEVER blindly retried: after an ambiguous failure the order book is
            searched for this order's unique tag first, so a timeout that
            actually reached the exchange cannot become a duplicate position.

MARKET      Prices are rounded to the instrument's real tick_size from the
STRUCTURE   instrument master; quantities to lot_size. A pre-trade
            order_margins() check raises InsufficientMargin explicitly instead
            of letting the exchange reject the order silently.

Regulatory note: SEBI's retail-algo framework requires API order placement
to come from a static IP registered with the broker. GitHub Actions runners
have rotating IPs — live order placement needs a VPS with a fixed IP. Check
Zerodha's current Kite Connect requirements before enabling live mode.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from ..market import IST, TickTable, round_stop, round_to_tick
from ..ratelimit import KiteRateLimiter
from .base import Broker, EntryResolution, Fill, entry_action, exit_action

logger = logging.getLogger(__name__)

TOKEN_FILE = '.kite_token.json'
RETRYABLE = {'NetworkException', 'DataException', 'GeneralException', 'ConnectionError',
             'Timeout', 'ReadTimeout', 'ConnectTimeout'}
FATAL = {'InputException', 'OrderException', 'PermissionException'}


class TokenExpired(RuntimeError):
    pass


class InsufficientMargin(RuntimeError):
    pass


class OrderRejected(RuntimeError):
    pass


def token_expiry(created=None):
    """Kite tokens are invalidated at ~06:00 IST the next morning."""
    created = pd.Timestamp(created or pd.Timestamp.now(tz=IST))
    if created.tzinfo is None:
        created = created.tz_localize(IST)
    nxt = (created.tz_convert(IST).normalize() + pd.Timedelta(days=1, hours=6))
    return nxt


def load_token(path=TOKEN_FILE, now=None):
    env = os.environ.get('KITE_ACCESS_TOKEN')
    if env:
        return env
    if not os.path.exists(path):
        raise TokenExpired(f'no Kite access token (set KITE_ACCESS_TOKEN or run '
                           f'`python -m nsebot.kite_login`)')
    with open(path) as fh:
        data = json.load(fh)
    expires = pd.Timestamp(data['expires'])
    now = pd.Timestamp(now or pd.Timestamp.now(tz=IST))
    if now >= expires:
        raise TokenExpired(f'Kite token expired at {expires} — run `python -m nsebot.kite_login`')
    return data['access_token']


class KiteSession:
    """KiteConnect client + token + limiter + classified retries."""

    def __init__(self, api_key=None, client=None, limiter=None, token=None, token_path=TOKEN_FILE,
                 max_retries=3, base_delay_s=1.0, sleep=time.sleep):
        api_key = api_key or os.environ.get('KITE_API_KEY')
        if client is None:
            from kiteconnect import KiteConnect       # optional dependency
            if not api_key:
                raise TokenExpired('KITE_API_KEY is not set')
            client = KiteConnect(api_key=api_key)
        self.kite = client
        self.kite.set_access_token(token or load_token(token_path))
        self.limiter = limiter or KiteRateLimiter()
        self.max_retries, self.base_delay_s, self.sleep = max_retries, base_delay_s, sleep

    def call(self, category, method, *args, **kwargs):
        last = None
        for attempt in range(self.max_retries):
            self.limiter.acquire(category)
            try:
                return getattr(self.kite, method)(*args, **kwargs)
            except Exception as e:                       # classified below, never swallowed
                kind = type(e).__name__
                if kind == 'TokenException':
                    raise TokenExpired(f'Kite rejected the access token during {method}: {e}') from e
                if kind in FATAL or kind not in RETRYABLE:
                    raise
                last = e
                delay = self.base_delay_s * (2 ** attempt)
                logger.warning(f'  kite.{method} {kind} (attempt {attempt + 1}); retry in {delay:.1f}s')
                self.sleep(delay)
        raise last

    def validate(self):
        return self.call('default', 'profile')


class KiteBroker(Broker):
    name = 'kite'
    is_live = True

    def __init__(self, session, cache_dir='.cache', poll_s=1.0, poll_timeout_s=20.0, gtt_limit_buffer=0.01):
        self.s = session
        self.cache_dir = cache_dir
        self.poll_s, self.poll_timeout_s = poll_s, poll_timeout_s
        self.gtt_limit_buffer = gtt_limit_buffer
        self.ticks = self._load_instruments()

    # ── instrument master: real tick and lot sizes ──────────────────────────
    def _load_instruments(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f'kite_instruments_{datetime.now():%Y%m%d}.json')
        if os.path.exists(path):
            with open(path) as fh:
                return TickTable.from_kite_instruments(json.load(fh))
        rows = self.s.call('default', 'instruments', 'NSE')
        slim = [{'tradingsymbol': r.get('tradingsymbol'), 'tick_size': r.get('tick_size'),
                 'lot_size': r.get('lot_size'), 'segment': r.get('segment'),
                 'exchange': r.get('exchange')} for r in rows]
        with open(path, 'w') as fh:
            json.dump(slim, fh)
        return TickTable.from_kite_instruments(slim)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _k(self, name):
        return getattr(self.s.kite, name)

    @staticmethod
    def market_open(now=None):
        now = pd.Timestamp(now or pd.Timestamp.now(tz=IST))
        t = now.tz_convert(IST).time() if now.tzinfo else now.time()
        return now.weekday() < 5 and datetime.strptime('09:15', '%H:%M').time() <= t \
            <= datetime.strptime('15:30', '%H:%M').time()

    def available_cash(self, ledger_cash=None):
        m = self.s.call('default', 'margins', 'equity')
        return float(m['available'].get('live_balance', m['available'].get('cash', 0.0)))

    def _find_by_tag(self, tag):
        for o in self.s.call('default', 'orders') or []:
            if o.get('tag') == tag:
                return o.get('order_id')
        return None

    def _place(self, tag, **params):
        """Idempotent placement: an ambiguous failure checks the order book
        for this tag before trying again."""
        params['tag'] = tag[:20]
        try:
            return self.s.call('order', 'place_order', **params)
        except Exception as e:
            if type(e).__name__ in FATAL or isinstance(e, TokenExpired):
                raise OrderRejected(f'{params.get("tradingsymbol")}: {e}') from e
            existing = self._find_by_tag(params['tag'])
            if existing:
                logger.warning(f'  order {params["tag"]} reached the exchange despite {type(e).__name__}')
                return existing
            return self.s.call('order', 'place_order', **params)

    def _await(self, order_id):
        deadline = time.monotonic() + self.poll_timeout_s
        while True:
            hist = self.s.call('default', 'order_history', order_id) or []
            last = hist[-1] if hist else {}
            status = last.get('status')
            if status in ('COMPLETE', 'REJECTED', 'CANCELLED') or time.monotonic() > deadline:
                return last
            time.sleep(self.poll_s)

    def _check_margin(self, params):
        req = self.s.call('default', 'order_margins', [{
            'exchange': params['exchange'], 'tradingsymbol': params['tradingsymbol'],
            'transaction_type': params['transaction_type'], 'variety': params['variety'],
            'product': params['product'], 'order_type': params['order_type'],
            'quantity': params['quantity'], 'price': params.get('price') or 0,
            'trigger_price': params.get('trigger_price') or 0}])
        need = float(sum(r.get('total', 0.0) for r in req or []))
        have = self.available_cash()
        if need > have:
            raise InsufficientMargin(f"{params['tradingsymbol']}: needs ₹{need:,.0f}, "
                                     f"₹{have:,.0f} available")

    # ── Broker protocol ─────────────────────────────────────────────────────
    def place_entry(self, spec):
        k = self._k
        variety = k('VARIETY_REGULAR') if self.market_open() else k('VARIETY_AMO')
        params = dict(variety=variety, exchange=k('EXCHANGE_NSE'), tradingsymbol=spec.symbol,
                      transaction_type=k('TRANSACTION_TYPE_BUY') if spec.side == 'LONG'
                      else k('TRANSACTION_TYPE_SELL'),
                      quantity=int(spec.qty // self.ticks.lot(spec.symbol) * self.ticks.lot(spec.symbol)),
                      product=k('PRODUCT_MIS') if spec.product == 'MIS' else k('PRODUCT_CNC'),
                      order_type=k('ORDER_TYPE_MARKET'), validity=k('VALIDITY_DAY'))
        if params['quantity'] < 1:
            raise OrderRejected(f'{spec.symbol}: quantity rounds to zero at lot size')
        self._check_margin(params)
        oid = self._place(spec.tag or f'nb{spec.symbol[:10]}{int(time.time()) % 100000}', **params)
        logger.info(f'  kite: {params["transaction_type"]} {params["quantity"]} {spec.symbol} '
                    f'{params["product"]} {params["variety"]} -> {oid}')
        return oid

    def resolve_entry(self, pending, bar):
        last = (self.s.call('default', 'order_history', pending['order_id']) or [{}])[-1]
        status = last.get('status')
        spec = pending['spec']
        if status == 'COMPLETE':
            px, qty = float(last['average_price']), int(last.get('filled_quantity') or spec['qty'])
            return EntryResolution('FILLED', Fill(spec['symbol'], entry_action(spec['side']), qty, px,
                                                  0.0, pd.Timestamp(last.get('exchange_timestamp')
                                                                    or pd.Timestamp.now(tz=IST)),
                                                  pending['order_id'], 'kite fill'))
        if status in ('REJECTED', 'CANCELLED'):
            msg = last.get('status_message') or status
            logger.error(f'  kite: entry {spec["symbol"]} {status}: {msg}')
            return EntryResolution('CANCELLED', reason=f'{status}: {msg}')
        return EntryResolution('WAIT', reason=f'order status {status}')

    def exit(self, position, price, reason, when):
        k = self._k
        self.cancel_protection(position)          # never leave a resting stop to double-sell
        params = dict(variety=k('VARIETY_REGULAR') if self.market_open() else k('VARIETY_AMO'),
                      exchange=k('EXCHANGE_NSE'), tradingsymbol=position.symbol,
                      transaction_type=k('TRANSACTION_TYPE_SELL') if position.side == 'LONG'
                      else k('TRANSACTION_TYPE_BUY'),
                      quantity=int(position.qty),
                      product=k('PRODUCT_MIS') if position.mode == 'intraday' else k('PRODUCT_CNC'),
                      order_type=k('ORDER_TYPE_MARKET'), validity=k('VALIDITY_DAY'))
        oid = self._place(f'nbx{position.symbol[:10]}{int(time.time()) % 10000}', **params)
        px, note = float(price), 'kite AMO — fills at next open; journal shows decision price'
        if params['variety'] == k('VARIETY_REGULAR'):
            last = self._await(oid)
            if last.get('status') == 'COMPLETE':
                px, note = float(last['average_price']), 'kite fill'
            elif last.get('status') in ('REJECTED', 'CANCELLED'):
                raise OrderRejected(f'exit {position.symbol} {last.get("status")}: '
                                    f'{last.get("status_message")}')
        return Fill(position.symbol, exit_action(position.side), position.qty, px, 0.0,
                    pd.Timestamp(when), oid, note)

    def protect(self, position):
        """Exchange-side stop that survives the bot (and its token) going away."""
        k = self._k
        tick = self.ticks.tick(position.symbol, position.stop)
        trigger = round_stop(position.stop, position.side, tick)
        if position.mode == 'intraday':
            oid = self._place(f'nbs{position.symbol[:10]}{int(time.time()) % 10000}',
                              variety=k('VARIETY_REGULAR'), exchange=k('EXCHANGE_NSE'),
                              tradingsymbol=position.symbol,
                              transaction_type=k('TRANSACTION_TYPE_SELL') if position.side == 'LONG'
                              else k('TRANSACTION_TYPE_BUY'),
                              quantity=int(position.qty), product=k('PRODUCT_MIS'),
                              order_type=k('ORDER_TYPE_SLM'), trigger_price=trigger,
                              validity=k('VALIDITY_DAY'))
            position.meta['protection'] = {'kind': 'slm', 'id': oid, 'trigger': trigger}
            return oid
        limit = round_to_tick(trigger * (1 - self.gtt_limit_buffer), tick, 'down')
        resp = self.s.call('order', 'place_gtt', trigger_type=k('GTT_TYPE_SINGLE'),
                           tradingsymbol=position.symbol, exchange=k('EXCHANGE_NSE'),
                           trigger_values=[trigger], last_price=float(position.entry_price),
                           orders=[{'transaction_type': k('TRANSACTION_TYPE_SELL'),
                                    'quantity': int(position.qty), 'price': limit,
                                    'order_type': k('ORDER_TYPE_LIMIT'), 'product': k('PRODUCT_CNC')}])
        gid = resp.get('trigger_id') if isinstance(resp, dict) else resp
        position.meta['protection'] = {'kind': 'gtt', 'id': gid, 'trigger': trigger}
        return gid

    def update_protection(self, position):
        prot = position.meta.get('protection')
        if not prot:
            return self.protect(position)
        tick = self.ticks.tick(position.symbol, position.stop)
        trigger = round_stop(position.stop, position.side, tick)
        if trigger == prot.get('trigger'):
            return prot['id']
        if prot['kind'] == 'slm':
            self.s.call('order', 'modify_order', variety=self._k('VARIETY_REGULAR'),
                        order_id=prot['id'], trigger_price=trigger)
        else:
            self.cancel_protection(position)
            return self.protect(position)
        prot['trigger'] = trigger
        return prot['id']

    def cancel_protection(self, position):
        prot = position.meta.pop('protection', None)
        if not prot:
            return
        try:
            if prot['kind'] == 'slm':
                self.s.call('order', 'cancel_order', variety=self._k('VARIETY_REGULAR'),
                            order_id=prot['id'])
            else:
                self.s.call('order', 'delete_gtt', prot['id'])
        except Exception as e:                    # already triggered/expired: log, carry on exiting
            logger.warning(f'  kite: could not cancel protection {prot}: {e}')


class KiteDataProvider:
    """Market data from Kite (requires the paid Connect plan). Same schema as
    YahooProvider, paced at Kite's 3 req/s historical limit."""

    INTERVALS = {1: 'minute', 3: '3minute', 5: '5minute', 15: '15minute', 60: '60minute'}

    def __init__(self, session):
        self.s = session
        rows = session.call('default', 'instruments', 'NSE')
        self.tokens = {r['tradingsymbol']: r['instrument_token'] for r in rows}
        self.tokens.setdefault('^NSEI', 256265)          # NIFTY 50 index token

    def _fetch(self, symbol, start, end, interval, intraday):
        tok = self.tokens.get(symbol)
        if tok is None:
            return None
        rows = self.s.call('historical', 'historical_data', tok, start, end, interval)
        if not rows:
            return None
        df = pd.DataFrame(rows).rename(columns={'date': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        if intraday:
            df['datetime'] = (df['datetime'].dt.tz_localize(IST) if df['datetime'].dt.tz is None
                              else df['datetime'].dt.tz_convert(IST))
        else:
            df['datetime'] = df['datetime'].dt.tz_localize(None) if df['datetime'].dt.tz is not None \
                else df['datetime']
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    def daily(self, symbols, lookback_days=400):
        end = datetime.now()
        start = end - timedelta(days=int(lookback_days * 1.5))
        return {s: d for s in symbols
                if (d := self._fetch(s, start, end, 'day', False)) is not None}

    def intraday(self, symbols, interval_minutes=5, lookback_days=5):
        end = datetime.now()
        start = end - timedelta(days=int(lookback_days) + 3)
        iv = self.INTERVALS.get(interval_minutes, '5minute')
        return {s: d for s in symbols
                if (d := self._fetch(s, start, end, iv, True)) is not None}
