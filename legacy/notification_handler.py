# notification_handler.py
# Module 5: Notification Handler - Email + SMS alerts

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class NotificationHandler:
    """
    Send trading signal alerts via Email and SMS
    """
    
    def __init__(self, use_email=True, use_sms=False):
        """
        Args:
            use_email: Send email notifications (default: True)
            use_sms: Send SMS via Twilio (default: False - requires paid account)
        """
        self.use_email = use_email
        self.use_sms = use_sms
        
        # Email setup
        if use_email:
            self.email_sender = os.getenv('EMAIL_SENDER')
            self.email_password = os.getenv('EMAIL_PASSWORD')
            self.email_recipient = os.getenv('EMAIL_RECIPIENT')
            
            if not all([self.email_sender, self.email_password, self.email_recipient]):
                logger.warning("⚠️ Email credentials incomplete. Email notifications disabled.")
                self.use_email = False
        
        # SMS setup (Twilio)
        if use_sms:
            try:
                from twilio.rest import Client
                account_sid = os.getenv('TWILIO_ACCOUNT_SID')
                auth_token = os.getenv('TWILIO_AUTH_TOKEN')
                self.twilio_client = Client(account_sid, auth_token)
                self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
                self.recipient_phone = os.getenv('RECIPIENT_PHONE')
                
                if not all([account_sid, auth_token, self.twilio_phone, self.recipient_phone]):
                    logger.warning("⚠️ Twilio credentials incomplete. SMS disabled.")
                    self.use_sms = False
            except Exception as e:
                logger.warning(f"⚠️ Twilio setup failed: {e}. SMS disabled.")
                self.use_sms = False
        
        logger.info(f"✓ NotificationHandler initialized (Email: {self.use_email}, SMS: {self.use_sms})")
    
    def send_signal(self, signal_type, signal_details, capital=100000):
        """
        Send trading signal via Email and/or SMS
        
        Args:
            signal_type: 'BUY' or 'SELL'
            signal_details: Dict with entry, SL, target, etc.
            capital: Total capital (for risk % calculation)
        
        Returns:
            bool: True if at least one notification sent successfully
        """
        
        if signal_type == 'HOLD':
            return False
        
        try:
            risk_percent = (signal_details['risk'] / capital) * 100
            
            # Format email body
            email_body = self._format_email_body(signal_details, risk_percent)
            
            # Format SMS body (shorter)
            sms_body = self._format_sms_body(signal_details, risk_percent)
            
            success = False
            
            # Send Email
            if self.use_email:
                if self._send_email(signal_details['symbol'], email_body):
                    success = True
            
            # Send SMS
            if self.use_sms:
                if self._send_sms(sms_body):
                    success = True
            
            return success
        
        except Exception as e:
            logger.error(f"✗ Error sending notification: {str(e)}")
            return False
    
    def _send_email(self, symbol, body):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_recipient
            msg['Subject'] = f"🎯 SWING TRADE SIGNAL: {symbol}"
            
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to Gmail SMTP
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)
            
            logger.info(f"✓ Email sent to {self.email_recipient}")
            return True
        
        except Exception as e:
            logger.error(f"✗ Email send failed: {str(e)}")
            return False
    
    def _send_sms(self, body):
        """Send SMS via Twilio"""
        try:
            message = self.twilio_client.messages.create(
                body=body,
                from_=self.twilio_phone,
                to=self.recipient_phone
            )
            
            logger.info(f"✓ SMS sent (SID: {message.sid})")
            return True
        
        except Exception as e:
            logger.error(f"✗ SMS send failed: {str(e)}")
            return False
    
    def _format_email_body(self, signal_details, risk_percent):
        """Format email with HTML styling"""
        
        entry_price = signal_details['entry_price']
        stop_loss = signal_details['stop_loss']
        target = signal_details['target_price']
        rsi = signal_details['indicators']['rsi']
        macd = signal_details['indicators']['macd']
        adx = signal_details['indicators']['adx']
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
        
        <div style="background-color: white; border-radius: 10px; padding: 20px; max-width: 600px; margin: 0 auto;">
        
        <h2 style="color: #27ae60; text-align: center;">🎯 SWING TRADE SIGNAL - BUY</h2>
        
        <hr>
        
        <h3 style="color: #2c3e50;">{signal_details['symbol']}</h3>
        <p><strong>Current Price:</strong> ₹{entry_price}</p>
        <p><strong>Time:</strong> {signal_details['timestamp'].strftime('%Y-%m-%d %H:%M IST')}</p>
        
        <hr>
        
        <h3 style="color: #2c3e50;">📊 Technical Analysis</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>RSI (14)</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{rsi:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>MACD</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{macd:.4f} (Bullish)</td>
            </tr>
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>ADX (Trend)</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{adx:.2f} (Strong)</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Entry Type</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{'Breakout' if signal_details['breakout'] else 'Pullback'}</td>
            </tr>
        </table>
        
        <h3 style="color: #2c3e50;">📋 Fundamentals</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>P/E Ratio</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{signal_details['fundamentals']['pe_ratio']}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Debt-to-Equity</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{signal_details['fundamentals']['debt_to_equity']}</td>
            </tr>
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>ROE</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{signal_details['fundamentals']['roe']}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Revenue Growth</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{signal_details['fundamentals']['revenue_growth']}</td>
            </tr>
        </table>
        
        <h3 style="color: #2c3e50;">💰 Trade Setup (1:3 Risk:Reward)</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Entry Price</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7; color: #27ae60; font-weight: bold;">₹{entry_price:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Stop-Loss</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7; color: #e74c3c; font-weight: bold;">₹{stop_loss:.2f}</td>
            </tr>
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Target Price</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7; color: #3498db; font-weight: bold;">₹{target:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Position Size</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{int(signal_details['position_size'])} shares</td>
            </tr>
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Risk Amount</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">₹{signal_details['risk']} ({risk_percent:.1f}%)</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Reward Amount</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">₹{signal_details['reward']}</td>
            </tr>
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Risk:Reward Ratio</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7; font-weight: bold;">1:{signal_details['risk_reward_ratio']:.1f}</td>
            </tr>
        </table>
        
        <h3 style="color: #2c3e50; margin-top: 20px;">⚡ Action Required</h3>
        <p style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #27ae60;">
            <strong>Place a BUY LIMIT order at ₹{entry_price:.2f}</strong> with Stop-Loss at ₹{stop_loss:.2f}<br>
            Hold until target ₹{target:.2f} or stop-loss is hit.<br>
            <strong>Expected hold duration:</strong> 3-10 trading days
        </p>
        
        <hr>
        
        <p style="color: #7f8c8d; font-size: 12px; text-align: center;">
            This is an automated trading signal. Do your own due diligence before trading.
        </p>
        
        </div>
        
        </body>
        </html>
        """
        
        return html
    
    def _format_sms_body(self, signal_details, risk_percent):
        """Format SMS (160 chars max)"""
        
        symbol = signal_details['symbol']
        entry = signal_details['entry_price']
        sl = signal_details['stop_loss']
        target = signal_details['target_price']
        
        sms = f"🎯 BUY {symbol} @ ₹{entry:.0f} | SL: ₹{sl:.0f} | Target: ₹{target:.0f} | Risk: {risk_percent:.0f}%"
        
        return sms
    
    def send_alert(self, subject, body):
        """
        Generic operational/system-health alert email — distinct from
        send_signal() (which expects a trade-signal-shaped dict) and
        _send_email() (which hardcodes a "SWING TRADE SIGNAL" subject line).

        Added to fix a real bug: run_paper_trading.py already calls
        alert_notifier.send_alert(subject=..., body=...) at two safety
        checkpoints (price-fetch health check failing, and the drawdown
        circuit breaker activating) — but this method never existed here,
        so triggering EITHER condition raised AttributeError and crashed
        the entire run_eod() outright, since neither call site is wrapped
        in a try/except. Worse, this meant the exact conditions meant to
        warn Tanmay something was wrong (degraded price data, or a
        drawdown halt) would silently fail to notify him at all — the
        crash happened inside the call meant to send that warning.

        Never raises — an alert failing to send should never crash the run
        that was trying to warn about a problem in the first place. Always
        logs the alert (visible in GitHub Actions logs) regardless of
        whether email is configured or delivery succeeds.
        """
        logger.warning(f"🔔 ALERT: {subject}\n{body}")

        if not self.use_email:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_recipient
            msg['Subject'] = f"⚠️ {subject}"
            msg.attach(MIMEText(body.replace('\n', '<br>'), 'html'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)

            logger.info(f"✓ Alert email sent to {self.email_recipient}")
            return True

        except Exception as e:
            logger.error(f"✗ Alert email failed: {str(e)}")
            return False


    def send_daily_brief(self, report=None, summary=None, trades_df=None, funnel=None,
                         calibrator=None, target=None, also_print=True):
        """
        Builds and sends the v11 daily brief. Never raises — a report that
        fails to send must not take down the run it was reporting on, the same
        contract send_alert() already holds.

        also_print keeps the brief visible on the console when email is not
        configured, which is the common case for a cron run on a free tier.
        """
        try:
            body = build_daily_brief(report, summary, trades_df, funnel, calibrator, target)
        except Exception as e:
            logger.error(f"Could not build daily brief: {e}")
            return False
        if also_print:
            print(body)
        ms = (report or {}).get('market_state') or {}
        subject = (f"NSE Bot — {ms.get('state', 'DAILY')} — "
                   f"{datetime.now().strftime('%d %b')}")
        try:
            return self.send_alert(subject, body)
        except Exception as e:
            logger.error(f"Could not send daily brief: {e}")
            return False

    def send_test_email(self):
        """Send test email to verify setup"""
        try:
            test_html = """
            <html>
            <body style="font-family: Arial, sans-serif;">
            <div style="background-color: white; border-radius: 10px; padding: 20px;">
            <h2 style="color: #27ae60;">✓ Test Email Successful!</h2>
            <p>Your NSE Swing Trading Bot email notifications are working.</p>
            <p>You will receive trading signals at this email address.</p>
            </div>
            </body>
            </html>
            """
            
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_recipient
            msg['Subject'] = "✓ NSE Trading Bot - Email Test"
            msg.attach(MIMEText(test_html, 'html'))
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)
            
            logger.info("✓ Test email sent successfully!")
            return True
        
        except Exception as e:
            logger.error(f"✗ Test email failed: {str(e)}")
            return False

# ═════════════════════════════════════════════════════════════════════════════
# DAILY BRIEF  (v11)
# ═════════════════════════════════════════════════════════════════════════════
# _format_email_body() formats ONE signal, which is what v3 needed: the bot
# emitted independent alerts and the operator decided. v11 decides for itself,
# so the useful report is no longer "here is a trade" but "here is what the
# system did today, and is it on track".
#
# Everything below is chosen on one test: would seeing this number change what
# the operator does tomorrow? Five things pass that test, and most of the old
# report does not:
#
#   PACE vs TARGET   the only number that says whether the objective is live.
#                    Stated as the win rate the current geometry needs, against
#                    the win rate actually being achieved. Everything else is
#                    downstream of this.
#   COST DRAG        the leak that consumed 75% of gross profit historically
#                    (₹573 of ₹767). It is silent, it compounds, and it is the
#                    first thing that goes wrong when position sizes drift down.
#   FUNNEL           when a day produces no trades, "which gate ate the
#                    universe" is the difference between a filter that is
#                    working and one that is mis-tuned. Without it, a quiet
#                    week is indistinguishable from a broken scanner.
#   MARKET STATE     why exposure is where it is, including the trigger text,
#                    so a defensive stretch reads as a decision rather than a
#                    malfunction.
#   POSITION HEALTH  R multiple, sessions held against each position's OWN
#                    horizon, broken thesis signals, and results proximity.
#
# Pure text, built by a module-level function so it can be printed to console
# when email is off and asserted against in tests. Plain text rather than HTML
# on purpose: this is read on a phone, and a fixed-width block survives that
# better than a table that reflows.

def build_daily_brief(report=None, summary=None, trades_df=None, funnel=None,
                      calibrator=None, target=None):
    """
    report:     orchestrator.TradingOrchestrator.run() output
    summary:    PaperTradingManager.get_summary() output
    trades_df:  the full trades frame (open and closed)
    funnel:     SignalGenerator.funnel_summary() for the scan
    calibrator: calibration.WinCalibrator, for the reliability line
    target:     {'equity': 75000, 'months': 3.5} or None to omit the pace block

    Every section degrades independently: a missing input drops its own block
    and never blanks the report.
    """
    L = []
    add = L.append
    ms = (report or {}).get('market_state') or {}

    add("=" * 62)
    add(f"  NSE SWING BOT — DAILY BRIEF   {datetime.now().strftime('%a %d %b %Y')}")
    add("=" * 62)

    # ── Market state ─────────────────────────────────────────────────────────
    if ms:
        add(f"\nMARKET      {ms.get('state','?')}   exposure {ms.get('exposure',0):.2f}x   "
            f"slots {ms.get('max_slots','?')}   score {ms.get('risk_score',0):.2f}")
        b = ms.get('breadth') or {}
        if b:
            add(f"            breadth {b.get('pct_above_20',0)*100:.0f}% >EMA20  "
                f"NH-NL {b.get('nh_nl_spread',0):+.2f}  5d {b.get('breadth_chg_5') or 0:+.2f}")
        for t in (ms.get('triggers') or []):
            add(f"            ⚠ {t}")
        if not ms.get('new_entries_allowed', True):
            add("            ENTRIES SUSPENDED — exits and trailing continue as normal")

    # ── Capital ──────────────────────────────────────────────────────────────
    if summary:
        eq = _f(summary.get('current_equity'))
        add(f"\nCAPITAL     equity ₹{eq:,.0f}   free ₹{_f(summary.get('free_cash')):,.0f}   "
            f"P&L ₹{_f(summary.get('total_pnl')):+,.0f}")
        pend = ((report or {}).get('plan') or {}).get('diagnostics', {}).get('pending_settlement')
        if pend:
            add(f"            ₹{pend:,.0f} settles T+1 — deployable tomorrow")

    # ── Pace against the objective ───────────────────────────────────────────
    # The honest version of "am I on track": what the current trade geometry
    # requires, against what it is delivering. Stated as a required WIN RATE
    # because that is the one input the operator can actually watch converge,
    # and because a bald "on track / not on track" hides which assumption broke.
    closed = _closed(trades_df)
    if target and summary and closed is not None and len(closed) >= 5:
        start = _f(summary.get('initial_equity'), 50000.0)
        eq = _f(summary.get('current_equity'), start)
        pnl = closed['net_pnl'].astype(float)
        n = len(pnl)
        per_trade_pct = float(pnl.mean()) / start * 100.0
        win_rate = float((pnl > 0).mean()) * 100.0
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        avg_w = float(wins.mean()) / start * 100.0 if len(wins) else 0.0
        avg_l = abs(float(losses.mean())) / start * 100.0 if len(losses) else 0.0

        months = float(target.get('months', 3.5))
        goal = float(target.get('equity', 75000))
        # Trades expected in the remaining window, from the pace actually
        # observed rather than an assumed cadence.
        trades_needed = max(int(n / max(months, 0.1) * months), 1)
        required_pct = ((goal / max(eq, 1.0)) ** (1.0 / max(trades_needed, 1)) - 1.0) * 100.0
        add(f"\nPACE        {n} closed | win {win_rate:.0f}% | ₹{float(pnl.mean()):+,.0f}/trade "
            f"({per_trade_pct:+.2f}% of capital)")
        if avg_w > 0 and (avg_w + avg_l) > 0:
            req_win = (required_pct + avg_l) / (avg_w + avg_l) * 100.0
            verdict = ('ON PACE' if win_rate >= req_win else
                       f'SHORT BY {req_win - win_rate:.0f}pp')
            add(f"            ₹{goal:,.0f} needs {required_pct:+.2f}%/trade "
                f"→ {req_win:.0f}% win rate at {avg_w:.1f}%/{avg_l:.1f}% win/loss — {verdict}")
        projected = eq * ((1 + per_trade_pct / 100.0) ** trades_needed)
        add(f"            at the current rate: ₹{projected:,.0f} in ~{months:.1f} months")

    # ── Cost drag ────────────────────────────────────────────────────────────
    if closed is not None and len(closed) and 'commission' in closed:
        comm = float(closed['commission'].astype(float).sum())
        gross = float(closed['gross_pnl'].astype(float).sum()) if 'gross_pnl' in closed else None
        notional = (closed['entry_price'].astype(float) * closed['position_size'].astype(float))
        bps = float((closed['commission'].astype(float) / notional.replace(0, float('nan'))).mean() * 1e4)
        line = f"\nCOSTS       ₹{comm:,.0f} paid | {bps:.0f} bps avg round trip"
        if gross and gross > 0:
            line += f" | {comm/gross*100:.0f}% of gross profit"
        add(line)
        if bps > 70:
            add("            ⚠ over 70 bps — position sizes are too small to carry the "
                "flat DP charge; check min_notional and tranche_ok")

    # ── Today ────────────────────────────────────────────────────────────────
    if report:
        ex = report.get('exits') or {}
        fills = report.get('fills') or {}
        add(f"\nTODAY       {report.get('candidates', 0)} candidates | "
            f"{len(fills.get('opened') or [])} filled | "
            f"{len(ex.get('closed') or [])} closed | "
            f"{len(ex.get('trailed') or [])} stops raised")
        for sym, px, note in (fills.get('opened') or []):
            add(f"   ▲ {sym:<12} ₹{px:,.2f}  {note}")
        for sym, reason, rmult in (ex.get('closed') or []):
            add(f"   ■ {sym:<12} {rmult:+.2f}R  {reason}")
        for sym, why in (fills.get('expired') or []):
            add(f"   ✗ {sym:<12} {why}")
        placed = report.get('placed') or []
        if placed:
            add(f"   → resting for tomorrow's open: {', '.join(placed)}")

    # ── Open positions ───────────────────────────────────────────────────────
    openp = _open(trades_df)
    if openp is not None and len(openp):
        add(f"\nOPEN ({len(openp)})")
        for _, r in openp.iterrows():
            entry, stop = _f(r.get('entry_price')), _f(r.get('stop_loss'))
            init = _f(r.get('initial_stop_loss'), stop)
            risk = max(entry - init, 1e-9)
            last = _f(r.get('last_price'), entry)
            rm = (last - entry) / risk
            held = int(_f(r.get('bars_held'), 0))
            horizon = int(_f(r.get('time_exit_bars'), 0)) or None
            bits = [f"{rm:+.2f}R", f"{held}/{horizon or '?'} bars"]
            if stop >= entry:
                bits.append("stop above entry")
            brk = _f(r.get('health_signals_at_exit'), None)
            if brk:
                bits.append(f"{int(brk)} thesis signals")
            add(f"   {str(r.get('symbol')):<12} {'  '.join(bits)}")

    # ── Funnel ───────────────────────────────────────────────────────────────
    if funnel:
        top = list(funnel.items())[:3]
        add("\nFUNNEL      " + " | ".join(f"{v}x {k[:34]}" for k, v in top))

    # ── Learned layers ───────────────────────────────────────────────────────
    # Both report their own activation state, because "the model is off" and
    # "the model is on and says this" are decisions the operator should be able
    # to tell apart at a glance.
    engines = (report or {}).get('engines') or {}
    if engines.get('risk_scalar'):
        rs = engines['risk_scalar']
        add(f"\nRISK SCALE  {rs.get('scalar', 1.0):.2f}x  ({rs.get('reason', '')})")
    if engines.get('meta'):
        add(f"\nMETA MODEL  {engines['meta']}")
    if engines.get('pyramids'):
        add("\nPYRAMIDS")
        for sym, size, why in engines['pyramids']:
            add(f"   ⬆ {sym:<12} +{size}  {why}")
    if engines.get('ladder'):
        add(f"\nLADDER      {engines['ladder']}")

    # ── Calibration ──────────────────────────────────────────────────────────
    if calibrator is not None:
        try:
            n = int(getattr(calibrator, 'n_trades', 0))
            if n and getattr(calibrator, 'brier_model', None) is not None:
                verdict = ('beating' if calibrator.brier_model < calibrator.brier_prior
                           else 'not yet beating')
                add(f"\nCALIBRATION fitted on {n} scored trades — {verdict} the analytic prior "
                    f"({calibrator.brier_model:.3f} vs {calibrator.brier_prior:.3f})")
            else:
                add(f"\nCALIBRATION {n} scored trades so far — still running on the analytic prior")
        except Exception:
            pass

    add("\n" + "=" * 62)
    return "\n".join(L)


def _f(value, default=0.0):
    try:
        if value is None:
            return default
        v = float(value)
        return default if v != v else v          # NaN check without importing numpy
    except (TypeError, ValueError):
        return default


def _closed(df):
    if df is None or not hasattr(df, 'columns') or 'status' not in df:
        return None
    out = df[df['status'] == 'CLOSED']
    return out if len(out) else None


def _open(df):
    if df is None or not hasattr(df, 'columns') or 'status' not in df:
        return None
    return df[df['status'] == 'OPEN']
