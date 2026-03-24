# notification_handler.py
# Module 5: Notification Handler - Email + SMS alerts

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os
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
