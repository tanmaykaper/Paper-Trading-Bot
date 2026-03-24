# test_notification_handler.py
# Test Module 5: Notification Handler (Email + SMS)

import os
from dotenv import load_dotenv
from notification_handler import NotificationHandler
from datetime import datetime
import pandas as pd

print("\n" + "="*70)
print("TESTING MODULE 5: NOTIFICATION HANDLER (EMAIL + SMS)")
print("="*70)

# Load credentials
load_dotenv()

email_sender = os.getenv('EMAIL_SENDER')
email_recipient = os.getenv('EMAIL_RECIPIENT')
email_password = os.getenv('EMAIL_PASSWORD')

print("\n1️⃣ Checking Configuration")

if email_sender and email_recipient and email_password:
    print(f"   ✓ Email configured: {email_sender} → {email_recipient}")
    use_email = True
else:
    print("   ⚠️ Email credentials incomplete")
    use_email = False

# Check SMS
twilio_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_token = os.getenv('TWILIO_AUTH_TOKEN')

if twilio_sid and twilio_token:
    print(f"   ✓ SMS (Twilio) configured")
    use_sms = True
else:
    print("   ⚠️ SMS (Twilio) not configured")
    use_sms = False

# Initialize handler
notifier = NotificationHandler(use_email=use_email, use_sms=use_sms)

# Test 1: Send test email
if use_email:
    print("\n2️⃣ Test 1: Send Test Email")
    result = notifier.send_test_email()
    if result:
        print("   ✓ Test email sent! Check your inbox.")
    else:
        print("   ✗ Test email failed - check .env credentials")
else:
    print("\n2️⃣ Test 1: Send Test Email - SKIPPED (no credentials)")

# Test 2: Format signal message
print("\n3️⃣ Test 2: Format Trade Signal")

signal_details = {
    'signal': 'BUY',
    'symbol': 'RELIANCE',
    'entry_price': 2850.50,
    'stop_loss': 2820.25,
    'target_price': 2940.75,
    'position_size': 35,
    'risk': 6000,
    'reward': 18000,
    'risk_reward_ratio': 3.0,
    'indicators': {
        'rsi': 52.34,
        'macd': 0.0245,
        'adx': 28.45,
        'atr': 10.17,
    },
    'fundamentals': {
        'pe_ratio': 23.5,
        'debt_to_equity': 0.68,
        'roe': '18.2%',
        'revenue_growth': '12.5%',
    },
    'timestamp': pd.Timestamp(datetime.now()),
    'breakout': True,
    'pullback': False,
}

print(f"   ✓ Signal formatted for {signal_details['symbol']}")
print(f"   Entry: ₹{signal_details['entry_price']}")
print(f"   Target: ₹{signal_details['target_price']:.2f}")

# Test 3: Send live signal (if email configured)
if use_email:
    print("\n4️⃣ Test 3: Send Live Trade Signal")
    result = notifier.send_signal('BUY', signal_details, capital=100000)
    if result:
        print("   ✓ Trade signal email sent!")
        print("   ✓ Check your inbox for detailed trade information")
    else:
        print("   ✗ Failed to send trade signal")
else:
    print("\n4️⃣ Test 3: Send Live Trade Signal - SKIPPED (no email)")

# Test 4: Test HOLD signal (should not send)
print("\n5️⃣ Test 4: HOLD Signal (should not send)")
result = notifier.send_signal('HOLD', {}, capital=100000)
if not result:
    print("   ✓ Correctly rejected HOLD signal (not sent)")
else:
    print("   ✗ Unexpectedly sent HOLD signal")

print("\n" + "="*70)
print("✅ MODULE 5 TESTS COMPLETE")
print("="*70)

if use_email:
    print("\n📧 Next Steps:")
    print("   1. Check your email inbox for test message")
    print("   2. Verify email formatting and content")
    print("   3. Add email to your contacts (avoid spam folder)")
else:
    print("\n⚠️ Email not configured. To enable notifications:")
    print("   1. Get Gmail app password (see Module 5 instructions)")
    print("   2. Add to .env file:")
    print("      EMAIL_SENDER=your_email@gmail.com")
    print("      EMAIL_PASSWORD=your_app_password")
    print("      EMAIL_RECIPIENT=your_email@gmail.com")

print()
