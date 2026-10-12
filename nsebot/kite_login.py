"""Daily Kite Connect login — the official request_token flow.

    export KITE_API_KEY=... KITE_API_SECRET=...
    python -m nsebot.kite_login                 # prints the login URL
    python -m nsebot.kite_login <request_token> # exchanges it, saves .kite_token.json

Kite access tokens are invalidated every morning (~06:00 IST), so this runs
once per trading day before the bot. Automating the password/TOTP login
itself is deliberately not provided: it is against Zerodha's terms for
retail API users and the first thing to break when their login page changes.
"""

import json
import os
import sys

import pandas as pd

from .broker.kite import TOKEN_FILE, token_expiry
from .market import IST


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    api_key, secret = os.environ.get('KITE_API_KEY'), os.environ.get('KITE_API_SECRET')
    if not api_key or not secret:
        print('Set KITE_API_KEY and KITE_API_SECRET first.')
        return 2
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=api_key)
    if not argv:
        print('1. Open this URL, log in, and copy request_token from the redirect URL:\n')
        print('   ' + kite.login_url())
        print('\n2. Run: python -m nsebot.kite_login <request_token>')
        return 0
    session = kite.generate_session(argv[0], api_secret=secret)
    created = pd.Timestamp.now(tz=IST)
    payload = {'access_token': session['access_token'], 'user_id': session.get('user_id'),
               'created': str(created), 'expires': str(token_expiry(created))}
    with open(TOKEN_FILE, 'w') as fh:
        json.dump(payload, fh, indent=1)
    os.chmod(TOKEN_FILE, 0o600)
    print(f"✓ Token saved to {TOKEN_FILE} for {payload['user_id']}; valid until {payload['expires']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
