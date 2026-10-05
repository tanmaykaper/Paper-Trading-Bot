"""Tradeable universes and sector labels.

The swing list is V1/V2's 255-name scan universe with dead tickers fixed
(renames and a suspension). The intraday list is narrower on purpose: MIS
needs tight spreads and deep books, which in practice means F&O-eligible
large and liquid mid caps.
"""

SECTORS = {
    'AUTO': ['MARUTI', 'TMPV', 'BAJAJ-AUTO', 'MOTHERSON', 'BALKRISIND', 'SUPRAJIT', 'HEROMOTOCO',
             'TVSMOTOR', 'ASHOKLEY', 'BHARATFORG', 'ENDURANCE', 'SUNDRMFAST', 'EXIDEIND', 'ARE&M',
             'CEATLTD', 'MRF', 'M&M', 'EICHERMOT'],
    'BANK': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'FEDERALBNK',
             'AUBANK', 'RBLBANK', 'IDFCFIRSTB', 'BANDHANBNK', 'CANBK', 'PNB', 'BANKBARODA',
             'UNIONBANK', 'INDIANB', 'KARURVYSYA', 'CUB', 'JKBANK'],
    'CAPGOODS': ['SIEMENS', 'ABB', 'GRINDWELL', 'DIXON', 'AMBER', 'CUMMINSIND', 'THERMAX', 'AIAENG',
                 'TIMKEN', 'SCHAEFFLER', 'KEI', 'POLYCAB', 'HAVELLS', 'VOLTAS', 'BLUESTARCO'],
    'CEMENT': ['ULTRACEMCO', 'KAJARIACER', 'SHREECEM', 'AMBUJACEM', 'ACC', 'DALBHARAT', 'JKCEMENT',
               'RAMCOCEM', 'STARCEMENT', 'GRASIM'],
    'CHEMICALS': ['DEEPAKNTR', 'AARTIIND', 'VINATIORGA', 'NAVINFLUOR', 'PIIND', 'SRF', 'ATUL',
                  'FINEORG', 'CLEAN', 'TATACHEM', 'GHCL', 'NOCIL', 'ALKYLAMINE'],
    'CONSUMER': ['TITAN', 'RADICO', 'TRENT', 'ABFRL', 'VMART', 'METROBRAND', 'BATAINDIA', 'RELAXO',
                 'PAGEIND', 'DEVYANI', 'JUBLFOOD', 'SAPPHIRE', 'ASIANPAINT'],
    'DEFENCE': ['HAL', 'BEL', 'BDL', 'MAZDOCK', 'COCHINSHIP', 'SOLARINDS', 'ASTRAMICRO', 'MTARTECH',
                'PARAS', 'ZENTEC', 'DATAPATTNS', 'BEML', 'GRSE', 'GARFIBRES', 'APOLLOMICRO', 'SIKA',
                'IDEAFORGE'],
    'ENERGY': ['RELIANCE', 'ONGC', 'BPCL', 'GAIL', 'IOC', 'HINDPETRO', 'PETRONET', 'IGL', 'MGL',
               'OIL', 'COALINDIA', 'NTPC', 'POWERGRID', 'TORNTPOWER'],
    'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'TATACONSUM', 'VSTIND', 'DABUR', 'GODREJCP', 'COLPAL',
             'EMAMILTD', 'JYOTHYLAB', 'BAJAJCON', 'ZYDUSWELL', 'HATSUN'],
    'HEALTHCARE': ['APOLLOHOSP', 'MAXHEALTH', 'FORTIS', 'NH', 'MEDANTA', 'KIMS', 'LALPATHLAB',
                   'METROPOLIS'],
    'INFRA': ['LT', 'NCC', 'KPIL', 'IRB', 'GMRAIRPORT', 'RVNL', 'IRCON', 'ENGINERSIN', 'HGINFRA',
              'ADANIPORTS', 'ADANIENT'],
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'MPHASIS', 'LTM', 'PERSISTENT', 'COFORGE',
           'KPITTECH', 'TATAELXSI', 'INTELLECT', 'HAPPSTMNDS', 'OFSS', 'CYIENT', 'SONATSOFTW',
           'BIRLASOFT', 'ZENSARTECH', 'NEWGEN', 'TATATECH', 'MASTEK'],
    'METAL': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'APLAPOLLO', 'RATNAMANI', 'VEDL', 'NMDC',
              'NATIONALUM', 'JINDALSTEL', 'SAIL', 'WELCORP', 'JSL'],
    'NBFC': ['BAJFINANCE', 'BAJAJFINSV', 'CHOLAFIN', 'CREDITACC', 'MUTHOOTFIN', 'MANAPPURAM',
             'LICHSGFIN', 'PFC', 'RECLTD', 'SBICARD', 'IIFL', 'POONAWALLA', 'SHRIRAMFIN',
             'ABCAPITAL'],
    'NEWAGE_TECH': ['ETERNAL', 'NYKAA', 'PAYTM', 'POLICYBZR', 'DELHIVERY', 'IRCTC', 'NAUKRI',
                    'INDIAMART', 'CARTRADE', 'MAPMYINDIA', 'EASEMYTRIP', 'NAZARA', 'SWIGGY',
                    'FIRSTCRY', 'OLAELEC', 'IXIGO', 'TBOTEK', 'ZAGGLE', 'RATEGAIN'],
    'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'ALKEM', 'TORNTPHARM', 'AUROPHARMA', 'GRANULES',
               'IPCALAB', 'LUPIN', 'GLENMARK', 'ZYDUSLIFE', 'BIOCON', 'LAURUSLABS', 'NATCOPHARM',
               'AJANTPHARM', 'ERIS', 'FDC', 'SUVENPHAR'],
    'REALTY': ['DLF', 'SOBHA', 'PHOENIXLTD', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE', 'MAHLIFE',
               'ANANTRAJ'],
    'RENEWABLE_EV': ['SUZLON', 'WAAREEENER', 'ADANIGREEN', 'NTPCGREEN', 'ACMESOLAR', 'PREMIERENE',
                     'JSWENERGY', 'TATAPOWER', 'INOXWIND', 'KPIGREEN', 'SWSOLAR', 'ORIENTGREEN'],
    'TELECOM': ['BHARTIARTL', 'INDUSTOWER', 'TEJASNET', 'HFCL', 'ITI'],
}

SECTOR_MAP = {sym: sector for sector, syms in SECTORS.items() for sym in syms}

SWING_UNIVERSE = list(dict.fromkeys(s for syms in SECTORS.values() for s in syms))

# F&O-eligible, high-turnover names: tight spreads matter more than breadth
# when the whole trade lives inside one session.
INTRADAY_UNIVERSE = [
    'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'SBIN', 'AXISBANK', 'KOTAKBANK',
    'BHARTIARTL', 'LT', 'ITC', 'HINDUNILVR', 'BAJFINANCE', 'BAJAJFINSV', 'MARUTI', 'M&M',
    'TMPV', 'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'SUNPHARMA', 'CIPLA', 'DRREDDY',
    'LUPIN', 'TITAN', 'ULTRACEMCO', 'GRASIM', 'WIPRO', 'HCLTECH', 'TECHM', 'PERSISTENT',
    'COFORGE', 'ONGC', 'NTPC', 'POWERGRID', 'COALINDIA', 'BPCL', 'IOC', 'GAIL', 'TATAPOWER',
    'ADANIENT', 'ADANIPORTS', 'INDUSINDBK', 'CANBK', 'PNB', 'BANKBARODA', 'DLF', 'BEL', 'HAL',
    'ETERNAL', 'TRENT', 'CHOLAFIN', 'SHRIRAMFIN', 'PFC', 'RECLTD', 'HEROMOTOCO', 'EICHERMOT',
    'BAJAJ-AUTO', 'TVSMOTOR', 'DIXON', 'NAUKRI',
]

INDEX_SYMBOL = '^NSEI'      # Nifty 50 — regime input for both modes
VIX_SYMBOL = '^INDIAVIX'


def sector_of(symbol):
    return SECTOR_MAP.get(symbol, 'OTHER')
