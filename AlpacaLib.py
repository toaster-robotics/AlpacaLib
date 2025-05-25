import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os
from typing import Optional, List, TypedDict, Union, Dict, Any, Tuple
load_dotenv()

API_KEY = os.getenv('ALPACA_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET')

BASE_TRADE_URL = 'https://api.alpaca.markets'
BASE_DATA_URL = 'https://data.alpaca.markets'


class Account(TypedDict):
    account_number: str
    status: str
    cash: float
    portfolio_value: float
    non_marginable_buying_power: float
    accrued_fees: float
    market_value: float
    equity: float
    last_equity: float
    buying_power: float
    initial_margin: float
    maintenance_margin: float


def fetch_url(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, max_retries: int = 3, sleep: float = 2):
    attempts = 0
    while attempts < max_retries:
        try:
            response = requests.get(url, params, headers=headers)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if e.response is not None:
                data = response.json()
                message = data['message']
                raise RuntimeError(f'[FETCH URL FAILED] {message}')
            attempts += 1
            if attempts < max_retries:
                time.sleep(sleep)


def get_account() -> Account:
    endpoint = '/v2/account'
    params = {
    }
    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,

    }
    response = fetch_url(BASE_TRADE_URL + endpoint, params, headers)
    data = response.json()

    account = {
        'account_number': data['account_number'],
        'status': data['status'],
        'cash': float(data['cash']),
        'portfolio_value': float(data['portfolio_value']),
        'non_marginable_buying_power': float(data['non_marginable_buying_power']),
        'accrued_fees': float(data['accrued_fees']),
        'market_value': float(data['long_market_value']),
        'equity': float(data['equity']),
        'last_equity': float(data['last_equity']),
        'buying_power': float(data['buying_power']),
        'initial_margin': float(data['initial_margin']),
        'maintenance_margin': float(data['maintenance_margin']),
    }
    return account


def get_activities(after: Optional[Union[pd.Timestamp, str]] = None) -> pd.DataFrame:
    def fix_time(df: pd.DataFrame):
        # Assume df already has these:
        df['transaction_time'] = pd.to_datetime(
            df['transaction_time'], errors='coerce')
        df['date'] = pd.to_datetime(
            df['date'], errors='coerce').dt.tz_localize('America/New_York')
        missing_mask = df['transaction_time'].isna(
        )

        # Count backward for each group based on original order
        reverse_offset = (
            df[missing_mask]
            .groupby('date')
            .cumcount(ascending=False)
        )

        # Assign synthetic times with reverse offsets and timezone
        df.loc[missing_mask, 'synthetic_datetime'] = df.loc[missing_mask,
                                                            'date'] + pd.to_timedelta(reverse_offset, unit='s')

        df['synthetic_datetime'] = df['synthetic_datetime'].fillna(
            df['transaction_time'])

        df['transaction_time'] = df['synthetic_datetime']
        df.drop(columns=['date', 'synthetic_datetime'], inplace=True)

        df['transaction_time'] = df['transaction_time'].dt.tz_convert(
            'America/New_York')
        df['transaction_time'] = df['transaction_time'].dt.round('s')

    endpoint = '/v2/account/activities'
    params = {
    }
    if after is not None:
        after = after.isoformat() if isinstance(after, pd.Timestamp) else after
        params['after'] = after

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,

    }

    page_token = None
    activities = []
    while True:
        if page_token is not None:
            params['page_token'] = page_token

        response = fetch_url(BASE_TRADE_URL + endpoint, params, headers)
        data = response.json()

        activities += data
        if len(data) == 100:
            page_token = data[-1]['id']
        else:
            break

    df = pd.DataFrame(activities)
    df = df.reindex(columns=['id',
                             'activity_type',
                             'transaction_time',
                             'type',
                             'price',
                             'qty',
                             'side',
                             'symbol',
                             'leaves_qty',
                             'order_id',
                             'cum_qty',
                             'order_status',
                             'activity_sub_type',
                             'date',
                             'net_amount',
                             'description',
                             'status',
                             'cusip',
                             'per_share_amount',
                             'corporate_action_id',
                             'swap_rate',])

    fix_time(df)
    df[['type', 'side', 'symbol', 'order_id', 'order_status', 'description', 'status', 'cusip', 'activity_sub_type', 'corporate_action_id']] = df[[
        'type', 'side', 'symbol', 'order_id', 'order_status', 'description', 'status', 'cusip', 'activity_sub_type', 'corporate_action_id']].fillna('')
    df[['price', 'qty', 'leaves_qty', 'cum_qty', 'swap_rate', 'net_amount', 'per_share_amount']] = df[[
        'price', 'qty', 'leaves_qty', 'cum_qty', 'swap_rate', 'net_amount', 'per_share_amount']].fillna(0)
    df[['price', 'qty', 'leaves_qty', 'cum_qty', 'swap_rate', 'net_amount', 'per_share_amount']] = df[[
        'price', 'qty', 'leaves_qty', 'cum_qty', 'swap_rate', 'net_amount', 'per_share_amount']].astype(float)

    return df


def get_market_clock() -> pd.Series:
    endpoint = '/v2/clock'
    params = {}
    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,

    }

    response = fetch_url(BASE_TRADE_URL + endpoint, params, headers)
    s = pd.Series(response.json())
    s[['timestamp', 'next_open', 'next_close']] = pd.to_datetime(
        s[['timestamp', 'next_open', 'next_close']], format='ISO8601').dt.tz_convert('America/New_York')
    return s


def get_market_calendar(start: Optional[Union[pd.Timestamp, str]] = None, end: Optional[Union[pd.Timestamp, str]] = None):
    endpoint = '/v2/calendar'
    params = {}
    if start is not None:
        start = start.isoformat() if isinstance(start, pd.Timestamp) else start
        params['start'] = start
    if end is not None:
        end = end.isoformat() if isinstance(end, pd.Timestamp) else end
        params['end'] = end
    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,

    }

    response = fetch_url(BASE_TRADE_URL + endpoint, params, headers)
    data = response.json()
    df = pd.DataFrame(data)

    df['open'] = pd.to_datetime(
        df['date'] + ' ' + df['open']).dt.tz_localize('America/New_York')
    df['close'] = pd.to_datetime(
        df['date'] + ' ' + df['close']).dt.tz_localize('America/New_York')
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.drop(columns=['session_open', 'session_close',
            'settlement_date'], inplace=True)

    return df


def process_bars(response: requests.Response, latest=False) -> pd.DataFrame:
    columns = ['date', 'symbol', 'open', 'high',
               'low', 'close', 'volume', 'trades', 'vw']
    if response.status_code == 200:
        data = response.json()['bars']
        frames = []
        for symbol, bars in data.items():
            if latest:
                frame = pd.DataFrame([bars])
            else:
                frame = pd.DataFrame(bars)
            if frame.empty:
                continue
            frame['symbol'] = symbol
            frames.append(frame)

        if len(frames) > 0:
            df = pd.concat(frames)
            df['t'] = pd.to_datetime(df['t']).dt.tz_convert('America/New_York')
            df = df.rename(columns={
                'c': 'close',
                'h': 'high',
                'l': 'low',
                'n': 'trades',
                'o': 'open',
                't': 'date',
                'v': 'volume',
            })
            df = df[columns]
            return df.sort_values(by=['symbol', 'date']).reset_index(drop=True)

    return pd.DataFrame(columns=columns)


def get_stock_historicals(symbols: List[str], start_date: Optional[Union[pd.Timestamp, str]] = None, end_date: Optional[Union[pd.Timestamp, str]] = None, resolution: str = '1D') -> pd.DataFrame:
    endpoint = '/v2/stocks/bars'
    params = {
        'symbols': ','.join([i.upper() for i in symbols]),
        'timeframe': resolution,
        'adjustment': 'split',
        'feed': 'iex',
        'limit': 10000,
    }
    if start_date is not None:
        start_date = start_date.isoformat() if isinstance(
            start_date, pd.Timestamp) else start_date
        params['start'] = start_date
    if end_date is not None:
        end_date = end_date.isoformat() if isinstance(
            end_date, pd.Timestamp) else end_date
        params['end'] = end_date

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }

    page_token = None
    frames = []
    while True:
        if page_token is not None:
            params['page_token'] = page_token

        response = fetch_url(BASE_DATA_URL + endpoint, params, headers)

        df = process_bars(response)
        frames.append(df)

        page_token = response.json().get('next_page_token')
        if not page_token:
            break
    df = pd.concat(frames, ignore_index=True)
    return df.set_index(['date', 'symbol'])


def get_option_historicals(symbols: List[str], start_date: Optional[Union[pd.Timestamp, str]] = None, end_date: Optional[Union[pd.Timestamp, str]] = None, resolution: str = '1D') -> pd.DataFrame:
    endpoint = '/v1beta1/options/bars'
    params = {
        'symbols': ','.join([i.upper() for i in symbols]),
        'timeframe': resolution,
        'limit': 10000,
    }
    if start_date is not None:
        start_date = start_date.isoformat() if isinstance(
            start_date, pd.Timestamp) else start_date
        params['start'] = start_date
    if end_date is not None:
        end_date = end_date.isoformat() if isinstance(
            end_date, pd.Timestamp) else end_date
        params['end'] = end_date

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }

    page_token = None
    frames = []
    while True:
        if page_token is not None:
            params['page_token'] = page_token

        response = fetch_url(BASE_DATA_URL + endpoint, params, headers)

        df = process_bars(response)
        frames.append(df)

        page_token = response.json().get('next_page_token')
        if not page_token:
            break
    df = pd.concat(frames, ignore_index=True)
    return df.set_index(['date', 'symbol'])


def get_crypto_historicals(symbols: List[str], start_date: Optional[Union[pd.Timestamp, str]] = None, end_date: Optional[Union[pd.Timestamp, str]] = None, resolution: str = '1D') -> pd.DataFrame:
    endpoint = '/v1beta3/crypto/us/bars'
    params = {
        'symbols': ','.join([i.upper() for i in symbols]),
        'timeframe': resolution,
        'limit': 10000,
    }
    if start_date is not None:
        start_date = start_date.isoformat() if isinstance(
            start_date, pd.Timestamp) else start_date
        params['start'] = start_date
    if end_date is not None:
        end_date = end_date.isoformat() if isinstance(
            end_date, pd.Timestamp) else end_date
        params['end'] = end_date

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }

    page_token = None
    frames = []
    while True:
        if page_token is not None:
            params['page_token'] = page_token

        response = fetch_url(BASE_DATA_URL + endpoint, params, headers)
        df = process_bars(response)
        frames.append(df)

        page_token = response.json().get('next_page_token')
        if not page_token:
            break
    df = pd.concat(frames, ignore_index=True)
    return df.set_index(['date', 'symbol'])


def get_historicals(
    stock_symbols: List[str] = [],
    option_symbols: List[str] = [],
    crypto_symbols: List[str] = [],
    start_date: Optional[Union[pd.Timestamp, str]] = None,
    end_date: Optional[Union[pd.Timestamp, str]] = None,
    resolution: str = '1D'
) -> pd.DataFrame:
    dfs = []
    if len(stock_symbols) > 0:
        df = get_stock_historicals(
            stock_symbols, start_date=start_date, end_date=end_date, resolution=resolution)
        if not df.empty:
            dfs.append(df)
    if len(option_symbols) > 0:
        df = get_option_historicals(
            option_symbols, start_date=start_date, end_date=end_date, resolution=resolution)
        if not df.empty:
            dfs.append(df)
    if len(crypto_symbols) > 0:
        df = get_crypto_historicals(
            crypto_symbols, start_date=start_date, end_date=end_date, resolution=resolution)
        if not df.empty:
            dfs.append(df)
    if len(dfs) > 1:
        df = pd.concat(dfs)
    else:
        df = dfs[0]
    return df


def latest_stock_bar(symbols: List[str]) -> pd.DataFrame:
    endpoint = '/v2/stocks/bars/latest'
    params = {
        'symbols': ','.join([i.upper() for i in symbols]),
        # 'feed': 'delayed_sip',
        'feed': 'iex',
    }

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }
    response = fetch_url(BASE_DATA_URL + endpoint, params, headers)
    data = response.json()['bars']

    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df['t'] = pd.to_datetime(df['t']).dt.tz_convert('America/New_York')
    df = df.rename(columns={
        'index': 'symbol',
        'c': 'close',
        'h': 'high',
        'l': 'low',
        'n': 'trades',
        'o': 'open',
        't': 'date',
        'v': 'volume',
    })
    df = df[['date', 'symbol', 'open', 'high',
             'low', 'close', 'volume', 'trades', 'vw']]
    return df.set_index(['date', 'symbol'])


def latest_crypto_bar(symbols: List[str]) -> pd.DataFrame:
    endpoint = '/v1beta3/crypto/us/latest/bars'
    params = {
        'symbols': ','.join([i.upper() for i in symbols]),
    }

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }
    response = fetch_url(BASE_DATA_URL + endpoint, params, headers)
    data = response.json()['bars']
    # print(data)

    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df['t'] = pd.to_datetime(df['t']).dt.tz_convert('America/New_York')
    df = df.rename(columns={
        'index': 'symbol',
        'c': 'close',
        'h': 'high',
        'l': 'low',
        'n': 'trades',
        'o': 'open',
        't': 'date',
        'v': 'volume',
    })
    df = df[['date', 'symbol', 'open', 'high',
             'low', 'close', 'volume', 'trades', 'vw']]
    return df.set_index(['date', 'symbol'])


def asset_info(symbol: str) -> dict:
    endpoint = '/v2/assets/%s' % symbol.upper()

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }
    response = requests.get(BASE_TRADE_URL + endpoint, headers=headers)
    data = response.json()
    return data


snapshot_rename1 = {
    'dailyBar': 'daily_bar',
    'latestQuote': 'latest_quote',
    'latestTrade': 'latest_trade',
    'minuteBar': 'minute_bar',
    'prevDailyBar': 'prev_bar',
}

snapshot_rename2 = {
    'c': 'close',
    'h': 'high',
    'l': 'low',
    'n': 'trades',
    'o': 'open',
    't': 'date',
    'v': 'volume',
    'ap': 'ask_price',
    'as': 'ask_size',
    'ax': 'ask_exchange',
    'bp': 'bid_price',
    'bs': 'bid_size',
    'bx': 'bid_exchange',
    'p': 'price',
    's': 'size',
    'x': 'exchange',
    'i': 'trade_id',
}


def process_snapshots(snapshots: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    date = pd.Timestamp.utcnow().tz_convert('America/New_York')
    is_option = 'greeks' in next(iter(snapshots.values()))

    frames = []
    for symbol in snapshots.keys():
        snapshot = snapshots[symbol]
        c = ','.join(snapshot['latestQuote'].pop('c', ''))
        snapshot['latestQuote']['condition'] = c
        c = ','.join(snapshot['latestTrade'].pop('c', ''))
        snapshot['latestTrade']['condition'] = c

        if is_option:
            snapshot['greeks']['implied_vol'] = snapshot.pop(
                'impliedVolatility')

        frame = pd.json_normalize(snapshot, sep='.').rename(
            columns=lambda x: tuple(x.split('.')))
        frame.columns = frame.columns = pd.MultiIndex.from_tuples(
            frame.columns)
        frame['symbol'] = symbol
        frame['date'] = date
        frames.append(frame)

    df = pd.concat(frames)
    df.columns = pd.MultiIndex.from_tuples([
        (snapshot_rename1.get(grp, grp), snapshot_rename2.get(sub, sub)) for grp, sub in df.columns
    ])
    df = df.set_index(['date', 'symbol'])
    df['daily_bar', 'date'] = pd.to_datetime(
        df['daily_bar', 'date']).dt.tz_convert('America/New_York')

    df['latest_quote', 'date'] = pd.to_datetime(
        df['latest_quote', 'date']).dt.tz_convert('America/New_York')
    df['latest_trade', 'date'] = pd.to_datetime(
        df['latest_trade', 'date']).dt.tz_convert('America/New_York')
    df['minute_bar', 'date'] = pd.to_datetime(
        df['minute_bar', 'date']).dt.tz_convert('America/New_York')
    df['prev_bar', 'date'] = pd.to_datetime(
        df['prev_bar', 'date']).dt.tz_convert('America/New_York')

    return df


def get_stock_snapshots(symbols: List[str]) -> pd.DataFrame:
    endpoint = '/v2/stocks/snapshots'

    params = {
        'symbols': ','.join([i.upper() for i in symbols]),
        # 'feed': 'sip',
        'feed': 'iex',
    }

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }
    response = fetch_url(BASE_DATA_URL + endpoint, params, headers)
    data = response.json()
    df = process_snapshots(data)
    df = df.drop(columns=[
        ('latest_quote', 'ask_exchange'),
        ('latest_quote', 'bid_exchange'),
        ('latest_quote', 'z'),
        ('latest_trade', 'exchange'),
        ('latest_trade', 'trade_id'),
        ('latest_trade', 'z'),
    ])
    return df


def get_option_snapshots(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = '/v1beta1/options/snapshots'

    params = {
        'symbols': ','.join([i.upper() for i in symbols]),
        # 'feed': 'opra',
        'feed': 'indicative',
    }

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }
    response = fetch_url(BASE_DATA_URL + endpoint, params, headers)
    data = response.json()
    df = process_snapshots(data['snapshots'])

    greeks = df['greeks']
    df = df.loc[:, ~df.columns.get_level_values(0).isin(['greeks'])]
    df = df.drop(columns=[
        ('latest_quote', 'ask_exchange'),
        ('latest_quote', 'bid_exchange'),
        ('latest_trade', 'exchange'),
    ])

    return df, greeks


def get_crypto_snapshots(symbols: List[str]) -> pd.DataFrame:
    endpoint = '/v1beta3/crypto/us/snapshots'

    params = {
        'symbols': ','.join([i.upper() for i in symbols]),
    }

    headers = {
        'Accept': 'application/json',
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': SECRET_KEY,
    }
    response = fetch_url(BASE_DATA_URL + endpoint, params, headers)
    data = response.json()
    df = process_snapshots(data['snapshots'])
    df[('latest_trade', 'condition')] = df[('latest_trade', 'tks')]
    df = df.drop(columns=[
        ('latest_trade', 'trade_id'),
        ('latest_trade', 'tks'),
    ])
    # print(df.dtypes)
    return df


if __name__ == '__main__':
    # df = get_activities()
    # print(asset_info('MSTY1'))
    # print(asset_info('MSTY'))
    # stock_symbol_check('AAPL')
    # stock_symbol_check('AAPL')
    # stock_symbol_check('AAPL')

    # df = get_stock_historicals(
    #     symbols=['AAPL'],
    #     # start_date='2025-01-01'
    #     start_date='1999-01-01',
    # )
    # df = get_crypto_historicals(
    #     symbols=['BTC/USD'],
    #     # start_date='2025-01-01'
    #     start_date='1999-01-01'
    # )
    # df = get_option_historicals(
    #     symbols=['CEP250718C00045000'],
    #     start_date='2025-05-15',
    # )
    # df = get_historicals(
    #     stock_symbols=['AAPL'],
    #     option_symbols=['CEP250718C00045000'],
    #     start_date='2025-05-15',
    # )

    # print(df)

    # ac = get_account()
    # print(ac)

    # df = latest_stock_bar(['TSLA', 'AAPL'])
    # print(df)

    # df = latest_crypto_bar(['BTC/USD', 'USDC/USD'])
    # print(df)

    symbols = ['AAPL', 'TSLA']
    df = get_stock_snapshots(symbols)
    # print(df)

    symbols = ['CEP250718C00045000', 'CEP250718C00050000']
    df, greeks = get_option_snapshots(symbols)
    # print(df)

    symbols = ['BTC/USD', 'USDC/USD']
    df = get_crypto_snapshots(symbols)
