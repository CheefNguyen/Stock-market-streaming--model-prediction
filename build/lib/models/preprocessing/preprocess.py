import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
from ta.trend import ADXIndicator
# from config import *

def create_ticker_dict(df):
    result = {}
    for ticker in df['code'].unique():
        ticker_data = df[df['code'] == ticker].copy()
        result[ticker] = ticker_data
    return result

def add_technical_indicators(dict):
    for ticker, data in dict.items():
        # Calculate MACD
        macd = MACD(data['close']).macd()
        macd_signal = MACD(data['close']).macd_signal()
        macd_histogram = MACD(data['close']).macd_diff()

        # Calculate RSI
        rsi = RSIIndicator(data['close']).rsi()

        # Calculate CCI
        cci = CCIIndicator(data['high'], data['low'], data['close']).cci()

        # Calculate ADX
        adx = ADXIndicator(data['high'], data['low'], data['close']).adx()

        # Add indicators to DataFrame
        data['macd'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Histogram'] = macd_histogram
        data['rsi'] = rsi
        data['cci'] = cci
        data['adx'] = adx
