import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
from ta.trend import ADXIndicator
# from config import *

def create_ticker_data(df, ticker):
    return {ticker: df[df['code'] == ticker].reset_index(drop=True)}

def add_technical_indicators(df):
    df = df.sort_values(by="date", ascending=True)

    macd = MACD(df['close']).macd()
    macd_signal = MACD(df['close']).macd_signal()
    macd_histogram = MACD(df['close']).macd_diff()

    # Calculate RSI
    rsi = RSIIndicator(df['close']).rsi()

    # Calculate CCI
    cci = CCIIndicator(df['high'], df['low'], df['close']).cci()

    # Calculate ADX
    adx = ADXIndicator(df['high'], df['low'], df['close']).adx()

    # Add indicators to DataFrame
    df['macd'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_histogram
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = adx

    return df