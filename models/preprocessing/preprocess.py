import numpy as np
import pandas as pd
from stockstats import StockDataFrame as sdf
# from config import *

def load_dataset(*, file_name: str) -> pd.DataFrame:
    data = pd.read_csv(file_name)
    return data

def add_tech_indicators(df):
    """
    Calcualte technical indicators using stockstats package
    :param: df - pandas dataframe
    :return: df - pandas dataframe
    """
    stock = sdf.retype(df.copy())

    unique_ticker = stock.code.unique()
    
    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    for i in range(len(unique_ticker)):
        ## macd
        temp = stock[stock.code == unique_ticker[i]]['macd']
        temp = pd.DataFrame(temp)
        macd = pd.concat([macd, temp], ignore_index=True)
        ## rsi
        temp = stock[stock.code == unique_ticker[i]]['rsi_30']
        temp = pd.DataFrame(temp)
        rsi = pd.concat([rsi, temp], ignore_index=True)
        ## cci
        temp = stock[stock.code == unique_ticker[i]]['cci_30']
        temp = pd.DataFrame(temp)
        cci = pd.concat([cci, temp], ignore_index=True)
        ## adx
        temp = stock[stock.code == unique_ticker[i]]['dx_30']
        temp = pd.DataFrame(temp)
        dx = pd.concat([dx, temp], ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df