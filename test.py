import requests
from datetime import datetime
import pandas as pd

from models.env.TradingEnv import *
from models.models.DQNAgent import *
from models.preprocessing.preprocess import *

from pymongo import MongoClient
from dotenv import load_dotenv
import os

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
from ta.trend import ADXIndicator

code = "TCB"
today = "2024-05-17"
# today = datetime.today().strftime('%Y-%m-%d')
# API_VNDIRECT = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{code}~date:gte:2024-01-01~date:lte:{today}&size=9990&page=1"
API_VNDIRECT = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:EIB~date:gte:2015-01-01~date:lte:2024-05-17&size=99900&page=1"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

response = requests.get(API_VNDIRECT,verify=True, headers=headers)
raw_data = response.json()['data']

df = pd.DataFrame(raw_data)

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

# load_dotenv()
# DBUSERNAME = os.environ.get("DB_USERNAME")
# DBPASSSWORD = os.environ.get("DB_PASSWORD")
# cluster = MongoClient(f"mongodb+srv://{DBUSERNAME}:{DBPASSSWORD}@clusterthesis.keduavv.mongodb.net/")


# data_dict = df.to_dict(orient='records')

# db = cluster["thesis"]
# collection = db["dailyRawData"]
# collection.insert_many(data_dict)

print(df[df['date'] == "2024-05-17"].to_dict('records'))
