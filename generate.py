import csv
import yfinance as yf
import pandas as pd
import numpy as np
from calculate import *

calc = Calculator('stock_tickers.csv')
print(calc.get_num())
print(calc.get_names())
print(calc.test('FB'))

