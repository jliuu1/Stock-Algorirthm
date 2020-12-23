# generating the data used for the machine learning model

from calculate import *

calc = Calculator('stock_tickers.csv')

year = 2000
month = '01'
day = '01'

for i in range(9):
    start_date = str(year + i) + "-" + month + "-" + day
    end_date = str(year + i + 10) + "-" + month + "-" + day