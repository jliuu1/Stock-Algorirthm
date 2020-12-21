import csv
import yfinance as yf
from calculate import Calculator

temp1 = open('stock_tickers.csv', 'r')
temp2 = open('cagr_data.csv', 'w')
temp3 = open('sharpe_data.csv', 'w')
temp4 = open('sortino_data.csv', 'w')
ticker_list = csv.reader(temp1)
cagr_list = csv.writer(temp2)
sharpe_list = csv.writer(temp3)
sortino_list = csv.writer(temp4)

for ticker in ticker_list:
	#im not sure what data i wanna work wtih yet so this will rest on the fence

temp1.close()
temp2.close()
temp3.close()
temp4.close()