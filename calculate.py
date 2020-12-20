import csv
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

class Calculator:

	# REQUIRES:	a file that exists
	# EFFECTS: 	generates instance variabes that can be later used
	#			for calculations 
	def __init__(self, file_name):
		self.ticker_names = []
		self.ticker_data = {}
		self.num_stocks = 0

		try:
			ticker_file = open(file_name, 'r')
		except:
			print("Invalid File Name")
		
		ticker_list = csv.reader(ticker_file)

		for ticker in ticker_list:
			self.ticker_names.append(ticker[0])
			ticker_object = yf.Ticker(ticker[0])
			self.ticker_data[ticker[0]] = ticker_object
			self.num_stocks += 1
			if self.num_stocks % 100 == 0:
				print('Running stock #' , self.num_stocks)

		ticker_file.close()
		print('Finished compiling a dict of size' , self.num_stocks)

	# RETURNS:	the number of stocks that the calculator has stored
	def get_num(self):
		return self.num_stocks

	# RETURNS:	the list of all the tickers 
	def get_names(self):
		return self.ticker_names


	def test(self, ticker):
		# return self.ticker_data[ticker].info
		return self.ticker_data[ticker].history

	# REQUIRES: valid date string in the form YYYY-MM-DD
	# RETURNS:	the datetime object for the given date
	def get_date(self, date_string):
		temp = date_string.split('-')
		curr_day = datetime.date(int(temp[0]), int(temp[1]), int(temp[2]))

		return curr_day

	# REQUIRES: valid date string in the form YYYY-MM-DD
	# RETURNS:	the datetime object for the next day
	def day_after(self, date_string):
		curr_day = self.get_date(date_string)
		# print(curr_day)
		increment_day = datetime.timedelta(days=1)
		next_day = curr_day + increment_day
		# print(next_day)

		return next_day

	# REQUIRES: valid ticker name that was initialized in the calculator
	#			and limit date (exclusive)
	# RETURNS:	the return for the stock in the given time frame
	def calculate_return(self, ticker_name, start_date, end_date):

		start_price = self.ticker_data[ticker_name].history(start=start_date, end=self.day_after(start_date))['Close'].iloc[0]
		end_price = self.ticker_data[ticker_name].history(start=end_date, end=self.day_after(end_date))['Close'].iloc[0]

		# print(start_price, end_price)
		return_percentage = (end_price - start_price) / (start_price) * 100.0

		# return ticker_name + " had a " + str(round(return_percentage, 2)) + "% return"
		return return_percentage

	# REQUIRES: valid ticker name that was initialized in the calculator
	#			and limit date (exclusive)
	# RETURNS:	the CAGR for the stock in the given time frame
	def calculate_CAGR(self, ticker_name, start_date, end_date):
		return_percentage = float(self.calculate_return(ticker_name, start_date, end_date))
		d0 = self.get_date(start_date)
		d1 = self.get_date(end_date)
		delta = d1 - d0
		year_frac = float(delta.days / 365.25)

		cagr = ((return_percentage / 100) + 1) ** (1 / year_frac) - 1

		return cagr

	# REQUIRES: valid start and end dates in form YYYY-MM-DD
	# RETURNS:	the sharpe ratio of a given stock prior to the end date
	def risk_free_rate(self, start_date, end_date):
		ten_year_treasury_bond = yf.Ticker("^TNX")
		risk_free_rate = ten_year_treasury_bond.history(start=start_date, end=end_date)['Close'].mean()

		return risk_free_rate


	# REQUIRES: valid ticker name that was initialized in the calculator
	#			and limit date (exclusive)
	# RETURNS:	the sharpe ratio of a given stock prior to the end date
	def calculate_sharpe_ratio(self, ticker_name, start_date, end_date):
		
		price_history = self.ticker_data[ticker_name].history(start=start_date, end=end_date)['Close']
		
		returns = [0] * (price_history.size - 1)
		for i in range(price_history.size - 1):
			daily_return = (price_history.iloc[i + 1] - price_history.iloc[i]) / price_history.iloc[i] * 100
			returns[i] = daily_return

		pd_returns = pd.Series(returns)
		return_percentage = self.calculate_return(ticker_name, start_date, end_date)
		stdev = pd_returns.std()
		volatility = stdev * np.sqrt(price_history.size)
		rfr = self.risk_free_rate(start_date, end_date)

		# print(return_percentage)
		# print(stdev)
		# print(volatility)
		
		sharpe_ratio = (return_percentage - rfr) / volatility
		
		return sharpe_ratio

	# REQUIRES: valid ticker name that was initialized in the calculator
	#			and limit date (exclusive)
	# RETURNS:	the sortino ratio of a given stock prior to the end date
	def calculate_sortino_ratio(self, ticker_name, start_date, end_date):
		
		price_history = self.ticker_data[ticker_name].history(start=start_date, end=end_date)['Close']
		
		negative_count = 0
		returns = [None] * (price_history.size - 1)
		for i in range(price_history.size - 1):
			daily_return = (price_history.iloc[i + 1] - price_history.iloc[i]) / price_history.iloc[i] * 100
			if daily_return < 0:
				returns[i] = daily_return
				negative_count += 1

		pd_returns = pd.Series(returns)
		return_percentage = self.calculate_return(ticker_name, start_date, end_date)
		stdev = pd_returns.std()
		volatility = stdev * np.sqrt(negative_count)
		rfr = self.risk_free_rate(start_date, end_date)

		# print(return_percentage)
		# print(stdev)
		# print(volatility)
		
		sharpe_ratio = (return_percentage - rfr) / volatility
		
		return sharpe_ratio
		return 0

	def calculate_data_for_timeframe(self, start_date, end_date):
		
		with open("data_for_timeframe.csv", "w") as data_file:
			writer = csv.writer(data_file)
			writer.writerow(['Ticker', 'Sharpe', 'Sortino', 'CAGR'])
			
			for ticker_name in self.ticker_names:
				try:
					ticker_sharpe = self.calculate_sharpe_ratio(ticker_name, start_date, end_date)
					ticker_sortino = self.calculate_sortino_ratio(ticker_name, start_date, end_date)
					ticker_CAGR = self.calculate_CAGR(ticker_name, start_date, end_date)
				except:
					continue
				writer.writerow([ticker_name, ticker_sharpe, ticker_sortino, ticker_CAGR])

