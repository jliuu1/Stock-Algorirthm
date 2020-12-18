import csv
import yfinance as yf
import pandas as pd
import numpy as np

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
		return self.ticker_data[ticker].info
		# return self.ticker_data[ticker].info['volume']

	# REQUIRES: valid ticker name that was initialized in the calculator
	#			and limit date (exclusive)
	# RETURNS:	the expected return value for the stock in the given time frame
	def caluclate_expected_return(self, ticker_name, end_date):
		#history = self.ticker_data[ticker_name].history(period='max', interval='1wk')
		#if history['date'] < end_date we're gonna have to implement a date comparison function

		return 0
	
	# REQUIRES: valid ticker name that was initialized in the calculator
	#			and limit date (exclusive)
	# RETURNS:	the risk-free rate for the stock in the given time frame
	def calculate_risk_free_rate(self, ticker_name, end_date):
		return 0

	# REQUIRES: valid ticker name that was initialized in the calculator
	#			and limit date (exclusive)
	# RETURNS:	the sharpe ratio of a given stock prior to the end date
	def calculate_sharpe_ratio(self, ticker_name, end_date):
		'''
		theres 2 ways to implement this function. either we generate 
		the expected return and risk-free rate outside and average it
		(which i, actually dont really agree with) or
		we for loop through the dates of the stock in this function and
		generate each value. this way its easier ot calculate standard
		devation.
		'''
		return 0

	# REQUIRES: valid ticker name that was initialized in the calculator
	#			and limit date (exclusive)
	# RETURNS:	the sortino ratio of a given stock prior to the end date
	def calculate_sortino_ratio(self, ticker_name):
		'''
		same debate as above but i thikn the forloop method will be
		easier to calculate standard devation of downside
		'''
		return 0