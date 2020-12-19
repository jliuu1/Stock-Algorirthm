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

	def day_after(self, date_string):
		temp = date_string.split('-')
		curr_day = datetime.date(int(temp[0]), int(temp[1]), int(temp[2]))
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

		return ticker_name + " had a " + str(round(return_percentage, 2)) + "% return"

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