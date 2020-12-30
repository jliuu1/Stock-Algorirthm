import csv
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import calendar
 
class Calculator:
 
	# REQUIRES: a file that exists
	# EFFECTS:  generates instance variabes that can be later used
	#           for calculations
	def __init__(self, file_name):
		self.ticker_names = []
		self.ticker_data = {}
		self.ticker_histories = {} # dataframe of history
		self.ticker_starts = {}
		self.num_stocks = 0

		try: # should be with
			ticker_file = open(file_name, 'r')
		except:
			print("Invalid File Name")

		ticker_list = csv.reader(ticker_file)

		for ticker in ticker_list:
			self.ticker_names.append(ticker[0])
			ticker_object = yf.Ticker(ticker[0])
			
			self.ticker_histories[ticker[0]] = Calculator.download_history(ticker_object)
			
			self.num_stocks += 1
			if self.num_stocks % 100 == 0:
				print('Running stock #' , self.num_stocks)

		self.find_ticker_start()

		with open("cache/^TNX.csv") as datafile:
			self.rfr = pd.read_csv(datafile)

		ticker_file.close()
		print('Finished compiling a dict of size' , self.num_stocks)

	@staticmethod
	def download_history(ticker):
		filename = f'cache/{ticker.ticker}.csv'
		try:
			# try to read
			return pd.read_csv(filename)
		except:
			# download data
			data = ticker.history(period='max')
			# create new file and write data to it
			# create cache folder if it doesn't exist
			# os.mkdirs('cache', ) TODO probably good idea to fix this
			with open(filename, 'w') as cache:	
				cache.write(data.to_csv())
				return data

	# REQUIRES: 
	# MODIFIES: Class attribute ticker_times, filling the dict values with the first recorded data in YFinance for the stock
	# RETURNS:
	def find_ticker_start(self):
		for ticker in self.ticker_names:
			try:
				df = self.ticker_histories[ticker]
				self.ticker_starts[ticker] = pd.to_datetime(df.iloc[1]["Date"])
			except:
				print("Could not find", ticker)

	# RETURNS:  the number of stocks that the calculator has stored
	def get_num(self):
		return self.num_stocks

	# RETURNS:  the list of all the tickers
	def get_names(self):
		return self.ticker_names
	
	# RETURNS: 	the dict of all tickers and their start dates
	def get_ticker_starts(self):
		return self.ticker_starts


	def test(self, ticker):
	# return self.ticker_data[ticker].info
		return self.ticker_data[ticker].history

	# REQUIRES: valid date string in the form YYYY-MM-DD
	# RETURNS:  the datetime object for the given date
	def get_date(self, date_string):
		temp = date_string.split('-')
		curr_day = datetime.date(int(temp[0]), int(temp[1]), int(temp[2]))

		return curr_day
	
	# REQUIRES: valid date object
	# RETURNS:  string for the date
	def date_to_string(self, date):
		return date.strftime("%Y-%m-%d")

	# REQUIRES: number of days as int
	# RETURNS:  timedelta object for number of days
	def get_time_delta(self, num_days):
		return datetime.timedelta(days = num_days)

	# REQUIRES: valid datestring in the form YYYY-MM-DD
	# RETURNS:  the date of the date 1 month after
	def month_after_notstring(self, date):
		date += datetime.timedelta(days=calendar.monthrange(date.year,date.month)[1])
		return date

	# REQUIRES: valid datestring in the form YYYY-MM-DD
	# RETURNS:  the string of the date 5 years before
	def five_years_earlier(self, date_string):
		temp = date_string.split('-')
		return str(int(temp[0]) - 5) + "-" + temp[1] + "-" + temp[2]

	# REQUIRES: 2 valid dates as strings in the format YYYY-MM-DD
	# RETURNS:  the number of months in that timeframe
	def num_months(self, start_date_string, end_date_string):
		return int((self.get_date(end_date_string) - self.get_date(start_date_string)).days / 30.4)

	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the return for the stock in the given time frame
	def calculate_return(self, ticker_name, start_date, end_date):
		df = self.ticker_histories[ticker_name]
		after_start_date = df["Date"] >= start_date
		before_end_date = df["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_dates = df.loc[between_two_dates]

		end_price = filtered_dates.iloc[-1]["Close"]
		start_price = filtered_dates.iloc[0]["Close"]

		return_percentage = (end_price - start_price) / (start_price) * 100.0

		return return_percentage

	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the CAGR for the stock in the given time frame
	def calculate_curr_per_CAGR(self, ticker_name, start_date, end_date):
		
		if end_date < self.date_to_string(self.ticker_starts[ticker_name]):
			return 0
		
		return_percentage = float(self.calculate_return(ticker_name, start_date, end_date))
		d0 = self.get_date(start_date)
		d1 = self.get_date(end_date)
		delta = d1 - d0
		year_frac = float(delta.days / 365.25)

		cagr = ((return_percentage / 100) + 1) ** (1 / year_frac) - 1

		return cagr

	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the CAGR for the stock in the time frame after the given time frame
	def calculate_next_given_per_CAGR(self, ticker_name, start_date, num_days):
		
		delta = datetime.timedelta(days = num_days)
		d1 = self.get_date(start_date)
		end_date = self.date_to_string(d1 + delta)

		if end_date < self.date_to_string(self.ticker_starts[ticker_name]):
			return 0

		return_percentage = float(self.calculate_return(ticker_name, start_date, end_date))
		year_frac = float(num_days / 365.25)
		next_given_per_cagr = ((return_percentage / 100) + 1) ** (1 / year_frac) - 1

		return next_given_per_cagr


	# REQUIRES: valid start and end dates in form YYYY-MM-DD
	# RETURNS:  the sharpe ratio of a given stock prior to the end date
	def risk_free_rate(self, start_date, end_date):
		risk_free_rate = 0
		after_start_date = self.rfr["Date"] >= start_date
		before_end_date = self.rfr["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_dates = self.rfr.loc[between_two_dates]

		risk_free_rate = filtered_dates[["Close"]].mean()

		return risk_free_rate


	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the sharpe ratio of a given stock prior to the end date
	def calculate_sharpe_ratio(self, ticker_name, start_date, end_date):

		if end_date < self.date_to_string(self.ticker_starts[ticker_name]):
			return 0

		df = self.ticker_histories[ticker_name]
		after_start_date = df["Date"] >= start_date
		before_end_date = df["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_dates = df.loc[between_two_dates] # can prob be improved

		# periods could be 1 here
		# but when I do that pandas gives me infinite NaN
		# so it's -1 now :)
		# huh?
		# print(filtered_dates['Close'].diff(periods=1)[1:])
		# print(filtered_dates['Close'].diff(periods=-1)[:-1])

		# print(filtered_dates['Close'].diff(periods=1)[1:])
		# print(filtered_dates['Close'].diff(periods=1)[1:].size)
		# print(filtered_dates['Close'][1:])
		# print(filtered_dates['Close'][1:].size)

		returns = (filtered_dates['Close'].diff(periods=1)[1:] / filtered_dates['Close'][1:]) * 100
		# print(returns)

		return_percentage = self.calculate_return(ticker_name, start_date, end_date)
		stdev = returns.std()
		volatility = stdev * np.sqrt(len(returns))
		rfr = self.risk_free_rate(start_date, end_date)

		sharpe_ratio = (return_percentage - rfr) / volatility
		return sharpe_ratio.iloc[0]

	def calculate_sortino_ratio(self, ticker_name, start_date, end_date):

		if end_date < self.date_to_string(self.ticker_starts[ticker_name]):
			return 0

		df = self.ticker_histories[ticker_name]
		after_start_date = df["Date"] >= start_date
		before_end_date = df["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_dates = df.loc[between_two_dates]
		
		returns = (filtered_dates['Close'].diff(periods=1)[1:] / filtered_dates['Close'][1:]) * 100
		returns = returns[returns < 0]

		#starting_row = filtered_dates.index[0]
		#for i in range(len(filtered_dates['Close']) - 1):
		#	daily_return = (filtered_dates['Close'][starting_row + i + 1] - filtered_dates['Close'][starting_row + i]) / filtered_dates['Close'][starting_row + i] * 100
			
		#	if daily_return < 0:
		#		returns.append(daily_return)
		#		negative_count += 1

		return_percentage = self.calculate_return(ticker_name, start_date, end_date)
		stdev = returns.std()
		volatility = stdev * np.sqrt(len(returns))
		rfr = self.risk_free_rate(start_date, end_date)

		sortino_ratio = (return_percentage - rfr) / volatility

		return sortino_ratio.iloc[0]
		
	# REQUIRES: 2 valid dates as strings in the format YYYY-MM-DD
	# RETURNS:  A csv of the data for all stocks in that time frame
	def calculate_ticker_data_for_timeframe(self, csv_writer, ticker_name, start_date, end_date):
		num_months = self.num_months(start_date, end_date)

		sharpe_values = []
		sortino_values = []
		CAGR_values = []
		date_iterator_start = self.get_date(start_date)
		date_iterator_end = self.month_after_notstring(date_iterator_start)
		month = 0

		while month < num_months:
			sharpe_values.append(self.calculate_sharpe_ratio(ticker_name, self.date_to_string(date_iterator_start), self.date_to_string(date_iterator_end)))
			sortino_values.append(self.calculate_sortino_ratio(ticker_name, self.date_to_string(date_iterator_start), self.date_to_string(date_iterator_end)))
			CAGR_values.append(self.calculate_curr_per_CAGR(ticker_name, self.date_to_string(date_iterator_start), self.date_to_string(date_iterator_end)))
			
			date_iterator_start = date_iterator_end
			date_iterator_end = self.month_after_notstring(date_iterator_end)
			month += 1

		ticker_next_year_CAGR = self.calculate_next_given_per_CAGR(ticker_name, end_date, 365)

		csv_writer.writerow(['y', ticker_name, ticker_next_year_CAGR, start_date, end_date])
		csv_writer.writerow(sharpe_values)
		csv_writer.writerow(sortino_values)
		csv_writer.writerow(CAGR_values)