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
		self.ticker_starts = {}
		self.num_stocks = 0

		try:
			ticker_file = open(file_name, 'r')
		except:
			print("Invalid File Name")

		ticker_list = csv.reader(ticker_file)

		for ticker in ticker_list:
			self.ticker_names.append(ticker[0])
			ticker_object = yf.Ticker(ticker[0])
			# self.ticker_data[ticker[0]] = ticker_object
			self.ticker_histories[ticker[0]] = Calculator.download_history(ticker_object)
			self.num_stocks += 1
			if self.num_stocks % 100 == 0:
				print('Running stock #' , self.num_stocks)

		self.find_ticker_start()

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
			# os.mkdirs('cache', ) TODO probably good idea to fix this
			with open(filename, 'w') as cache:	
				cache.write(data.to_csv())
				return data

	# REQUIRES: 
	# MODIFIES: Class attribute ticker_times, filling the dict values with the first recorded data in YFinance for the stock
	# RETURNS:
	def find_ticker_start(self):
		for ticker in self.ticker_names:
			# ticker_history = self.ticker_data[ticker].history(period='max')
			ticker_history = self.ticker_history[ticker]
			start_date = pd.to_datetime(ticker_history.index[0])
			self.ticker_starts[ticker] = self.date_to_string(start_date)

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

	# REQUIRES: valid date string in the form YYYY-MM-DD
	# RETURNS:  the datetime object for the next market day
	def next_market_day(self, date_string):
		d0 = self.get_date(date_string)
		increment_day = datetime.timedelta(days=1)
		
		while (d0.weekday() == 5 or d0.weekday() == 6):
			d0 += increment_day

		# ACCOUNT FOR HOLIDAYS
		
		return self.date_to_string(d0)


	# REQUIRES: valid date string in the form YYYY-MM-DD
	# RETURNS:  the datetime object for the next day
	def day_after(self, date_string):
		curr_day = self.get_date(self.next_market_day(date_string))
		increment_day = datetime.timedelta(days=1)
		next_day = curr_day + increment_day

		return self.next_market_day(self.date_to_string(next_day))

	# REQUIRES: valid date string in the form YYYY-MM-DD
	# RETURNS:  the string for the date 1 month after
	def month_after(self, date_string):
		date = datetime.datetime.strptime(date_string, "%Y-%m-%d")
		date += datetime.timedelta(days=calendar.monthrange(date.year,date.month)[1])
		return self.next_market_day(self.date_to_string(date))

	# REQUIRES: 2 valid date strings in the form YYYY-MM-DD
	# RETURNS:  the date string of the period of the same length after end_date
	def next_per_end(self, start_date, end_date):
		d0 = self.get_date(start_date)
		d1 = self.get_date(end_date)
		difference = d1 - d0
		increment_day = datetime.timedelta(days=1)
		today = datetime.date.today()

		if (d1 + difference + increment_day) >= today:
			next_date = today - increment_day
			while (next_date.weekday() == 5 or next_date.weekday() == 6):
				next_date -= increment_day
		else:
			next_date = d1 + difference

		return self.next_market_day(self.date_to_string(next_date))
	
	# REQUIRES: 2 valid dates as strings in the format YYYY-MM-DD
	# RETURNS:  the number of months in that timeframe
	def num_months(self, start_date_string, end_date_string):
		return int((self.get_date(end_date_string) - self.get_date(start_date_string)).days / 30.4)

	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the return for the stock in the given time frame
	def calculate_return(self, ticker_name, start_date, end_date):

		start_price = self.ticker_data[ticker_name].history(start=self.next_market_day(start_date), end=self.day_after(start_date))['Close'].iloc[0]
		end_price = self.ticker_data[ticker_name].history(start=self.next_market_day(end_date), end=self.day_after(end_date))['Close'].iloc[0]
		
		return_percentage = (end_price - start_price) / (start_price) * 100.0

		return return_percentage

	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the CAGR for the stock in the given time frame
	def calculate_curr_per_CAGR(self, ticker_name, start_date, end_date):
		
		if end_date < self.ticker_starts[ticker_name]:
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
	# RETURNS:  the CAGR for the stock in the time frame after the same time frame
	def calculate_next_mirror_per_CAGR(self, ticker_name, start_date, end_date):
		next_per_end = self.next_per_end(start_date, end_date)
		return_percentage = float(self.calculate_return(ticker_name, self.next_market_day(end_date), next_per_end))

		next_period_len = self.get_date(next_per_end) - self.get_date(end_date)
		year_frac = float(next_period_len.days / 365.25)

		next_mirror_per_cagr = ((return_percentage / 100) + 1) ** (1 / year_frac) - 1

		return next_mirror_per_cagr

	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the CAGR for the stock in the time frame after the given time frame
	def calculate_next_given_per_CAGR(self, ticker_name, start_date, num_days):
		delta = datetime.timedelta(days = num_days)
		increment_day = datetime.timedelta(days=1)
		d1 = self.get_date(start_date)
		today = datetime.date.today()

		if (d1 + delta + increment_day) >= today:
			next_date = today - increment_day
			while (next_date.weekday() == 5 or next_date.weekday() == 6):
				next_date -= increment_day
		else:
			next_date = d1 + delta
		end_date = self.next_market_day(self.date_to_string(next_date))

		return_percentage = float(self.calculate_return(ticker_name, self.next_market_day(start_date), end_date))
		year_frac = float(num_days / 365.25)
		next_given_per_cagr = ((return_percentage / 100) + 1) ** (1 / year_frac) - 1

		return next_given_per_cagr


	# REQUIRES: valid start and end dates in form YYYY-MM-DD
	# RETURNS:  the sharpe ratio of a given stock prior to the end date
	def risk_free_rate(self, start_date, end_date):
		ten_year_treasury_bond = yf.Ticker("^TNX")
		risk_free_rate = ten_year_treasury_bond.history(start=self.next_market_day(start_date), end=self.next_market_day(end_date))['Close'].mean()

		return risk_free_rate


	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the sharpe ratio of a given stock prior to the end date
	def calculate_sharpe_ratio(self, ticker_name, start_date, end_date):

		if end_date < self.ticker_starts[ticker_name]:
			return 0

		price_history = self.ticker_data[ticker_name].history(start=self.next_market_day(start_date), end=self.next_market_day(end_date))['Close']

		returns = [0] * (price_history.size - 1)
		for i in range(price_history.size - 1):
			daily_return = (price_history.iloc[i + 1] - price_history.iloc[i]) / price_history.iloc[i] * 100
			returns[i] = daily_return

		pd_returns = pd.Series(returns)
		return_percentage = self.calculate_return(ticker_name, start_date, end_date)
		stdev = pd_returns.std()
		volatility = stdev * np.sqrt(price_history.size)
		rfr = self.risk_free_rate(start_date, end_date)

		sharpe_ratio = (return_percentage - rfr) / volatility

		return sharpe_ratio

	# REQUIRES: valid ticker name that was initialized in the calculator
	#           and limit date (exclusive)
	# RETURNS:  the sortino ratio of a given stock prior to the end date
	def calculate_sortino_ratio(self, ticker_name, start_date, end_date):

		if end_date < self.ticker_starts[ticker_name]:
			return 0

		price_history = self.ticker_data[ticker_name].history(start=self.next_market_day(start_date), end=self.next_market_day(end_date))['Close']

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

		sortino_ratio = (return_percentage - rfr) / volatility

		return sortino_ratio

	# REQUIRES: 2 valid dates as strings in the format YYYY-MM-DD
	# RETURNS:  A csv of the data for all stocks in that time frame
	def calculate_all_data_for_timeframe(self, start_date, end_date):

		# Calculates data for all tickers
		with open("practice_data.csv", "w") as data_file:
			writer = csv.writer(data_file)
			writer.writerow(['Ticker', 'Sharpe', 'Sortino', 'Current Period CAGR', 'Next Year CAGR'])

			for ticker_name in self.ticker_names:
				try:
					ticker_sharpe = self.calculate_sharpe_ratio(ticker_name, start_date, end_date)
					ticker_sortino = self.calculate_sortino_ratio(ticker_name, start_date, end_date)
					ticker_curr_CAGR = self.calculate_curr_per_CAGR(ticker_name, start_date, end_date)
					# ticker_next_mirror_CAGR = self.calculate_next_mirror_per_CAGR(ticker_name, start_date, end_date)
					ticker_next_year_CAGR = self.calculate_next_given_per_CAGR(ticker_name, end_date, 365)
				except:
					continue

				writer.writerow([ticker_name, ticker_sharpe, ticker_sortino, ticker_curr_CAGR, ticker_next_year_CAGR])
		
	# REQUIRES: 2 valid dates as strings in the format YYYY-MM-DD
	# RETURNS:  A csv of the data for all stocks in that time frame
	def calculate_ticker_data_for_timeframe(self, csv_writer, ticker_name, start_date, end_date):

		num_months = self.num_months(start_date, end_date)
		# print("Months:", num_months)

		sharpe_values = []
		sortino_values = []
		CAGR_values = []
		increment_date = datetime.timedelta(days=1)
		date_iterator_start = self.get_date(start_date)
		date_iterator_end = self.get_date(self.month_after(start_date))
		month = 0

		while month != num_months:
			while True:
				try:
					# sharpe_values.append(self.calculate_sharpe_ratio(ticker_name, self.date_to_string(date_iterator_start), self.date_to_string(date_iterator_end)))
					# sortino_values.append(self.calculate_sortino_ratio(ticker_name, self.date_to_string(date_iterator_start), self.date_to_string(date_iterator_end)))
					CAGR_values.append(self.calculate_curr_per_CAGR(ticker_name, self.date_to_string(date_iterator_start), self.date_to_string(date_iterator_end)))
					# ticker_next_mirror_CAGR = self.calculate_next_mirror_per_CAGR(ticker_name, date_iterator_start, date_iterator_end)
					break
				except:	
					print(f'trying next date: {self.date_to_string(date_iterator_start)}')
					date_iterator_start += increment_date
					date_iterator_end += increment_date
			
			date_iterator_start = date_iterator_end
			date_iterator_end = self.get_date(self.month_after(self.date_to_string(date_iterator_end)))
			month += 1

		try:
			ticker_next_year_CAGR = self.calculate_next_given_per_CAGR(ticker_name, end_date, 365)
		except:
			ticker_next_year_CAGR = 0

		csv_writer.writerow([ticker_name, ticker_next_year_CAGR, start_date, end_date])
		# csv_writer.writerow(sharpe_values)
		#csv_writer.writerow(sortino_values)
		csv_writer.writerow(CAGR_values)
			
			



 


