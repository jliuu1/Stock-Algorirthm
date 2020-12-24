# generating the data used for the machine learning model

from calculate import *
from tqdm import tqdm
import time

start_time = time.time()

calc = Calculator('stock_tickers.csv')

year = 2000
month = '01'
day = '02'       # 01 is New Year's Day, market's aren't open

with open("practice_data.csv", "w") as data_file:
	
    writer = csv.writer(data_file)
    writer.writerow(['Ticker', 'Sharpe', 'Sortino', 'Current Period CAGR', 'Next Year CAGR'])

    for i in tqdm(range(calc.num_stocks)):
        
        ticker = calc.ticker_names[i]
        first_listing = calc.ticker_starts[ticker]

        for i in range(14):
            start_date = str(year + i) + "-" + month + "-" + day
            end_date = str(year + i + 5) + "-" + month + "-" + day

            if first_listing < start_date:
                try:
                    calc.calculate_ticker_data_for_timeframe(writer,ticker, start_date, end_date)
                except:
                    continue


print(time.time() - start_time, "seconds to generate data for", calc.get_num(), "stocks")
