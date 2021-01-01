# generating the data used for the machine learning model

from calculate import Calculator, csv, datetime
from tqdm import tqdm
import time

start_time = time.time()

calc = Calculator('stock_tickers.csv')

year = 2000
year_segment = 5
month = '01'
day = '02'                                                  # 01 is New Year's Day, market's aren't open
buffer_per_year = 200
days_buffer = datetime.timedelta(days = buffer_per_year * year_segment)         

with open("training_data.csv", "w") as data_file:
	
    writer = csv.writer(data_file)

    for i in tqdm(range(calc.num_stocks)):
        
        ticker = calc.ticker_names[i]
        first_listing = calc.ticker_starts[ticker]

        for i in range(2020 - year - year_segment):
            start_date = str(year + i) + "-" + month + "-" + day
            end_date = str(year + i + 5) + "-" + month + "-" + day

            # no point for the data if a significant proportion of entries are zeros
            if ((calc.get_date(start_date) + days_buffer) > first_listing):
                try:
                    calc.calculate_ticker_data_for_timeframe(writer, ticker, start_date, end_date)
                except:
                    print(f"{ticker} between {start_date} and {end_date} failed to generate")


print(time.time() - start_time, "seconds to generate data for", calc.get_num(), "stocks")

