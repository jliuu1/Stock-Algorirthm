from calculate import *
import time

calc = Calculator('stock_tickers.csv')

print("Calculating Sharpe: ")
print(calc.calculate_sharpe_ratio('LX', '2019-08-14', '2020-01-03'))

# print("Calculating Sortino: ")
# print(calc.calculate_sortino_ratio('LX', '2019-08-14', '2020-01-03'))

# start_time = time.time()
# print("Calculating Sortino: ")
# print(calc.calculate_sortino_ratio('DSX', '2019-08-15', '2019-09-15'))
# print(time.time() - start_time, "seconds to generate sortino data")

# print("Generating Data: ")
# calc.calculate_all_data_for_timeframe('2019-08-14', '2020-01-07')

# start_date = calc.get_date('2019-01-01')
# print(calc.calculate_sharpe_ratio('GVA', start_date, '2020-01-01'))

# start_time = time.time()

# with open("testing_data.csv", "w") as data_file:
#     writer = csv.writer(data_file)
#     calc.calculate_ticker_data_for_timeframe(writer,'WHG','2000-01-10', '2005-01-10')   

# print(time.time() - start_time, "seconds to generate data")

# print(calc.get_ticker_starts())