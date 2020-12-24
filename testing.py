from calculate import *
import time

calc = Calculator('stock_tickers.csv')
# print(calc.get_num())
# print(calc.get_names())
# print(calc.test('FB'))

# print(calc.day_after('2018-07-10'))
# print(calc.day_after('2013-12-31'))

# print("Calculating Sharpe: ")
# print(calc.calculate_sharpe_ratio('FB', '2019-08-14', '2020-01-03'))

# start_time = time.time()
# print("Calculating Sortino: ")
# print(calc.calculate_sortino_ratio('DSX', '2019-08-15', '2019-09-15'))
# print(time.time() - start_time, "seconds to generate sortino data")

# print(calc.next_market_day('2020-12-20'))
# print(calc.next_per_end('2019-08-14', '2020-01-03'))

# print(calc.calculate_return('AMZN', '2019-08-14', '2020-01-03'))
# print(calc.calculate_return('AMZN', '2020-01-03', '2020-05-26'))
# print(calc.calculate_curr_per_CAGR('AMZN', '2019-08-14', '2020-01-07'))
# print(calc.calculate_next_mirror_per_CAGR('AMZN', '2019-08-14', '2020-01-07'))
# print(calc.calculate_next_given_per_CAGR('LX', '2019-08-14', 365))

# print("Generating Data: ")
# calc.calculate_all_data_for_timeframe('2019-08-14', '2020-01-07')

# start_date = calc.get_date('2019-01-01')
# print(calc.calculate_sharpe_ratio('GVA', start_date, '2020-01-01'))

start_time = time.time()

with open("testing_data.csv", "w") as data_file:
    writer = csv.writer(data_file)
    calc.calculate_ticker_data_for_timeframe(writer,'WHG','2000-01-10', '2005-01-10')   

print(time.time() - start_time, "seconds to generate sortino data")

# print(calc.get_ticker_starts())