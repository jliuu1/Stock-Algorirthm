
from calculate import *

calc = Calculator('stock_tickers.csv')
# print(calc.get_num())
# print(calc.get_names())
# print(calc.test('FB'))

# print(calc.day_after('2018-07-10'))
# print(calc.day_after('2013-12-31'))

# print(calc.calculate_return('FB', '2019-08-14', '2020-01-15'))
# print(calc.calculate_CAGR('FB', '2018-07-12', '2019-01-03'))

# print("Calculating Sharpe: ")
# print(calc.calculate_sharpe_ratio('FB', '2019-08-14', '2020-01-03'))

# print("Calculating Sortino: ")
# print(calc.calculate_sortino_ratio('FB', '2019-08-14', '2020-01-03'))

print("Generating Data: ")
calc.calculate_data_for_timeframe('2019-08-14', '2020-01-03')