
from calculate import *

calc = Calculator('stock_tickers.csv')
# print(calc.get_num())
# print(calc.get_names())
# print(calc.test('FB'))

# print(calc.day_after('2018-07-10'))
# print(calc.day_after('2013-12-31'))

print(calc.calculate_return('FB', '2018-07-12', '2019-01-03'))