from calculate import *
import time

calc = Calculator('stock_tickers.csv')

# print("Calculating Sharpe: ")
# print(calc.calculate_sharpe_ratio('LX', '2019-08-14', '2020-01-03'))

# print("Calculating Sortino: ")
# print(calc.calculate_sortino_ratio('AMD', '2019-08-14', '2020-01-03'))

# start_time = time.time()
# print("Calculating Sortino: ")
# print(calc.calculate_sortino_ratio('DSX', '2019-08-15', '2019-09-15'))
# print(time.time() - start_time, "seconds to generate sortino data")

# print("Generating Data: ")
# calc.calculate_all_data_for_timeframe('2019-08-14', '2020-01-07')

# start_date = calc.get_date('2019-01-01')
# print(calc.calculate_sharpe_ratio('GVA', start_date, '2020-01-01'))

# testing SMTI
# print(calc.calculate_return('SMTI', '2000-01-10', '2005-01-10'))

# # 2002-11-10 2002-12-10
# # 2003-02-10 2003-03-10
# # 2003-05-10 2003-06-10

# df = calc.ticker_histories['SMTI']
# print(df)
# after_start_date = df["Date"] >= "2004-04-10"
# before_end_date = df["Date"] <= "2004-05-10"
# between_two_dates = after_start_date & before_end_date
# filtered_dates = df.loc[between_two_dates]

# print(filtered_dates)

# end_price = filtered_dates.iloc[-1]["Close"]
# start_price = filtered_dates.iloc[0]["Close"]

start_time = time.time()

with open("practice.csv", "w") as data_file:
    writer = csv.writer(data_file)
    calc.calculate_ticker_data_for_timeframe(writer,'SMTI','2000-01-10', '2005-01-10')  

# Sharpe Dates Failed (inf, -inf, nan)

# 2002-08-10 2002-09-10
# 2002-09-10 2002-10-10
# 2002-10-10 2002-11-10
# 2002-12-10 2003-01-10
# 2003-01-10 2003-02-10
# 2003-03-10 2003-04-10
# 2003-04-10 2003-05-10
# 2003-07-10 2003-08-10
# 2003-10-10 2003-11-10
# 2004-01-10 2004-02-10
# 2004-02-10 2004-03-10
# 2004-03-10 2004-04-10

# Sortino Dates Failed (inf, -inf, nan)

# 2001-12-10 2002-01-10
# 2002-02-10 2002-03-10
# 2002-04-10 2002-05-10
# 2002-05-10 2002-06-10
# 2002-07-10 2002-08-10
# 2002-08-10 2002-09-10
# 2002-09-10 2002-10-10
# 2002-10-10 2002-11-10
# 2002-12-10 2003-01-10
# 2003-01-10 2003-02-10
# 2003-03-10 2003-04-10
# 2003-04-10 2003-05-10
# 2003-07-10 2003-08-10
# 2003-08-10 2003-09-10
# 2003-09-10 2003-10-10
# 2003-10-10 2003-11-10
# 2004-01-10 2004-02-10
# 2004-02-10 2004-03-10
# 2004-03-10 2004-04-10
# 2004-05-10 2004-06-10
# 2004-06-10 2004-07-10
# 2004-07-10 2004-08-10
# 2004-08-10 2004-09-10

# CAGR Dates > 4096   (more than doubling that month)

# 2000-01-10 2000-02-10
# 2002-06-10 2002-07-10
# 2003-06-10 2003-07-10
# 2004-04-10 2004-05-10

print(time.time() - start_time, "seconds to generate data")


