import csv

with open('stock_raw_list.csv', 'r') as stocks_file, \
        open('stock_tickers.csv', 'w') as ticker_file:
        
    stocks = csv.reader(stocks_file, delimiter=',')
    writer = csv.writer(ticker_file)

    # shrinking list of stocks
    stockfilter = 1
    count = 0

    for row in stocks:
        if (count % stockfilter) == 0:
            if row[2] != 'NA' and row[0] != 'Symbol':
                writer.writerow([row[0]])
        count += 1
                