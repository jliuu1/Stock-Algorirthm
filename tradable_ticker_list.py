import csv

with open('stock_raw_list.csv', 'r') as stocks_file, \
        open('stock_tickers.csv', 'w') as ticker_file:
        
    stocks = csv.reader(stocks_file, delimiter=',')
    writer = csv.writer(ticker_file)

    for row in stocks:
        if row[2] != 'NA' and row[0] != 'Symbol':
            writer.writerow([row[0]])
                