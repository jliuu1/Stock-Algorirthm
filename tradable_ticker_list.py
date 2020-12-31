import csv

with open('stock_raw_list.csv', 'r') as stocks_file, \
        open('stock_tickers.csv', 'w') as ticker_file:
        
    stocks = csv.reader(stocks_file, delimiter=',')
    writer = csv.writer(ticker_file)

    # shrinking list of stocks
    stockfilter = 5
    count = 1

    for row in stocks:
        if (count % stockfilter) == 0:
            if row[2] != 'NA' and row[0] != 'Symbol' and '/' not in row[0] and len(row[0]) < 5:
                writer.writerow([row[0]])
        count += 1
                