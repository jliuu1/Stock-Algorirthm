import csv
from tqdm import tqdm 
import numpy as np
import time

LABELS = {"Short": -2, "Bad": -1, "Neutral": 0, "Good": 1, "Long": 2}
training_data = []
short_count = 0             # -50%+ CAGR
bad_count = 0               # -50% - -20% CAGR
neutral_count = 0           # -20% - 20% CAGR
good_count = 0              # 20% - 50% CAGR
long_count = 0              # 50%+ CAGR

data_count = 0
rows_read = 0

start_time = time.time()

with open("testing_data.csv", "r") as data_file:
    data_reader = csv.reader(data_file)

    for row in data_reader:

        if row[0] == "y":
            if float(row[2]) <= -.5:
                short_count += 1
                y = np.eye(5)[0]
            elif float(row[2]) <= -.2:
                bad_count += 1
                y = np.eye(5)[1]
            elif float(row[2]) <= .2:
                neutral_count += 1
                y = np.eye(5)[2]
            elif float(row[2]) <= .5:
                good_count += 1
                y = np.eye(5)[3]
            else:
                long_count += 1
                y = np.eye(5)[4]

            X = [next(data_reader), next(data_reader), next(data_reader)]

            training_data.append([X, y])
            data_count += 1

print(training_data[0][1])
print(time.time() - start_time, "seconds to generate data for", data_count, "data_points")

print("Short: ", short_count)
print("Bad: ", bad_count)
print("Neutral: ", neutral_count)
print("Good: ", good_count)
print("Long: ", long_count)

