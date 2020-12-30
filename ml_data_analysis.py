import csv
import numpy as np
import time
import torch
from tqdm import tqdm

class Conversion:
	def __init__(self, filename, shuffle):
		self.LABELS = {"Short": -2, "Bad": -1, "Neutral": 0, "Good": 1, "Long": 2}
		self.training_data = []
		self.short_count = 0             # -50%+ CAGR
		self.bad_count = 0               # -50% - -20% CAGR
		self.neutral_count = 0           # -20% - 20% CAGR
		self.good_count = 0              # 20% - 50% CAGR
		self.long_count = 0              # 50%+ CAGR

		self.data_count = 0
		self.rows_read = 0
		self.data_length = None

		start_time = time.time()
		with open(filename, 'r') as data_file:
			data = csv.reader(data_file)
			for row in data:
				if row[0] == "y":
					if float(row[2]) <= -.5:
						self.short_count += 1
						y = np.eye(5)[0]
					elif float(row[2]) <= -.2:
						self.bad_count += 1
						y = np.eye(5)[1]
					elif float(row[2]) <= .2:
						self.neutral_count += 1
						y = np.eye(5)[2]
					elif float(row[2]) <= .5:
						self.good_count += 1
						y = np.eye(5)[3]
					else:
						self.long_count += 1
						y = np.eye(5)[4]

					#X = [next(data), next(data), next(data)]
					X = next(data) + next(data) + next(data)

					if self.data_length is None:
						self.data_length = len(X[1])

					self.training_data.append([X, y])
					self.data_count += 1

		print("Short: ", self.short_count)
		print("Bad: ", self.bad_count)
		print("Neutral: ", self.neutral_count)
		print("Good: ", self.good_count)
		print("Long: ", self.long_count)

		if shuffle is True:
			np.random.shuffle(self.training_data)

		self.tensor_data = torch.Tensor([[float(k) for k in i[0]] for i in self.training_data])
		self.tensor_data = self.tensor_data / 1e10
		#self.tensor_data = torch.Tensor([[[float(k) for k in j] for j in i[0]] for i in self.training_data])
		#print(self.tensor_data[0])
		self.tensor_outputs = torch.Tensor([i[1] for i in self.training_data])

		print(time.time() - start_time, "seconds to generate data for", self.data_count, "data_points")

	def get_data_length(self):
		return self.data_length

	def data_total(self):
		return self.data_count
	
	def get_data_len(self):
		return len(self.tensor_data)

	def get_training_data(self, VAL_PCT):
		val_size = int(len(self.tensor_outputs) * VAL_PCT)
		train_X = self.tensor_data[:-val_size]
		train_y = self.tensor_outputs[:-val_size]
		return train_X, train_y
	
	def get_testing_data(self, VAL_PCT):
		val_size = int(len(self.tensor_outputs) * VAL_PCT)
		test_X = self.tensor_data[-val_size:]
		test_y = self.tensor_outputs[-val_size:]
		return test_X, test_y

c = Conversion("practice.csv", True)