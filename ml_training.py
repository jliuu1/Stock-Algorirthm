import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ml_data_analysis import Conversion


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#we're gonna input a 3 column data set so n x 3 where n starts from 5? 10? 15? years ago when the stocks began? and then then if it began later subsitute zeros in.
		# self.conv1 = nn.Conv2d(1, 64, 2)
		# self.conv2 = nn.Conv2d(64, 128, 2)
		# self.conv3 = nn.Conv2d(128, 256, 2)

		# x = torch.randn(60, 3).view(-1, 1, 3, 60)
		# self._to_linear = None
		# self.convs(x)

		self.fc1 = nn.Linear(3 * 60, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 128)
		self.fc4 = nn.Linear(128, 5)
	
	# def convs(self, x):
	# 	x = F.max_pool2d(F.relu(self.conv1(x)), 2)
	# 	x = F.max_pool2d(F.relu(self.conv2(x)), 2)
	# 	x = F.max_pool2d(F.relu(self.conv3(x)), 2)

	# 	if self._to_linear is None:
	# 		self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
	# 	return x

	def forward(self, x):
		#x = self.convs(x)
		#x = x.view(-1, self._to_linear)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return F.log_softmax(x, dim=1)

net = Net()
# print(net)

#OPTIMIZATION STEP

optimizer = optim.Adam(net.parameters(), lr=1e-6) #some use 1e-6
loss_function = nn.MSELoss()
c = Conversion('practice.csv', True)
train_X, train_y = c.get_training_data(0.25)
test_X, test_y = c.get_testing_data(0.25)


BATCH_SIZE = 1 #number of stocks we run each time
EPOCHS = 5 # how many times we run through the training data in general

for epoch in range(EPOCHS):
	for i in range(0, c.get_data_len(), BATCH_SIZE):

		batch_X = train_X[i:i+BATCH_SIZE].view(-1, 180)
		batch_y = train_y[i:i+BATCH_SIZE]

		net.zero_grad()

		outputs = net(batch_X)
		loss = loss_function(outputs, batch_y)
		loss.backward()
		optimizer.step()

	print(f"Epoch: {epoch}. Loss: {loss}")


correct = 0
correct_short = 0
correct_bad = 0
correct_neutral = 0
correct_good = 0
correct_long = 0
total = 0

with torch.no_grad():
	for i in range(len(test_X)):
		real_class = torch.argmax(test_y[i])
		net_out = net(test_X[i].view(-1, 180))
		predicted_class = torch.argmax(net_out)

		# print(predicted_class, real_class)
		if predicted_class == real_class:
			correct += 1
			# if torch.equal(predicted_class, torch.Tensor(np.eye(5)[0])):
			# 	correct_short += 1
			# if torch.equal(predicted_class, torch.Tensor(np.eye(5)[1])):
			# 	correct_bad += 1
			# if torch.equal(predicted_class, torch.Tensor(np.eye(5)[2])):
			# 	correct_neutral += 1
			# if torch.equal(predicted_class, torch.Tensor(np.eye(5)[3])):
			# 	correct_good += 1
			# if torch.equal(predicted_class, torch.Tensor(np.eye(5)[4])):
			# 	correct_long += 1
		total += 1

print("Accuracy: ", round(correct/total, 3))
print(correct_short , correct_bad, correct_neutral, correct_good, correct_long)