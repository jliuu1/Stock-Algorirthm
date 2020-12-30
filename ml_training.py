import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ml_data_analysis import Conversion


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#we're gonna input a 3 column data set so n x 3 where n starts from 5? 10? 15? years ago when the stocks began? and then then if it began later subsitute zeros in.
		self.conv1 = nn.Conv2d(1, 64, 3)
		self.conv2 = nn.Conv2d(64, 128, 3)
		self.conv3 = nn.Conv2d(128, 256, 3)

		x = torch.randn(60, 3).view(-1, 1, 3, 60)
		self._to_linear = None
		self.convs(x)

		self.fc1 = nn.Linear(self._to_linear, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 5)
	
	def convs(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = F.max_pool2d(F.relu(self.conv3(x)), 2)

		if self._to_linear is None:
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
		return x

	def foward(self, x):
		x = self.convs(x)
		x = x.view(-1, self._to_linear)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return F.log_softmax(x, dim=1)

net = Net()
print(net)

#OPTIMIZATION STEP

optimizer = optim.Adam(net.parameters(), lr=1e-3) #some use 1e-6
loss_function = nn.MSELoss()
c = Conversion('practice.csv', True)
batch_X, batch_y = c.get_training_data(.25)


BATCH_SIZE = 50 #number of stocks we run each time
EPOCHS = 5 # how many times we run through the training data in general

for epoch in range(EPOCHS):
	for i in range(0, c.get_data_len(), BATCH_SIZE):
		net.zero_grad()

		outputs = net(batch_X)
		loss = loss_function(outputs, batch_y)
		loss.backward()
		optimizer.step()
	print(f"Epoch: {epoch}. Loss: {loss}")
