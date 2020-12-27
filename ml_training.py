import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n = 1000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #we're gonna input a 3 column data set so n x 3 where n starts from 5? 10? 15? years ago when the stocks began? and then then if it began later subsitute zeros in.
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)

        x = torch.randn(n, 3).view(-1, 1, n, 3)
        x = torch.flatten(x, 1)
        self._to_linear = x.size()[1]

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
    
    
    def foward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

net = Net()
print(net)

#OPTIMIZATION STEP

optmizer = optim.Adam(net.parameters(), lr=1e-3) #some use 1e-6
loss_function = nn.MSELoss()

# some data compiling thing where X is data and y is answer
# were going to have to store our data in a x * n * 3 tensor where x is number of stocks
#for the tesnor of 5/10 years. we gotta move it around so like some start at the v start, some v end, etc
# X stands for data
VAL_PCT = .25
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]
print(len(train_X), len(test_X))


BATCH_SIZE = 50 #number of stocks we run each time
EPOCHS = 5 # how many times we run through the training data in general

for epoch in range(EPOCHS):
    for i in range(0, len(train_X), BATCH_SIZE):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, n, 3)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}. Loss: {loss}")
