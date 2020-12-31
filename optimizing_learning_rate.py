import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ml_data_analysis import Conversion
from ml_training import Net

# net = Net()

learning_rate_accuracy = dict()
learning_rates = [8e-8, 9e-8, 1e-7, 2e-7, 3e-7,
                  4e-7, 5e-7, 6e-7, 7e-7, 8e-7,
                  9e-7, 1e-6, 2e-6, 3e-6, 4e-6,
                  5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 
                  1e-5, 2e-5, 3e-5, 4e-5, 5e-5]

for k in tqdm(range(len(learning_rates))):

    # OPTIMIZATION STEP
    optimizer = optim.Adam(net.parameters(), lr=learning_rates[k])
    loss_function = nn.MSELoss()

    accuracies = []

    for i in range(10):

        for epoch in range(EPOCHS):
            for i in range(0, len(train_X), BATCH_SIZE):
            
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 8)
                batch_y = train_y[i:i+BATCH_SIZE]

                net.zero_grad()
                
                outputs = net(batch_X)
                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # print(f"Epoch: {epoch + 1}. Loss: {loss}")


        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(test_X)):
                real_class = torch.argmax(test_y[i])
                net_out = net(test_X[i].view(-1, 8))
                predicted_class = torch.argmax(net_out)

                if predicted_class == real_class:
                    correct += 1
                total += 1

        accuracy = round(correct/total, 3)
        # print("Accuracy: ", accuracy)

        accuracies.append(accuracy)

    learning_rate_accuracy[learning_rates[k]] = (sum(accuracies) / len(accuracies))

learning_rate_rankings = dict(sorted(learning_rate_accuracy.items(), key=lambda item: item[1]))

print(learning_rate_rankings)