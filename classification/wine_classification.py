import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math


# Custom class that loads dataset from a given directory
class LoadDataset(Dataset):

    def __init__(self, file_name, delim=","):
        raw_data = np.loadtxt(fname=file_name, delimiter=delim, dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(raw_data[:, 1:])
        self.y = torch.from_numpy((raw_data[:, 0]).astype(np.int64)) - 1 # Adjust class to start from 0 to n_class - 1
        self.n_samples = raw_data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

# Load wine dataset
wine_data = LoadDataset('../sample_datasets/wine.csv')

batch_size = 5
n_iterations = math.ceil(len(wine_data)/batch_size)

# Split data into train and test
train, test = random_split(wine_data, [140, 38], generator=torch.Generator().manual_seed(123))

# Transform to data loader for batching
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=batch_size)

# Design multinomial logistic regression model
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 30)
        self.layer2 = nn.Linear(30, 30)
        self.layer3 = nn.Linear(30, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Instantiate the model (input_size, output_size)
model = Model(wine_data.x.shape[1], 3)

# Construct loss function(Cross Entropy Loss) and optimizer(Adam)
num_epochs = 200
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Model training
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass and loss
        y_pred = model(inputs)
        loss = loss_func(y_pred, labels)
        
        # Backward pass and weights update
        loss.backward()
        optimizer.step()
    
        # Zero out gradients
        optimizer.zero_grad()

        if(i+1) % 25 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_iterations}, loss = {loss.item():.4f}')

with torch.no_grad():
    # Create empty tensors to hold predictions and targets
    temp_pred = torch.tensor([])
    temp_y = torch.tensor([])

    # Eval model with test data
    for _, (x_test, y_test) in enumerate(test_loader):
        prediction = model(x_test)
        temp_pred = torch.cat((temp_pred, prediction))
        temp_y = torch.cat((temp_y, y_test))
        
    accuracy = (torch.argmax(temp_pred, dim=1) == temp_y).type(torch.FloatTensor).mean()  
    print(f'\naccuracy = {accuracy.item():.4f}\n')