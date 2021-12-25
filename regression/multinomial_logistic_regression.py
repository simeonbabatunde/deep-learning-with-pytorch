import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess iris dataset
iris_data = datasets.load_iris()
x, y = iris_data.data, iris_data.target
target_names, feature_names = iris_data.target_names, iris_data.feature_names

# Scale data to have mean 0 and variance 1
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# split data into training and test
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=123)

# Convert numpyt to tensor
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
y_test = torch.from_numpy(y_test.astype(np.int64))

# Design multinomial logistic regression model with softmax as the last layer
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 30)
        self.layer2 = nn.Linear(30, 30)
        self.layer3 = nn.Linear(30, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=1)
        return x

model = Model(x_train.shape[1], 3)

# Construct loss function(Cross Entropy Loss) and optimizer(Adam)
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Model training
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(x_train)
    loss = loss_func(y_pred, y_train)

    # Backward pass and weights update
    loss.backward()
    optimizer.step()

    # Zero out gradients
    optimizer.zero_grad()

    if(epoch+1) % 50 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    prediction = model(x_test)
    accuracy = (torch.argmax(prediction, dim=1) == y_test).type(torch.FloatTensor).mean()  
    print(f'\naccuracy = {accuracy.item():.4f}\n')
