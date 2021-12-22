import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Generate and prepare regression datasets
x_np, y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# Convert dataset from numpy to tensor
x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
# Reshape y from a single row to column vector. Use any of the two methods
# y = y.unsqueeze(1)
y = y.view(y.shape[0], 1)
# Extract numner of samples and features
n_samples, n_features = x.shape

# Design linear regression model (1-layer)
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Construct loss function(MSE) and optimizer(SGD)
learning_rate = 0.01
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Model training loop
num_epochs = 200
for epoch in range(num_epochs):
    # Forward pass and loss 
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    # Backward pass
    loss.backward()
    # Update weights and reset grads value
    optimizer.step()
    optimizer.zero_grad()

    # Print training summary
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

# detach final prediction from computational graph i.e. grads=False
prediction = model(x).detach().numpy()

# Make plot 
plt.plot(x_np, y_np, 'ro')
plt.plot(x_np, prediction, 'b')
plt.show()