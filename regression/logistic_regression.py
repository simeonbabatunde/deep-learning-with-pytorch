from scipy.sparse.construct import rand
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess breast cancer binary classification dataset
bc_data = datasets.load_breast_cancer()
x, y = bc_data.data, bc_data.target

# Scale features to have mean 0 and variance 1
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

n_samples, n_features = x.shape
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=123)

# Convert from numpy to tensor
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
# Reshape target into column vector
y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)

# Design logistic regression model, with sigmoid as the last layer
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)

# Construct loss function(BCE->Binary Cross Entropy) and optimizer(SGD)
learning_rate = 0.01
loss_func = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Model training
num_epochs = 200
for epoch in range(num_epochs):
    # Forward pass and loss
    y_p = model(x_train)
    loss = loss_func(y_p, y_train)

    # Backward pass and weights update
    loss.backward()
    optimizer.step()

    # Zero out gradients
    optimizer.zero_grad()

    if(epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Evaluate model accuracy without updating computational graph
with torch.no_grad():
    prediction_prob = model(x_test)
    predicted_class = prediction_prob.round()
    accuracy = predicted_class.eq(y_test).sum() / float(y_test.shape[0])

    print(f'\naccuracy = {accuracy.item():.4f}\n')