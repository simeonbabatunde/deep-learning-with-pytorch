import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# Configure device (gpu or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyper parameters
input_size = 784 # channel 1 * width 28 * height 28
hidden_size = 120
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.001

# Load MNIST dataset
train_data = datasets.MNIST(root='./sample_datasets', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='./sample_datasets', train=False, transform=transforms.ToTensor())

# Encapsulate in dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Preview images
# batch = iter(train_loader)
# images, labels = batch.next() 
# print(images.shape, labels.shape)
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i][0], cmap='gray')
# plt.show()

# Design fully connected model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.l2(out)
        return out

model = Model(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images from [100, 1, 28, 28] to [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{num_epochs} step: {i+1}/{n_steps} loss:{loss.item():.4f}')

# Testing
with torch.no_grad():
    n_samples, n_correct = 0, 0
    for images, labels in test_loader:
        # Reshape images from [100, 1, 28, 28] to [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    accuracy = 100.0 * n_correct / n_samples
    print(f'\naccuracy = {accuracy}\n')

