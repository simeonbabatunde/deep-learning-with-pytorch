import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 200
batch_size = 100
learning_rate = 0.001

# Compute mean and std for normalization
# data = datasets.CIFAR10(root='./sample_datasets/junk', train=True, transform=transforms.ToTensor(), download=True)
# print(data[0][0].shape) # [3, 32, 32]
# mean, std = 0, 0
# for img, _ in data:
#     mean += img.mean([1,2])
#     std += img.std([1,2])
# print(mean/len(data), std/len(data)) 
# mean -> [0.4914, 0.4822, 0.4465] std -> [0.2023, 0.1994, 0.2010]

# Custom transform to mean=0 & std=1
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# Load CIFAR10 image datasets 
train_data = datasets.CIFAR10(root='./sample_datasets', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root='./sample_datasets', train=False, transform=transform, download=False)

# Visualize normalized image
# norm_img = np.array(train_data[0][0])
# # Transpose from shape of (3,,) to shape of (,,3)
# norm_img = norm_img.transpose(1,2,0)
# plt.imshow(norm_img)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# Encapsulate in dataloader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# preview some random train samples
# sample = iter(train_loader)
# images, labels = sample.next()
# print(images.shape) # [100, 3, 32, 32]

# Implement CNN model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # 3 chann -> 6 chann, 5x5 filter, img size 32x32 -> 28x28 | out_size = (W - F + 2P)/S + 1
        self.pool1 = nn.MaxPool2d(2 ,2)      # 6 chann, img size 28x28 -> 14X14
        self.conv2 = nn.Conv2d(6, 12, 5)    # 6 chann -> 12 chann, 5x5 filter, img size 14x14 -> 10x10 | out_size = (W - F + 2P)/S + 1
        self.pool2 = nn.MaxPool2d(2 ,2)      # 12 chann img size 10x10 -> 5X5
        self.fc1 = nn.Linear(12*5*5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 12*5*5) # Flatten 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
n_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{num_epochs} step: {i+1}/{n_steps} loss: {loss.item():.4f}')

print('Training Completed')

with torch.no_grad():
    n_samples, n_correct = 0, 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (max value, index)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples
    print(f'\nAccuracy: {accuracy} %\n')

     

