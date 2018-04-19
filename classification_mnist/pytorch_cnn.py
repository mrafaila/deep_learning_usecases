import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import time

# Hyper Parameters
num_epochs = 2
batch_size = 128
#learning_rate = 0.01

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/mnist/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/mnist/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        filters_layer1 = 32
        filters_layer2 = 64
        dropout = 0.5
        kernel_size = (5,5)
        pool_size = (2,2)

        units_dense = 1024
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, filters_layer1, kernel_size=kernel_size[0], padding=2),
            nn.BatchNorm2d(filters_layer1),
            nn.ReLU(),
            nn.MaxPool2d(pool_size[0])
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(filters_layer1, filters_layer2, kernel_size=kernel_size[0], padding=2),
            nn.BatchNorm2d(filters_layer2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size[0]),
        )
        self.fc1 = nn.Linear(7*7*filters_layer2, units_dense)
        self.dropout = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(units_dense, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
        
cnn = CNN()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters())#, lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
    
    print(time.time()-start)
    print('sec per epoch')
    
    # Test the Model
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    start = time.time()
    duration = time.time()-start
    print(len(images)/duration)
    print('predicted images per sec')
    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')