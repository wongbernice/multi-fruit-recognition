import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # Input 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # Input 
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2) # Pooling

        # Define fully connected layers
        self.fc1 = nn.Linear(256 * 17 * 17, 512)
        self.fc2 = nn.Linear(512, 10)

        # Define dropout
        self.dropout = torch.nn.Dropout(0.5) # dropout with rate of 0.5

        # Define batchnorm
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.bn1_2 = nn.BatchNorm2d(32)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.bn4_2 = nn.BatchNorm2d(256)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.relu(self.bn1_2(self.conv1_2(X)))
        X = self.pool(X) 

        X = F.relu(self.bn2(self.conv2(X)))
        X = F.relu(self.bn2_2(self.conv2_2(X)))
        X = self.pool(X) 

        X = F.relu(self.bn3(self.conv3(X)))
        X = F.relu(self.bn3_2(self.conv3_2(X)))
        X = self.pool(X) 

        X = F.relu(self.bn4(self.conv4(X)))
        X = F.relu(self.bn4_2(self.conv4_2(X)))
        X = self.pool(X) 

        # Flatten
        X = X.view(X.size(0), -1)

        X = F.relu(self.fc1(X))
        X = self.dropout(X) # dropout

        X = self.fc2(X)

        return X