import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    dims = None
    classes = None
    def __init__(self, dims, classes):
        self.dims = dims
        self.classes = classes

        if self.dims == 100:

            self.conv1_kernel = 5
            self.conv2_kernel = 5

            self.network_width = 80 #out channels
            self.conv2_output_channels = 18

            self.outputs = len(self.classes)

            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, self.network_width, self.conv1_kernel)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(self.network_width, self.conv2_output_channels, self.conv2_kernel)
            self.conv3 = nn.Conv2d(self.conv2_output_channels, 20, 3)

            self.fc1 = nn.Linear(2000, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, self.outputs)

        elif self.dims == 50:
            self.conv1_kernel = 5
            self.conv2_kernel = 5

            self.network_width = 80 #out channels
            self.conv2_output_channels = 18

            self.outputs = len(self.classes)

            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, self.network_width, self.conv1_kernel)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(self.network_width, self.conv2_output_channels, self.conv2_kernel)
            self.conv3 = nn.Conv2d(self.conv2_output_channels, 20, 3)

            self.fc1 = nn.Linear(180, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, self.outputs)

    def forward(self, x):
        if self.dims == 100:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 20 * 10 * 10)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return x

        elif self.dims == 50:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 180)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return x