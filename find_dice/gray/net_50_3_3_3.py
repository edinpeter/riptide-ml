import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        self.conv1_kernel = 5
        self.conv2_kernel = 3
        self.conv3_kernel = 3
        self.conv4_kernel = 3

        self.network_width = 80 #out channels
        self.conv2_output_channels = 18
        self.conv3_output_channels = 20
        self.conv4_output_channels = 20

        self.outputs = 6

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, self.network_width, self.conv1_kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.network_width, self.conv2_output_channels, self.conv2_kernel)
        self.conv3 = nn.Conv2d(self.conv2_output_channels, self.conv3_output_channels , self.conv3_kernel)
        self.conv4 = nn.Conv2d(self.conv3_output_channels, self.conv4_output_channels, self.conv4_kernel)

        #self.fc1 = nn.Linear(20, 120)
        self.fc2 = nn.Linear(20, self.outputs)

    def forward(self, x):
        #print x
        x = self.pool(F.relu(self.conv1(x)))
        #print x
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        #print x
        x = x.view(-1, 20)
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
