import torch
import torch.nn as nn
from torch.nn import init

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # The shape of input images is (3*224*224)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5))
        # now the shape is (64*220*220)
        self.bn1 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # now the shape is (64*220*220)
        self.r1 = nn.ReLU(inplace=True)
        self.p1 = nn.MaxPool2d(kernel_size=(2, 2))
        # now the shape is (64*219*219)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        # now the shape is (64*217*217)
        self.bn2 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # now the shape is (64*217*217)
        self.r2 = nn.ReLU(inplace=True)
        self.p2 = nn.MaxPool2d(kernel_size=(2, 2))
        # now the shape is (64*216*216)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3))
        # now the shape is (32*214*214)
        self.bn3 = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # now the shape is (32*214*214)
        self.r3 = nn.ReLU(inplace=True)
        self.p3 = nn.MaxPool2d(kernel_size=(2, 2))
        # now the shape is (32*213*213)

        self.conv4 = nn.Conv2d(32, 16, kernel_size=(3, 3))
        # now the shape is (16*211*211)
        self.bn4 = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # now the shape is (16*211*211)
        self.r4 = nn.ReLU(inplace=True)
        self.p4 = nn.MaxPool2d(kernel_size=(2, 2))
        # now the shape is (16*210*210)

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # now the shape is (16)
        self.fc1 = nn.Linear(in_features=16, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=5)

        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform(param)
            if 'bias' in name:
                init.xavier_uniform(param)

    def forward(self, x):
        # The shape of x is (batch_size, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.p1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.r2(x)
        x = self.p2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.r3(x)
        x = self.p3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.r4(x)
        x = self.p4(x)

        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
