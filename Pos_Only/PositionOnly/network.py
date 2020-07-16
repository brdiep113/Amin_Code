'''
Builds up the structure of the network inspired by Unsuperpoint 
https://arxiv.org/pdf/1907.04011.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# define backbone block
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # four pairs of convolution layers
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)   # if having R, 4 input channels
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 8*8, 3, 1, 1)  # 8 = downsapling scale
        self.bn6 = nn.BatchNorm2d(64)
        self.pixel_shuffle = nn.PixelShuffle(8)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(F.max_pool2d(self.bn1(self.conv2(x)), 2, stride=2))
        
        x = F.leaky_relu(self.bn2(self.conv3(x)))
        x = F.leaky_relu(F.max_pool2d(self.bn2(self.conv4(x)), 2, stride=2))

        x = F.leaky_relu(self.bn3(self.conv5(x)))
        x = F.leaky_relu(F.max_pool2d(self.bn3(self.conv6(x)), 2, stride=2))

        x = F.leaky_relu(self.bn4(self.conv7(x)))
        x = F.leaky_relu(self.bn4(self.conv8(x)))

        x = F.leaky_relu(self.bn5(self.conv9(x)))
        x = F.sigmoid(self.bn6(self.conv10(x)))  # also we can use softmax
        # reshape
        x = self.pixel_shuffle(x)

        return x



        