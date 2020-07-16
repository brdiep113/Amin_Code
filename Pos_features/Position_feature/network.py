'''
Builds up the structure of the network inspired by Unsuperpoint 
https://arxiv.org/pdf/1907.04011.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# define map2xy function, considering the fact that the origin is bottom left &
# [0,0]
# def map2xy(cr_rel):
#     grid_c, grid_r = torch.meshgrid(torch.arange(cr_rel.size()[2]-1,-1,-1),
#                                                  torch.arange(cr_rel.size()[3]))
#     grid_cr = torch.stack([grid_c,grid_r],dim=0) 
#     grid_cr.repeat(cr_rel.size()[0],1,1,1)
#     xy = (grid_cr + cr_rel) * 4    # downsample scale=4 if differs, change it
#     return xy


# define backbone block
class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
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

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(F.max_pool2d(self.bn1(self.conv2(x)), 2, stride=2))
        
        x = F.leaky_relu(self.bn2(self.conv3(x)))
        x = F.leaky_relu(F.max_pool2d(self.bn2(self.conv4(x)), 2, stride=2))

        x = F.leaky_relu(self.bn3(self.conv5(x)))
        x = F.leaky_relu(F.max_pool2d(self.bn3(self.conv6(x)), 2, stride=2))

        x = F.leaky_relu(self.bn4(self.conv7(x)))
        x = F.leaky_relu(self.bn4(self.conv8(x)))

        return x

# define Position XY module
class PositionXY(nn.Module):

    def __init__(self):
        super(PositionXY, self).__init__()
        
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 8*8, 3, 1, 1)  # 8 = downsapling scale
        self.bn2 = nn.BatchNorm2d(64)
        self.pixel_shuffle = nn.PixelShuffle(8)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.sigmoid(self.bn2(self.conv2(x)))  # also we can use softmax
        # reshape
        x = self.pixel_shuffle(x)

        return x


class Feature(nn.Module):

    def __init__(self):
        super(Feature, self).__init__()

        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 16, 3, 1, 1)  # 16 is length of feature vectors
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.sigmoid(self.bn2(self.conv2(x)))
        # interpolate
        x = F.interpolate(x, scale_factor=8.0, mode='nearest')  # or 'bicubic'
        
        # ensure the output is binary (0|1), maybe we'd better use sigmoid 
        # or L2 norm (SuperPoint)
        # th = torch.Tensor([0.5])  # threshold
        # x = (x > th).float() 

        return x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.backbone = Backbone()
        self.position = PositionXY()
        self.feature = Feature()

    def forward(self, x):
        x = self.backbone(x)
        p = self.position(x)
        f = self.feature(x)

        return p, f


        