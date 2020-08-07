'''
Builds up the structure of the network inspired by SuperPoint
https://arxiv.org/abs/1712.07629
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_features=5):
        super(Model, self).__init__()
        # four pairs of convolution layers
        self.conv1 = nn.Conv2d(num_features, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(128)

        # position module
        self.conv9 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 8 * 8 + 1, 1, 1, 0)  # 8=scale & 1=dustbin
        self.bn10 = nn.BatchNorm2d(65)

        self.pixel_shuffle = nn.PixelShuffle(8)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2, stride=2))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)), 2, stride=2))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(F.max_pool2d(self.bn6(self.conv6(x)), 2, stride=2))

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        x = F.relu(self.bn9(self.conv9(x)))
        x = self.bn10(self.conv10(x))
        
        prob = torch.softmax(x, 1)                        # channel-wise softmax
        prob = prob[:, :-1, :, :]                         # removes dustbin dim.
        prob = self.pixel_shuffle(prob)
        prob = torch.squeeze(prob, 1)

        return {'logits' : x, 'prob' : prob}