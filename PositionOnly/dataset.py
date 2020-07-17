import glob
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


def generate_heatmap(point_list):
    heatmap = np.zeros((1, 128, 128))
    for point in point_list:
        # our origin in point coords is in bottom left (0,0) & it's cols are XY
        # XY (point[0]&point[1]) are also float and need rounding
        r = 128 - np.round(point[1])
        r = r.astype(int)
        c = np.round(point[0])
        c = c.astype(int)
        heatmap[:, r, c] = 1

    return heatmap


class MyDataset(Dataset):
    def __init__(self, root_path, transforms=None):
        '''
        Args:
            root_path (string): path to the root folder containing all folders
            transform: pytorch transforms for transforms and tensor conversion
        '''
        self.transforms = transforms
        # get the images list
        self.image_list = glob.glob(root_path + '/Image/' + '*')
        # get the points list
        self.point_list = glob.glob(root_path + '/Point_Location/' + '*')

        # calculate length
        self.dataset_length = len(self.image_list)

    def __getitem__(self, index):
        # get image name from the image list
        single_image_path = self.image_list[index]
        # Open image (as a PIL.Image object) & must be converted to tensor
        # TODO: replace Image with skimage
        with Image.open(single_image_path).convert('RGB') as img:
            # convert to numpy, dim = 128x128
            img_as_np = np.array(img) / 255
            # Transform image to tensor, change data type
            img_tensor = torch.from_numpy(img_as_np).float()
            img_tensor = img_tensor.permute(2, 0, 1)
        img.close()

        # get point path from the point list
        single_point_path = self.point_list[index]
        # open the file containing point locations
        with open(single_point_path) as json_file:
            # conve'''
            # Builds up the structure of the network inspired by Unsuperpoint
            # https://arxiv.org/pdf/1907.04011.pdf
            # '''
            #
            # import torch
            # import torch.nn as nn
            # import torch.nn.functional as F
            #
            #
            # class Model(nn.Module):
            #
            #     def __init__(self):
            #         super(Model, self).__init__()
            #         # four pairs of convolution layers
            #         self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)   # if having R, 4 input channels
            #         self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
            #         self.bn1 = nn.BatchNorm2d(32)
            #
            #         self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
            #         self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
            #         self.bn2 = nn.BatchNorm2d(64)
            #
            #         self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            #         self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
            #         self.bn3 = nn.BatchNorm2d(128)
            #
            #         self.conv7 = nn.Conv2d(128, 256, 3, 1, 1)
            #         self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
            #         self.bn4 = nn.BatchNorm2d(256)
            #
            #         self.conv9 = nn.Conv2d(256, 256, 3, 1, 1)
            #         self.bn5 = nn.BatchNorm2d(256)
            #
            #         self.conv10 = nn.Conv2d(256, 8*8, 3, 1, 1)  # 8 = downsapling scale
            #         self.bn6 = nn.BatchNorm2d(64)
            #         self.pixel_shuffle = nn.PixelShuffle(8)
            #
            #     def forward(self, x):
            #         x = F.leaky_relu(self.bn1(self.conv1(x)))
            #         x = F.leaky_relu(F.max_pool2d(self.bn1(self.conv2(x)), 2, stride=2))
            #
            #         x = F.leaky_relu(self.bn2(self.conv3(x)))
            #         x = F.leaky_relu(F.max_pool2d(self.bn2(self.conv4(x)), 2, stride=2))
            #
            #         x = F.leaky_relu(self.bn3(self.conv5(x)))
            #         x = F.leaky_relu(F.max_pool2d(self.bn3(self.conv6(x)), 2, stride=2))
            #
            #         x = F.leaky_relu(self.bn4(self.conv7(x)))
            #         x = F.leaky_relu(self.bn4(self.conv8(x)))
            #
            #         x = F.leaky_relu(self.bn5(self.conv9(x)))
            #         x = F.sigmoid(self.bn6(self.conv10(x)))  # also we can use softmax
            #         # reshape
            #         x = self.pixel_shuffle(x)
            #
            #         return x
            #
            #
            #
            #         rt to numpy array (must be nx2)
            data = json.load(json_file)
            x_pts = np.array((data["X"]))
            y_pts = np.array((data["Y"]))
            points = np.vstack((x_pts, y_pts)).T
            # generate point heatmap from point locations
            point_map = generate_heatmap(points)
            # convert to tensor, change data type
            point_map_tensor = torch.from_numpy(point_map).float()
        json_file.close()

        # Transform image to tensor
        if self.transforms:
            img_tensor = self.transforms(img_tensor)
            point_map_tensor = self.transforms(point_map_tensor)

        # Return image and the label
        return {'image': img_tensor, 'point_map': point_map_tensor}

    def __len__(self):
        return self.dataset_length