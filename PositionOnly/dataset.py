import glob
import json
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset
from skimage.feature import corner_shi_tomasi


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
    def __init__(self, root_path, transforms=None, choose_features=range(5)):
        '''
        Args:
            root_path (string): path to the root folder containing all folders
            transform: pytorch transforms for transforms and tensor conversion
            choose_features (a list of integers): contains feature indices, 
                                                    e.g. [0,1,2,3,4] or 
                                                         [0,1,2] or
                                                         [0,3]
        '''
        self.transforms = transforms
        self.choose_features = choose_features
        # get the images list
        self.image_list = sorted(glob.glob(root_path + '/Image/' + '*.png'))
        # get the features list
        self.feature_list = sorted(glob.glob(root_path + '/4D/' + '*.mat'))
        # get the points list
        self.point_list = sorted(glob.glob(root_path + 
                                 '/Point_Location/' + 
                                 '*.json'))

        # calculate length
        self.dataset_length = len(self.image_list)

    def __getitem__(self, index):
        # get image name from the image list
        single_image_path = self.image_list[index]
        # Open image (as a PIL.Image object) & must be converted to tensor
        with Image.open(single_image_path).convert('RGB') as img:
            # generate Shi-Tomasi response matrix
            imggray = img.convert('L')
            imggray = np.array(imggray)
            response = corner_shi_tomasi(imggray)
            # convert to numpy, dim = 128x128
            response_as_np = np.array(response)
            # Transform image to tensor, change data type
            response_tensor = torch.from_numpy(response_as_np).float()
            response_tensor = response_tensor.unsqueeze(dim=0)
        img.close()

        # get input name from the input list
        single_feature_path = self.feature_list[index]
        # Open input (as a numpy array) & must be converted to tensor
        feat = loadmat(single_feature_path)
        feat = feat['input_4D']
        # Transform input np.array to tensor, change data type
        feat_tensor = torch.from_numpy(feat).float()
        feat_tensor = feat_tensor.permute(2, 0, 1)

        input_tensor = torch.cat([feat_tensor, response_tensor], dim=0)
        input_tensor = input_tensor[self.choose_features, :, :]

        # get point path from the point list
        single_point_path = self.point_list[index]
        # open the file containing point locations
        with open(single_point_path) as json_file:
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
        return {'image': input_tensor, 'point_map': point_map_tensor}

    def __len__(self):
        return self.dataset_length