
import torch
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset


def generate_heatmap(point_list):
    heatmap = np.zeros((1, 128,128))
    for point in point_list: 
        # our origin in point coords is in bottom left (0,0) & it's cols are XY
        # XY (point[0]&point[1]) are also float and need rounding
        r = 128 - np.round(point[1])
        r = r.astype(int) 
        c = np.round(point[0])
        c = c.astype(int)
        heatmap[:, r, c] = 1
    
    return  heatmap


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
        img = Image.open(single_image_path).convert('RGB')
        # Convert to numpy, dim = 1024x1024
        img_as_np = np.array(img)/255
        # Transform image to tensor, change data type
        img_tensor = torch.from_numpy(img_as_np).float()
        img_tensor = img_tensor.permute(2,0,1)
        # get point path from the point list
        single_point_path = self.point_list[index]
        # open points location file 
        point_loc = pd.read_json(single_point_path)
        # convert to numpy array (must be nx2)
        points = np.array(point_loc)
        # generate point heatmap  (PIL.Image type) from point locations
        point_map = generate_heatmap(points)
        # convert to tensor, change data type
        point_map_tensor = torch.from_numpy(point_map).float()  

        # Transform image to tensor
        if self.transforms:
            img_tensor = self.transforms(img_tensor)
            point_map_tensor = self.transforms(point_map_tensor)

        # Return image and the label
        return (img_tensor, point_map_tensor)

    def __len__(self):
        return self.dataset_length