'''Checks the train results
   saves in train_resuls folder '''

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from network import Model
from dataset import MyDataset
from utils import loss_position

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom dataset
my_dataset = MyDataset('.')

# Define data loader
batch_size = 1
validation_split = .1
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(my_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices = indices[split:]

# load model
model = Model().to(device=device)
model.load_state_dict(torch.load('model_saved.pth', 
                                 map_location=torch.device(device)))
model = model.float()

for ind in train_indices:
    data = my_dataset[ind]
    img = data['image']
    position_target = data['point_map']

    img = img.to(device=device)
    position_target = position_target.to(device=device)

    img = img.unsqueeze(dim=0)
    position_target = position_target.unsqueeze(dim=0)

    pred = model(img)
    logits = pred['logits']
    position_map = pred['prob']

    loss_pos = loss_position(logits, position_target)

    print(loss_pos.item())

    target = data['point_map'] 
    target = target.squeeze()
    target = target.detach().cpu().numpy()
    logits = logits.squeeze().permute(1, 2, 0)      # must be (16,16,65)
    logits = logits.detach().cpu().numpy()
    position_map = position_map.squeeze()           # must be (128,128)
    position_map = position_map.detach().cpu().numpy()
    
    mdic = {'target' : target,'logits' : logits, 'position_map': position_map}
    savemat(f"train_results/%.06d.mat"%(ind), mdic)

