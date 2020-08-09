'''Checks results on test data
   saves in results/test folder '''

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from network import Model
from dataset import MyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom dataset
# the order of input features [gray, gmag, gdir, edges, shi_tomasi response]
choose_features = [0, 1, 2, 3, 4]
n_features = len(choose_features)
my_dataset = MyDataset('datasets/Test', 
                       choose_features=choose_features)

# Setting configurations
batch_size = 1

# Define data loader
# Creating data indices for training and validation splits:
dataset_size = len(my_dataset)
indices = list(range(dataset_size))

# check if dataset load order is correct
# for ind in val_indices:
#     print(ind)
#     img, _ = my_dataset[ind]
#     plt.figure()
#     plt.imshow(img.permute(1,2,0))
#     plt.show()

# load model
model = Model(num_features=n_features).to(device=device)
model.load_state_dict(torch.load('stats/model_saved.pth'))
model = model.float()
model.eval()

for ind in indices:
    data = my_dataset[ind]
    img = data['image']
    img = img.to(device=device)
    img = img.unsqueeze(dim=0)
    pred = model(img)
    logits = pred['logits']
    position_map = pred['prob']
    
    logits = logits.squeeze().permute(1, 2, 0)      # must be (16,16,65)
    logits = logits.detach().cpu().numpy()
    position_map = position_map.squeeze()           # must be (128,128)
    position_map = position_map.detach().cpu().numpy()
        
    mdic = {'logits' : logits, 'position_map': position_map}
    savemat(f"results/test/%.06d.mat"%(ind), mdic)

