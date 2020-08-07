'''Checks the prediction results on validation data
   saves in valid_resuls folder '''

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
my_dataset = MyDataset('.', choose_features=choose_features)

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
val_indices = indices[:split]

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

for ind in val_indices:
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
    savemat(f"valid_results/%.06d.mat"%(ind), mdic)

