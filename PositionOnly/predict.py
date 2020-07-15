import torch
import numpy as np
from scipy.io import savemat
from network import Model
from dataset import MyDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define custom dataset
my_dataset = MyDataset('.')

# Define data loader
batch_size = 1
validation_split = .3
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

# load model
model = Model().to(device=device)
model.load_state_dict(torch.load('model_saved.pth'))
model = model.float()
model.eval()

for ind in val_indices:
    img, _ = my_dataset[ind]
    position_map = model(img)
    
    position_map = position_map.squeeze()           # must be (128,128)
    position_map = position_map.detach().numpy()
        
    mdic = {'position_map': position_map}
    savemat(f"results/%.d.mat"%(ind), mdic)

