'''Checks the train results
   saves in train_resuls folder '''

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from network import Model
from dataset import MyDataset
from utils import loss_position, loss_feature

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom dataset
# the order of input features [gray, gmag, gdir, edges, shi_tomasi response]
choose_features = [0, 1, 2, 3, 4]
n_features = len(choose_features)
my_dataset = MyDataset('datasets/TrainingValidation', 
                       choose_features=choose_features)

# Setting some parameters
alpha = 0.1
batch_size = 1
validation_split = .1
shuffle_dataset = True
random_seed= 42

# Define data loader
# Creating data indices for training and validation splits:
dataset_size = len(my_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices = indices[split:]

# load model
model = Model(num_features=n_features).to(device=device)
model.load_state_dict(torch.load('stats/model_saved.pth', 
                                 map_location=torch.device(device)))
model = model.float()

for ind in train_indices:
    data = my_dataset[ind]
    img = data['image']
    position_target = data['point_map']
    feature_target = data['feature_map']
    img = img.to(device=device)
    position_target = position_target.to(device=device)
    feature_target = feature_target.to(device=device)

    img = img.unsqueeze(dim=0)
    position_target = position_target.unsqueeze(dim=0)
    feature_target = feature_target.unsqueeze(dim=0)

    position_pred, feature_pred = model(img)
    logits_pos = position_pred['logits']
    position_map = position_pred['prob']
    logits_feat = feature_pred['logits']
    feature_map = feature_pred['features']

    loss_pos = loss_position(logits_pos, position_target)
    loss_feat = loss_feature(logits_feat, feature_target)
    loss = loss_pos + alpha * loss_feat  #TODO: coef. must be tuned
    print(loss.item())

    target_pos = data['point_map'] 
    target_pos = target_pos.squeeze()
    target_pos = target_pos.detach().cpu().numpy()
    target_feat = data['feature_map'] 
    target_feat = target_feat.squeeze()
    target_feat = target_feat.detach().cpu().numpy()
    logits_pos = logits_pos.squeeze().permute(1, 2, 0)      # must be (16,16,65)
    logits_pos = logits_pos.detach().cpu().numpy()
    logits_feat = logits_feat.squeeze().permute(1, 2, 0)  # must be (128,128,16)
    logits_feat = logits_feat.detach().cpu().numpy()
    position_map = position_map.squeeze()                    # must be (128,128)
    position_map = position_map.detach().cpu().numpy()
    feature_map = feature_map.squeeze().permute(1, 2, 0)  # must be (128,128,16)          
    feature_map = feature_map.detach().cpu().numpy()
    
    mdic = {'target_pos' : target_pos, 'target_feat' : target_feat, 
            'position_map': position_map, 'feature_map' : feature_map,
            'logits_pos' : logits_pos, 'logits_feat' : logits_feat}
    savemat(f"results/train/%.06d.mat"%(ind), mdic)

