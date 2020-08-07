'''
This script aims at training the model
'''

import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from network import Model
from dataset import MyDataset
from torch.utils.data import DataLoader
from utils import loss_position, loss_feature
from scipy.io import savemat


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# Arguments
num_epochs = 5
alpha = 0.1
# Define transforms
# transformations = transforms.Compose([])

# Define custom dataset
# the order of input features [gray, gmag, gdir, edges, shi_tomasi response]
choose_features = [0, 1, 2, 3, 4]
n_features = len(choose_features)
my_dataset = MyDataset('.', choose_features=choose_features)

# Define data loader
batch_size = 32
validation_split = .1
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(my_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]
# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(my_dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(my_dataset, batch_size=batch_size, sampler=valid_sampler)


# model, loss, and optimizer settings
model = Model(num_features=n_features)
model.to(device=device)
# print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), 
                             eps=1e-08, weight_decay=0.001, amsgrad=False)

# training
# print(model.state_dict())
model = model.float()
loss_pos_train = []
loss_feat_train = []
loss_train = []
loss_pos_val = []
loss_feat_val = []
loss_val = []

for epoch in range(num_epochs):

    # Train
    model.train()

    # Sum of losses from this epoch
    epoch_loss_pos_train = 0
    epoch_loss_feat_train = 0
    epoch_loss_train = 0

    for i, data in enumerate(train_loader):

        # Zeros the gradients of all optimized torch.Tensors
        optimizer.zero_grad()

        # Load data to tensors
        img = data['image']
        position_target = data['point_map']
        feature_target = data['feature_map']
        img = img.to(device=device)
        position_target = position_target.to(device=device)
        feature_target = feature_target.to(device=device)

        # Calculate loss
        pred_position, pred_feature = model(img)
        logits = pred_position['logits']
        features = pred_feature['logits']
        loss_pos = loss_position(logits, position_target)
        loss_feat = loss_feature(features, feature_target)
        loss = loss_pos + alpha * loss_feat        # TODO: coef. must be tuned 
        
        epoch_loss_pos_train += loss_pos.item() * img.size(0)
        epoch_loss_feat_train += loss_feat.item() * img.size(0) 
        epoch_loss_train += loss.item() * img.size(0)

        # Zero gradients, perform a backward pass, and update the weights
        loss.backward()
        optimizer.step()
        
        # TODO: add checkpoint model saving periodically
    loss_pos_train.append(epoch_loss_pos_train/len(train_indices))
    loss_feat_train.append(epoch_loss_feat_train/len(train_indices))
    loss_train.append(epoch_loss_train/len(train_indices))
    # print statistics
    print(f"epoch:[%.d] Train loss: %.5f, loss_pos: %.5f, loss_feat: %.5f"
           %(epoch+1, loss_train[-1], loss_pos_train[-1], loss_feat_train[-1]))

    # Evaluate perfomance on validation periodically
    # validation every 1 epochs
    if (epoch+1) % 1 == 0:

        # Validation
        model.eval()
        
        epoch_loss_pos_val = 0.0
        epoch_loss_feat_val = 0.0
        epoch_loss_val = 0.0

        with torch.no_grad():
                
            for _, data in enumerate(validation_loader):
                img = data['image']
                position_target = data['point_map']
                feature_target = data['feature_map']
                img = img.to(device=device)
                position_target = position_target.to(device=device)
                feature_target = feature_target.to(device=device)

                pred_position, pred_feature = model(img)
                logits = pred_position['logits']
                features = pred_feature['logits']
                # loss calculation
                loss_val_pos = loss_position(logits, position_target)
                loss_val_feat = loss_feature(features, feature_target)
                val_loss = loss_val_pos + alpha * loss_val_feat  #TODO: coef. must be tuned
                epoch_loss_pos_val += loss_val_pos.item() * img.size(0)
                epoch_loss_feat_val += loss_val_feat.item() * img.size(0)
                epoch_loss_val += val_loss.item() * img.size(0)

        loss_pos_val.append(epoch_loss_pos_val/len(val_indices))
        loss_feat_val.append(epoch_loss_feat_val/len(val_indices))
        loss_val.append(epoch_loss_val/len(val_indices))
        # print statistics
        print(f"epoch:[%.d] Valid loss: %.5f, loss_pos: %.5f, loss_feat: %.5f" 
              %(epoch+1, loss_val[-1], loss_pos_val[-1], loss_feat_val[-1]))

# Save the model
torch.save(model.state_dict(), 'stats/model_saved.pth')

# Save some log statistics
loss_train = np.array(loss_train)
loss_val = np.array(loss_val)
loss_pos_train = np.array(loss_pos_train)
loss_feat_train = np.array(loss_feat_train)
loss_pos_val = np.array(loss_pos_val)
loss_feat_val = np.array(loss_feat_val)

mdic = {'loss_train' : loss_train, 'loss_val' : loss_val,
        'loss_pos_train' : loss_pos_train, 'loss_pos_val' : loss_pos_val,
        'loss_feat_train' : loss_feat_train, 'loss_feat_val' : loss_feat_val,
        'batch_size' : batch_size, 'validation_split' : validation_split,
        'dataset_size' : dataset_size, 'random_seed' : random_seed}
savemat("stats/log.mat", mdic)

# Plot loss Evolution
plt.plot(loss_train, label='training loss')
plt.plot(loss_val, label='validation loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('stats/loss_evolution.png', bbox_inches='tight')
plt.show()