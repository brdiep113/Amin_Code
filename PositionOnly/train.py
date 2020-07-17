'''
This script aims at training the model
'''


import torch
import torchvision
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from network import Model
from dataset import MyDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# Arguments
num_epochs = 5

# Define transforms
# transformations = transforms.Compose([])

# Define custom dataset
my_dataset = MyDataset('.')

# Define data loader  #TODO: Can be replaced with random_split
batch_size = 8
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

train_indices, val_indices = indices[split:], indices[:split]
# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(my_dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(my_dataset, batch_size=batch_size, sampler=valid_sampler)


# model, loss, and optimizer settings
model = Model()
model.to(device=device)
# print(model)

# TODO: find a proper loss e.g., BCEWithLogitLoss
loss_position = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

# training
# print(model.state_dict())
model = model.float()
loss_train = []
loss_val = []

for epoch in range(num_epochs):

    # Train
    model.train()

    # Sum of losses from this epoch
    epoch_loss_train = 0

    for _, data in enumerate(train_loader):

        # Load data to tensors
        img = data['image']
        position_target = data['point_map']
        img = img.to(device=device, dtype=torch.float32)
        position_target = position_target.to(device=device)

        # Calculate loss
        position_map = model(img)
        loss_pos = loss_position(position_map, position_target)
        epoch_loss_train += loss_pos.item() * img.size(0)

        # Backpropagate and update optimizer learning rate
        optimizer.zero_grad()
        loss_pos.backward()
        optimizer.step()

        # TODO: add checkpoint model saving periodically

    loss_train.append(epoch_loss_train/len(train_indices))
    # print statistics
    print(f"epoch:[%.d] Training loss: %.5f" %(epoch+1, loss_train[-1]))

    # Evaluate perfomance on validation periodically
    # validation every 1 epochs
    if (epoch+1) % 1 == 0:
        
        epoch_loss_val = 0.0

        for _, data in enumerate(validation_loader):
            img = data['image']
            position_target = data['point_map']
            img = img.to(device=device)
            position_target = position_target.to(device=device, dtype=torch.float32)
            position_map = model(img)

            # loss calculation
            loss_pos_val = loss_position(position_map, position_target)
            epoch_loss_val += loss_pos_val.item() * img.size(0)

        loss_val.append(epoch_loss_val/len(val_indices))
        # print statistics
        print(f"epoch:[%.d] Validation loss: %.5f" %(epoch+1, loss_val[-1]))


torch.save(model.state_dict(), 'model_saved.pth')

# Plot loss Evolution
plt.plot(loss_train, label='training loss')
plt.plot(loss_val, label='validation loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize the model and save the graph
# g = make_dot(affine_params, params=dict(model.named_parameters()))
# g.view('model', './summary')