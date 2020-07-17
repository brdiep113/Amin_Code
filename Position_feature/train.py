'''
This script aims at training the model
'''


import torch
import torchvision
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from network import Model
from dataset import MyDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# loss function
def loss_descriptor(estimation, target):
    zeros = torch.zeros_like(estimation)
    error = torch.max(target - estimation, zeros)
    loss_val = torch.mean(torch.pow(error, 2))
    return loss_val

# Arguments
num_epochs = 1 

# Define transforms
# transformations = transforms.Compose([])

# Define custom dataset
my_dataset = MyDataset('.')

# Define data loader  #TODO: Can be replaced with random_split
batch_size = 8
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

train_indices, val_indices = indices[split:], indices[:split]
# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(my_dataset, 
                                          batch_size=batch_size, 
                                          sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(my_dataset,
                                                batch_size=batch_size, 
                                                sampler=valid_sampler)


# model, loss, and optimizer settings
model = Model()
model.to(device=device)
# print(model)

# TODO: find a proper loss e.g., BCEWithLogitLoss
loss_position = nn.BCELoss() 
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.001, 
                             betas=(0.9, 0.999), 
                             eps=1e-08, 
                             weight_decay=0.001, 
                             amsgrad=False)

# training
# print(model.state_dict())
model = model.float()

for epoch in range(num_epochs):

    # Train
    model.train()

    # Sum of losses from this epoch
    epoch_loss = 0

    for _, data in enumerate(train_loader):
        
        # Load data to tensors
        img = data['image']
        position_target = data['point_map']
        feature_target = data['feature_map']
        img = img.to(device=device, dtype=torch.float32)
        position_target = position_target.to(device=device)
        feature_target = feature_target.to(device=device)

        # Calculate Loss
        position_map, feature_map = model(img)
        loss_pos = loss_position(position_map, position_target)
        loss_desc = loss_descriptor(feature_map, feature_target)
        #FIXME: Can add some weights or other aggregation method
        loss = loss_pos + loss_desc   
        epoch_loss += loss.item()

        # Backpropagate and update optimizer learning rate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: add checkpoint model saving periodically
        # if iterations % args.save_every == 0:
        #     snapshot_prefix = os.path.join(args.save_path, 'snapshot')
        #     snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(), iterations)
        #     torch.save(model, snapshot_path)
        #     for f in glob.glob(snapshot_prefix + '*'):
        #         if f != snapshot_path:
        #             os.remove(f)


    # print statistics
    print(f"epoch:[%.d] Training loss: %.3f" %(epoch+1, loss))
    
    # Evaluate perfomance on validation periodically
    # validation every 1 epochs
    if (epoch+1) % 1 == 0:
        validation_loss = 0.0
        num_batch = 0
        for _, data in enumerate(validation_loader):
            model.eval()

            img = data['image']
            position_target = data['point_map']
            feature_target = data['feature_map']
            img = img.to(device=device)
            position_target = position_target.to(device=device, 
                                                 dtype=torch.float32)
            feature_target = feature_target.to(device=device, 
                                                 dtype=torch.float32)
            position_map, feature_map = model(img)


            # loss calculation
            loss_pos = loss_position(position_map, position_target)
            loss_desc = loss_descriptor(feature_map, feature_target)
            loss = loss_pos + loss_desc
            validation_loss += loss.item()
            num_batch += 1 
        
        # print statistics
        print(f"epoch:[%.d] Validation loss: %.3f" %(epoch+1,
                                                     validation_loss/num_batch))


torch.save(model.state_dict(), 'model_saved.pth')

# print(model.state_dict())

# Visualize the model and save the graph
# g = make_dot(affine_params, params=dict(model.named_parameters()))
# g.view('model', './summary')