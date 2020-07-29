% Shows feature maps in latent space of the model
clc
clear

load('latent.mat') %NxHxWxC  N: #step 
% results are for the first input of each batch 

% Set the feature map you want to observe out of 128
n_feature = 25;

% Set the show | mode=1: Single Epoch & mode=2: All epochs
mode = 2;
%% show selected feature map for a single epoch 
if mode == 1
    % Set epoch in question
    n_epoch = 1;   
    fm = latent(n_epoch,:,:,n_feature);
    fm = squeeze(fm);
    % if you wanna have it with the original size, then comment the below line
    fm = imresize(fm,8, 'Method','bicubic');
    imshow(fm)
end
%% You can also observe the change trend for feature n_feautre(e.g., 3rd feature) 
if mode == 2
    for i=1:size(latent,1)

        fm = latent(i,:,:,n_feature);
        fm = squeeze(fm);
        fm = imresize(fm,8, 'Method','bicubic');
        imshow(fm)
        pause(1)  % waits for 1 sec
    end
end