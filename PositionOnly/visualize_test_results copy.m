% This script gets an image and 2 sets of coords then scatter plots points
% on the image along with a circle showing the direction

clc
clear

% set the name of the results
num_result = 18;

% read image
im = imread(['datasets/Test/Image/', num2str(num_result,'%06.f') ,'.png']);
% read results in mat files (predicted points)
load(['results/test/', num2str(num_result,'%06.f') ,'.mat']);

%  postition
p = position_map > 0.5; % for now, cause the results are not limited to 0|1
[r,c, ~] = find(p);
points = [c, r];

%% visualization
figure
imshow(im)
hold on
plot(points(:,2), points(:,1), 'y*')