% This script gets an image and 2 sets of coords then scatter plots points
% on the image along with a circle showing the direction

clc
clear

% set the name of the results
num = 18;

% read image
im = imread(['datasets/TrainingValidation/Image/', num2str(num,'%06.f') ,'.png']);
% read results in mat files (predicted points)
load(['results/train/', num2str(num,'%06.f') ,'.mat']);

% read json files (target points and their label)
fname = ['datasets/TrainingValidation/Point_Location/', num2str(num, '%06.f'),'.json'];
val = jsondecode(fileread(fname));
target_points = [size(im,1)-val.Y, val.X];

%  postition
p = position_map > 0.5; % for now, cause the results are not limited to 0|1
[r,c, ~] = find(p);
points = [c, r];

%% visualization
figure
imshow(im)
hold on
plot(points(:,2), points(:,1), 'y*')

%% Accuracy
% position
p = single(p);
id = sub2ind([128, 128],round(target_points(:,1)), round(target_points(:,2)));
target_p = zeros(128);
target_p(id)=1;

[acc_total_position] = accuracy(p,target_p);
table(acc_total_position)