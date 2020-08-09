% This script gets an image and 2 sets of coords then scatter plots points
% on the image along with a circle showing the direction

clc
clear

% set the name of the results
num = 11;

% read image
im = imread(['datasets/TrainingValidation/Image/', num2str(num,'%06.f') ,'.png']);
% read results in mat files (predicted points)
load(['results/train', num2str(num,'%06.f') ,'.mat']);

% read json files (target points and their label)
fname = ['datasets/TrainingValidation/Point_Location/', num2str(num, '%06.f'),'.json'];
val = jsondecode(fileread(fname));
target_points = [size(im,1)-val.Y, val.X];

fname = ['datasets/TrainingValidation/Coarse_Label/', num2str(num, '%06.f'),'.json'];
val = jsondecode(fileread(fname));
target_labels = val;

%  postition
p = position_map > 0.5; % for now, cause the results are not limited to 0|1
[r,c, ~] = find(p);
points = [c, r];

% feature
%feature_map = permute(feature_map, [2,3,1]);
pred_labels = zeros(length(r), 16);
for k = 1:length(r)
    pred_labels(k,:) = feature_map(r(k), c(k), :);
end

pred_labels = single(pred_labels > 0.5); % for now
%% visualization
figure
imshow(im)
hold on
plot(points(:,1), points(:,2), 'y*')


pred_angles = pred_labels .* linspace(22.5,360,16);

for i=1:size(points,1)
    [~,~,angles] = find(pred_angles(i,:));
    alphas = zeros(length(angles),100);
    for j=1:length(angles)
        alpha = linspace(angles(j)-22.5, angles(j), 100); 
        origin = [points(i,2), points(i,1)];
    %     zer = zeros()
        patch([origin(2)  origin(2)+3*cosd(alpha) origin(2)],...
              [origin(1) origin(1)-3*sind(alpha) origin(1)],...
              'r','FaceAlpha',.5)
    end
end

%% Accuracy
% position
p = single(p);
id = sub2ind([128, 128],round(target_points(:,1)), round(target_points(:,2)));
target_p = zeros(128);
target_p(id)=1;

[acc_total_position] = accuracy(p,target_p);

% feature
f = zeros(size(feature_map));
for k = 1:length(r)
    f(r(k), c(k), :) = feature_map(r(k), c(k), :);
end
f = single(f > 0.5);
target_f = zeros(128,128,16);
r_t = round(target_points(:,1));
c_t = round(target_points(:,2));
for k = 1:length(r_t)
    target_f(r_t(k), c_t(k), :) = target_labels(k, :);
end

[acc_total_feature] = accuracy(f,target_f);
table(acc_total_position, acc_total_feature)