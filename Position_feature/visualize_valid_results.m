% This script gets an image and 2 sets of coords then scatter plots points
% on the image along with a circle showing the direction

clc
clear

% set the name of the results
num_result = 21;

% read image
im = imread(['Image/', num2str(num_result,'%06.f') ,'.png']);
% read results in mat files (predicted points)
load(['valid_results/', num2str(num_result,'%06.f') ,'.mat']);

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

pred_labels = single(pred_labels > 0.99); % for now
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
