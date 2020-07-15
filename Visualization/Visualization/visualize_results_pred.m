% This script gets an image and 2 sets of coords then scatter plots points
% on the image along with a circle showing the direction

clc
clear

% read image
num_result = 0;


im = imread(['Image/', num2str(num_result) ,'.png']);
% read json files (target points and their label)
load(['results/', num2str(num_result) ,'.mat']);

%  postition
p = position_map > 0.5; % for now, cause the results are not limited to 0|1
[r,c, ~] = find(p);
points = [c, r];

% feature
rr = repmat(r,16,1);
cc = repmat(c,16,1);
dd = repmat(1:16, length(r),1); dd = dd(:);
ind = sub2ind(size(feature_maps),rr,cc,dd); 
fms = feature_maps(ind);
features = reshape(fms, [], 16);
f = features > 0.5;
% read json files (predicted points and their label)


%% visualization
figure
imshow(im)
hold on
plot(points(:,2), points(:,1), 'y*')

pred_angles = f .* linspace(22.5,360,16);

for i=1:size(points,1)
    [~,~,angles] = find(pred_angles(i,:));
    alphas = zeros(length(angles),100);
    for j=1:length(angles)
        alpha = linspace(angles(j)-22.5, angles(j), 100); 
        origin = [points(i,1), points(i,2)];
        patch([origin(2)  origin(2)+5*cosd(alpha) origin(2)],...
              [origin(1) origin(1)-5*sind(alpha) origin(1)],...
              'r','FaceAlpha',.5)
    end
end
