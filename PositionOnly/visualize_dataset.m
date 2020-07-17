% This script gets an image and 2 sets of coords then scatter plots points
% on the image along with a circle showing the direction

clc
clear

num_data = 11;
% read image
im = imread(['Image/', num2str(num_data, '%06.f'),'.png']);
% read json files (target points and their label)
fname = ['Point_Location/', num2str(num_data, '%06.f'),'.json'];
val = jsondecode(fileread(fname));
target_points = [200-val.Y, val.X];

fname = ['Coarse_Label/', num2str(num_data, '%06.f'),'.json'];
val = jsondecode(fileread(fname));
target_labels = val;
% read json files (predicted points and their label)


%% visualization
figure
imshow(im)
hold on
plot(target_points(:,2), target_points(:,1), 'y*')

target_angles = target_labels .* linspace(22.5,360,16);

for i=1:size(target_points,1)
    [~,~,angles] = find(target_angles(i,:));
    alphas = zeros(length(angles),100);
    for j=1:length(angles)
        alpha = linspace(angles(j)-22.5, angles(j), 100); 
        origin = [target_points(i,1), target_points(i,2)];
    %     zer = zeros()
        patch([origin(2)  origin(2)+5*cosd(alpha) origin(2)],...
              [origin(1) origin(1)-5*sind(alpha) origin(1)],...
              'r','FaceAlpha',.5)
    end
end
