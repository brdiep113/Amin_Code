% This script generates 4D input features using the current images in Image
% folder and saves them in 4D folder
clc
clear

PATH_READ = 'Image';
PATH_WRITE = '4D';

paths = dir(fullfile(PATH_READ, '*.png'));
n_data = numel(paths);

for k = 1:n_data
    this_name = paths(k).name;
    this_path = fullfile(PATH_READ, this_name);
    gray = rgb2gray(imread(this_path));
    [Gx, Gy] = imgradientxy(gray);
    [Gmag, Gdir] = imgradient(Gx, Gy);
    edges = edge(gray,'Canny');
    
    % Normalize features between [0 1]
    gray = double(gray)/255;
    Gmag = Gmag/max(Gmag(:));
    Gdir = Gdir/max(Gdir(:));
    edges = double(edges);
    input_4D = cat(3, gray, Gmag, Gdir, edges);
    save(fullfile(PATH_WRITE, [this_name(1:end-4), '.mat']), 'input_4D')
end



