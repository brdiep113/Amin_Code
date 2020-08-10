function acc = point_accuracy(pred, target, radius)
% how many percent of the target points are correctly predicted
idx = rangesearch(pred, target, radius);
n_points = size(target, 1);
n_corrects = n_points - sum(cellfun(@isempty, idx));

acc = n_corrects/n_points * 100;

end

