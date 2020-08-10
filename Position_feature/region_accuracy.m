function acc = region_accuracy(pred,target, idx)
%The percentage of activation regions correctly predicted
%   idx : nearest pred points to each point in the target
ind = cellfun(@isempty, idx);
n_pred = size(pred,1);
pred = [pred; 2*ones(1,16)];
idx(ind) = {n_pred + 1};
idx = cellfun(@(x) x(1), idx);
pred = pred(idx, :);
compare = pred == target;
n_corrects = sum(compare(:));
n_samples = numel(compare);
acc = n_corrects / n_samples * 100;

end

