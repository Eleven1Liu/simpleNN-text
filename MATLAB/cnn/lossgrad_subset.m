function [net, loss] = lossgrad_subset(prob, model, net, batch_idx, task)

L = model.L;
LC = model.LC;

num_data = length(batch_idx);

Y = gpu(@zeros, [model.nL, num_data]);
Y(prob.y_mapped(batch_idx) + model.nL*[0:num_data-1]') = 1;
%Y = prob.label_mat(:, batch_idx);

% fun
K = size(Y, 1);
net = feedforward(prob.data(:, batch_idx), model, net);
preds =  1./(1+exp(-net.Z{L+1}));  % sigmoid
% this might be different
loss = -(dot(Y, log(preds)) + dot(1-Y, log(1.-preds)));
loss = sum(loss ./ K);

% BCELoss with logits
% max_val = (-net.Z{L+1}); % clamp min(0)
% max_val(max_val < 0) = 0.;
% loss = (1-Y) .* net.Z{L+1} + max_val + log((exp(-max_val) + exp(-net.Z{L+1}-max_val)));

% square error
% loss = norm(net.Z{L+1} - Y, 'fro')^2;

if strcmp(task, 'fungrad')
	% grad
    % binary cross entropy
    sig =  exp(net.Z{L+1})/(1+exp(net.Z{L+1}));
    v = (sig-Y);
	% size(v)
    % v = (-Y) .* (1./net.Z{L+1}) + (1-Y) .* 1./(1-net.Z{L+1});
	% v = 2*(net.Z{L+1} - Y); % batch size * label size, here v corresponds to dlossdZ{L+1}
	v = JTv(model, net, v);
	for m = 1 : L
		net.dlossdW{m} = v{m}(:, 1:end-1);
		net.dlossdb{m} = v{m}(:, end);
	end
end
