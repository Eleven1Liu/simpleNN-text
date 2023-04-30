function [net, loss] = lossgrad_subset(prob, model, net, batch_idx, task)

L = model.L;
LC = model.LC;

num_data = length(batch_idx);

Y = gpu(@zeros, [model.nL, num_data]);
Y(prob.y_mapped(batch_idx) + model.nL*[0:num_data-1]') = 1;
%Y = prob.label_mat(:, batch_idx);

% fun
net = feedforward(prob.data(:, batch_idx), model, net);

% original binary cross entropy
% K = size(Y, 1);
% preds =  1./(1+exp(-net.Z{L+1}));  % sigmoid
% loss = -(dot(Y, log(preds)) + dot(1-Y, log(1.-preds)));
% loss = sum(loss ./ K);

% pytorch: BCELoss with logits
% max_val = (-net.Z{L+1}); % clamp min(0)
% max_val(max_val < 0) = 0.;
% loss = (1-Y) .* net.Z{L+1} + max_val + log((exp(-max_val) + exp(-net.Z{L+1}-max_val)));

% mystery loss
loss = -Y .* net.Z{L+1} + log(1+exp(net.Z{L+1}));

fprintf('batch loss: %g\n', mean(mean(loss)));
loss = sum(mean(loss));

% square error
% loss = norm(net.Z{L+1} - Y, 'fro')^2;

if strcmp(task, 'fungrad')
	% grad
    % backward is wrong in the second batch
    v = -Y + (exp(net.Z{L+1}) ./ (1.+exp(net.Z{L+1})));

    % binary cross entropy
    % sig =  exp(net.Z{L+1})/(1+exp(net.Z{L+1}));
    % v = (sig-Y);
    % v = (-Y) .* (1./net.Z{L+1}) + (1-Y) .* 1./(1-net.Z{L+1});

	% v = 2*(net.Z{L+1} - Y); % batch size * label size, here v corresponds to dlossdZ{L+1}
	v = JTv(model, net, v);
	for m = 1 : L
		net.dlossdW{m} = v{m}(:, 1:end-1);
		net.dlossdb{m} = v{m}(:, end);
	end
end
