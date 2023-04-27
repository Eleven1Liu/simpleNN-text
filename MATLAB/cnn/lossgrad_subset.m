function [net, loss] = lossgrad_subset(prob, model, net, batch_idx, task)

L = model.L;
LC = model.LC;

num_data = length(batch_idx);

Y = gpu(@zeros, [model.nL, num_data]);
Y(prob.y_mapped(batch_idx) + model.nL*[0:num_data-1]') = 1;
%Y = prob.label_mat(:, batch_idx);

% fun
K = size(Y, 1);
net = feedforward(prob.data(:, batch_idx), model, net); % debug: 4.9579
preds =  1./(1+exp(-net.Z{L+1}));  % sigmoid

% pytorch
% Our solution is that BCELoss clamps its log function outputs to be
% greater than or equal to -100. This way, we can always have a finite loss value
% and a linear backward method.
% loss = dot(Y, max(log(preds), -100)) + dot(1-Y, max(log(1.-preds), -100));

% this might be different
loss = -(dot(Y, log(preds)) + dot(1-Y, log(1.-preds)));
% size(loss)
loss = sum(loss ./ K);
% loss = mean(loss);
% loss = norm(net.Z{L+1} - Y, 'fro')^2; % ??
% dloss = 2*(net.Z{L+1} - Y)

% loss = - (Y * log(sigmoid(x)) + (1-Y) * log(sigmoid(1-x)))

% sigmoid(x) = 1./(1+exp(-x)) = (1+exp(-x))^-1
% sigmoid(x)' = exp(-x) / (1+exp(-x))^2
%             = (1+exp(-x)-1) / (1+exp(-x))^2
%             = (1+exp(-x)) / (1+exp(-x))^2 - 1 / (1+exp(-x))^2
%             = 1 / (1+exp(-x)) - 1 / (1+exp(-x))^2
%             = 1 / (1+exp(-x)) (1 - 1 / (1+exp(-x)))
%             = sigmoid(x) * (1-sigmoid(x))

% dloss = - ((Y / sigmoid(x)) * (sigmoid(x))' +
%        	((1-Y) / sigmoid(1-x)) * (sigmoid(1-x))' * (1-x)')
%       = - ((Y / sigmoid(x)) * (sigmoid(x) * (1-sigmoid(x))) +
%        ((1-Y) / sigmoid(1-x)) * (sigmoid(1-x) * (1-sigmoid(1-x))) * (-1))
%       = -Y * (1-sigmoid(x)) + (1-Y) * (1-sigmoid(1-x))


if strcmp(task, 'fungrad')
	% grad
    % TBD: how to calculate the backward pass of BCE?
    % https://courses.grainger.illinois.edu/ECE417/fa2021/lectures/lec18.pdf

    % binary cross entropy
    sig =  exp(net.Z{L+1})/(1+exp(net.Z{L+1}));
    v = (sig-Y); % .* net.Z{L+1};
	% size(v)
    % v = (-Y) .* (1./net.Z{L+1}) + (1-Y) .* 1./(1-net.Z{L+1});
	% v = 2*(net.Z{L+1} - Y); % batch size * label size, here v corresponds to dlossdZ{L+1}
	v = JTv(model, net, v);
	for m = 1 : L
		net.dlossdW{m} = v{m}(:, 1:end-1);
		net.dlossdb{m} = v{m}(:, end);
	end
end
