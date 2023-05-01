function [net, loss] = lossgrad_subset(prob, model, net, batch_idx, task)

L = model.L;
LC = model.LC;

num_data = length(batch_idx);

Y = gpu(@zeros, [model.nL, num_data]);
Y(prob.y_mapped(batch_idx) + model.nL*[0:num_data-1]') = 1;
%Y = prob.label_mat(:, batch_idx);

% fun
net = feedforward(prob.data(:, batch_idx), model, net);

% binary cross entropy loss
loss = -Y .* net.Z{L+1} + log(1+exp(net.Z{L+1}));
loss = sum(mean(loss));

% square error
% loss = norm(net.Z{L+1} - Y, 'fro')^2;

if strcmp(task, 'fungrad')
	% grad
    K = size(Y, 1);
    v = (1./K) * (-Y + (exp(net.Z{L+1}) ./ (1.+exp(net.Z{L+1}))));

	% v = 2*(net.Z{L+1} - Y); % batch size * label size, here v corresponds to dlossdZ{L+1}
	v = JTv(model, net, v);
	for m = 1 : L
		net.dlossdW{m} = v{m}(:, 1:end-1);
		net.dlossdb{m} = v{m}(:, end);
	end
end
