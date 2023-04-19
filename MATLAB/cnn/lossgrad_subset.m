function [net, loss] = lossgrad_subset(prob, model, net, batch_idx, task)

L = model.L;
LC = model.LC;

num_data = length(batch_idx);

Y = gpu(@zeros, [model.nL, num_data]);
Y(prob.y_mapped(batch_idx) + model.nL*[0:num_data-1]') = 1;
%Y = prob.label_mat(:, batch_idx);

% fun
net = feedforward(prob.data(:, batch_idx), model, net); % debug: 4.9579
preds =  1./(1+exp(-net.Z{L+1}));
loss = -mean(dot(Y, log(preds)) + dot(1-Y, log(1-preds))); % toy: 0.8888
% loss = norm(net.Z{L+1} - Y, 'fro')^2; % ??

if strcmp(task, 'fungrad')
	% grad
    % TBD: how to calculate the backward pass of
    % binary_cross_entropy_with_logits?
    % pytorch: loss.backward
	v = 2*(net.Z{L+1} - Y);
	v = JTv(model, net, v);
	for m = 1 : L
		net.dlossdW{m} = v{m}(:, 1:end-1);
		net.dlossdb{m} = v{m}(:, end);	
	end
end
