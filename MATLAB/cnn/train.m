function model = train(prob, prob_v, param, net_config)

model = init_model(net_config);
net = init_net(model, param);

[model.labels, ~, prob.y_mapped] = unique(prob.y);

switch param.solver
	case 1
		model = newton(prob, prob_v, param, model, net);
	case 2
		model = sgd(prob, prob_v, param, model, net);
	case 3
		model = adam(prob, prob_v, param, model, net);
otherwise
	error('solver not correctly specified', param.solver);
end

function model = init_model(net_config)

model = struct;
model.net_config = net_config;
global gpu_use;
model.gpu_use = gpu_use;
global float_type;
model.float_type = float_type;

LC = net_config.LC;
L = net_config.L;

model.LC = LC;
model.L = L;
model.nL = net_config.nL;

model.ht_input = [net_config.ht_input; zeros(LC, 1)];  % height of input image
model.wd_input = [net_config.wd_input; zeros(LC, 1)];  % width of input image
model.ch_input = net_config.ch_input(:);  % #channels of input image
model.wd_pad_added = net_config.wd_pad_added(:);  % width of zero-padding around input image border
model.ht_pad = zeros(LC, 1);  % height of image after padding
model.wd_pad = zeros(LC, 1);  % width of image after padding
model.ht_conv = zeros(LC, 1);  % height of image after convolution
model.wd_conv = zeros(LC, 1);  % width of image after convolution
model.wd_filter = net_config.wd_filter(:);  % width of filter in convolution
model.strides = net_config.strides(:);  % strides of convolution
model.wd_subimage_pool = net_config.wd_subimage_pool(:);  % width of filter in pooling
model.full_neurons = net_config.full_neurons(:);  % #neurons in fully-connected layers
model.weight = cell(L, 1);
model.bias = cell(L, 1);
var_num = zeros(L, 1);

% load weights from LibMultiLabel
if isfield(net_config, 'init_weight_mat')
    load(net_config.init_weight_mat, 'conv_weight', 'conv_bias', 'linear_weight', 'linear_bias');
end

% convolutional layer
for m = 1 : LC
	model.ht_pad(m) = model.ht_input(m) + 2*model.wd_pad_added(m); % no padding here
	model.wd_pad(m) = model.wd_input(m) + 2*model.wd_pad_added(m); % no padding here

    model.ht_conv(m) = 1;  % a_out = 1
    model.wd_conv(m) = floor((model.wd_pad(m) - model.wd_filter(m))/model.strides(m)) + 1;

    model.ht_input(m+1) = 1;
    model.wd_input(m+1) = floor(model.wd_conv(m)/model.wd_subimage_pool(m));

    var_num(m) = model.ch_input(m+1)*(1*model.wd_filter(m)*model.ch_input(m) + 1);
    
    if exist('conv_weight','var') && exist('conv_bias','var')
        model.weight{m} = conv_weight;
        model.bias{m} = conv_bias;
    else
        % random initialize weights
        model.weight{m} = gpu(ftype(randn(model.ch_input(m+1), 1*model.wd_filter(m)*model.ch_input(m))*sqrt(2.0/(1*model.wd_filter(m)*model.ch_input(m)))));
        model.bias{m} = gpu(@zeros, [model.ch_input(m+1), 1]);
    end
end

% linear layer
num_neurons_prev = model.ht_input(LC+1)*model.wd_input(LC+1)*model.ch_input(LC+1);
for m = LC+1 : L
	num_neurons = model.full_neurons(m - LC);

    % load weights
    if exist('linear_weight','var') && exist('linear_bias','var')
        model.weight{m} = linear_weight;
        model.bias{m} = linear_bias;
    else
        model.weight{m} = gpu(ftype(randn(num_neurons, num_neurons_prev)*sqrt(2.0/(num_neurons_prev))));
        model.bias{m} = gpu(@zeros, [num_neurons, 1]);
    end
    
	var_num(m) = num_neurons * (num_neurons_prev + 1);
	num_neurons_prev = num_neurons;
end
% starting index of trained variables (including biases) for each layer
model.var_ptr = [1; cumsum(var_num)+1];
