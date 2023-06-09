function idx = find_index_phiZ(a, b, d, h, s)

first_channel_idx = gpu([0:0]'*d+1) + gpu([0:h-1]*a*d);
first_col_idx = first_channel_idx(:) + [0:d-1];
a_out = 1;
b_out = floor((b - h)/s) + 1;
column_offset = (gpu([0:a_out-1])' + gpu([0:b_out-1]*a))*s*d;
idx = column_offset(:)' + first_col_idx(:);
idx = idx(:);
