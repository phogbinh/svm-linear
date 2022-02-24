train_data = readmatrix(Def.DATA_FILENAME);
train_data_N = numel( train_data(:, 1) );

x = train_data(:, 1:2);
r = train_data(:, 3);

N = numel( x(1, :) );

cvx_begin
    variables w(N) b
    minimize( w' * w )
    subject to
        r .* (x * w + b) >= 1
cvx_end