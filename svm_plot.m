train_data = readmatrix('data/basic_train');
train_data_N = numel( train_data(:, 1) );
train_pos = zeros(train_data_N, 2);
train_pos_N = 0;
train_neg = zeros(train_data_N, 2);
train_neg_N = 0;

xmin = inf;
xmax = -inf;
ymin = inf;
ymax = -inf;
for i = 1:train_data_N
    xmin = min(xmin, train_data(i, 1));
    xmax = max(xmax, train_data(i, 1));
    ymin = min(ymin, train_data(i, 2));
    ymax = max(ymax, train_data(i, 2));
    if train_data(i, 3) == 1
        train_pos_N = train_pos_N + 1;
        train_pos(train_pos_N, :) = train_data(i, 1:2);
    else
        train_neg_N = train_neg_N + 1;
        train_neg(train_neg_N, :) = train_data(i, 1:2);
    end
end
train_pos = train_pos(1:train_pos_N, :);
train_neg = train_neg(1:train_neg_N, :);

xmin = xmin - 1;
xmax = xmax + 1;
ymin = ymin - 1;
ymax = ymax + 1;

w = matfile('train.mat').train_w;
b = matfile('train.mat').train_b;
[x_neg, y_neg] = plot_linear(w(1), w(2), b+1, xmin, xmax, ymin, ymax);
[x_pos, y_pos] = plot_linear(w(1), w(2), b-1, xmin, xmax, ymin, ymax);

figure
hold on
plot(x_neg, y_neg, 'b');
plot(x_pos, y_pos, 'r');
plot(train_pos(:, 1), train_pos(:, 2), '+r', ...
     'MarkerSize', 10);
plot(train_neg(:, 1), train_neg(:, 2), 'ob', ...
     'MarkerSize', 10);
hold off