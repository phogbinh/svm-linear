train_data = readmatrix('data/rings_of_saturn_train');
train_data_N = numel( train_data(:, 1) );

x = train_data(:, 1:2);
r = train_data(:, 3);

P = zeros(train_data_N, train_data_N);
for i = 1:train_data_N
    for j = 1:train_data_N
        P(i, j) = r(j) * r(i) * kernel(x(j, :), x(i, :));
    end
end

cvx_begin
    variable lambda(train_data_N)
    minimize(0.5*lambda'*P*lambda - ones(1, train_data_N) * lambda)
    subject to
        lambda >= 0
        lambda' * r == 0
cvx_end

b = -inf;
for i = 1:train_data_N
    if r(i) == 1
        wx = 0;
        for j = 1:train_data_N
            wx = wx + lambda(j) * r(j) * kernel(x(j, :), x(i, :));
        end
        if 1 - wx > b % smallest wx
            b = 1 - wx;
        end
    end
end