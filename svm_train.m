train_data = readmatrix(Def.DATA_FILENAME);
train_data_N = numel( train_data(:, 1) );

x = train_data(:, 1:2);
r = train_data(:, 3);

train_correct = 0;
f_min = inf; % optimization for widest street
train_xg = nan;
train_yg = nan;
train_lambda = nan;
train_b = nan;
for combination = 1:( bitshift(1, train_data_N) - 1 )
    gutters = zeros(train_data_N, 3);
    gutters_N = 0;
    for i = 1:train_data_N
        if bitget(combination, i)
            gutters_N = gutters_N + 1;
            gutters(gutters_N, :) = train_data(i, :);
        end
    end
    gutters = gutters(1:gutters_N, :);
    
    xg = gutters(:, 1:2);
    yg = gutters(:, 3);
    
    M = zeros(gutters_N + 1, gutters_N + 1);
    M(1, :) = [transpose(yg) 0];
    for i = 2:(gutters_N + 1)
        for j = 1:gutters_N
            M(i, j) = kernel( xg(i-1, :), xg(j, :) ) * yg(j);
        end
        M(i, gutters_N + 1) = 1;
    end

    a = [0; yg];
    
    if rank([M a]) == rank(M) + 1
        continue; % no solution: 'a' does not live in the column space of M
    elseif rank([M a]) ~= rank(M)
        throw( MException(1, 'rank([M a]) < rank(M) || rank([M a]) > rank(M) + 1') );
    end
    
    % if linear transformation M squishs the space into a lower dimension,
    % we use Moore-Penrose to solve the linear system; otherwise, standard
    % inverse matrix is used.
    if is_equal(det(M), 0)
        s = pinv(M) * a;
    else
        s = M \ a;
    end
    
    lambda = s(1:gutters_N);
    b = s(gutters_N + 1);
    
    correct = 0;
    for i = 1:train_data_N
        decision = 0;
        for j = 1:gutters_N
            decision = decision + lambda(j) * yg(j) * kernel( transpose( xg(j, :) ), transpose( x(i, :) ) );
        end
        decision = decision + b;
        predict = 0;
        if is_equal(decision, 1) || decision > 1
            predict = 1;
        elseif is_equal(decision, -1) || decision < -1
            predict = -1;
        end
        if predict == r(i)
            correct = correct + 1;
        end
    end
    
    f = 0;
    for i = 1:gutters_N
        for j = 1:gutters_N
            f = f + lambda(i) * lambda(j) * yg(i) * yg(j) * kernel( transpose( xg(i, :) ), transpose( xg(j, :) ) );
        end
    end
    if correct > train_correct || (correct == train_correct && f < f_min)
        train_correct = correct;
        f_min = f;
        train_xg = xg;
        train_yg = yg;
        train_lambda = lambda;
        train_b = b;
    end
end

save('train.mat', 'train_xg', 'train_yg', 'train_lambda', 'train_b');