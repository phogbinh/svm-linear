train_data = readmatrix('data/basic_train');
train_data_N = numel( train_data(:, 1) );

x = train_data(:, 1:2);
r = train_data(:, 3);

train_correct = 0;
f_min = inf; % optimization for widest street
train_w = nan;
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
            M(i, j) = dot( xg(i-1, :), xg(j, :) ) * yg(j);
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
    if det(M) == 0
        s = pinv(M) * a;
    else
        s = M \ a;
    end
    
    b = s(gutters_N + 1);
    
    lamda = s(1:gutters_N);
    sub_w = lamda .* yg .* xg;
    w = zeros(2, 1);
    for i = 1:gutters_N
        w = w + transpose( sub_w(i, :) );
    end
    
    correct = 0;
    for i = 1:train_data_N
        decision = dot( w, transpose( x(i, :) ) ) + b;
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
    
    f = dot(w, w);
    if correct > train_correct || (correct == train_correct && f < f_min)
        train_correct = correct;
        f_min = f;
        train_w = w;
        train_b = b;
    end
end

save('train.mat', 'train_w', 'train_b');