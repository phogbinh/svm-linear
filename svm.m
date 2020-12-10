train_data = [1 2 -1; 2 1 -1; 3 2 1; 4 3 1];
train_data_N = numel( train_data(:, 1) );

x = train_data(:, 1:2);
r = train_data(:, 3);

train_correct = 0;
f_min = inf; % optimization for widest street
final_w = nan;
final_b = nan;
for combination = 1:( bitshift(1, train_data_N) - 1 )
    gutters = [];
    for i = 1:train_data_N
        if bitget(combination, i)
            gutters(end + 1, :) = train_data(i, :);
        end
    end
    gutters_N = numel( gutters(:, 1) );
    
    xg = gutters(:, 1:2);
    y = gutters(:, 3);
    
    M = zeros(gutters_N + 1, gutters_N + 1);
    M(1, :) = [transpose(y) 0];
    for i = 2:(gutters_N + 1)
        for j = 1:gutters_N
            M(i, j) = dot( xg(i-1, :), xg(j, :) ) * y(j);
        end
        M(i, gutters_N + 1) = 1;
    end

    a = [0; y];
    
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
    sub_w = lamda .* y .* xg;
    w = zeros(2, 1);
    for i = 1:gutters_N
        w = w + transpose( sub_w(i) );
    end
    
    correct = 0;
    for i = 1:train_data_N
        decision = dot( w, transpose( x(i, :) ) ) + b;
        predict = 0;
        if decision >= 1
            predict = 1;
        elseif decision <= -1
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
        final_w = w;
        final_b = b;
    end
end

test_data = [1 1 -1; 2 0 -1; 3 1 1; 4 2 1];
test_data_N = numel( test_data(:, 1) );

test_correct = 0;
for i = 1:test_data_N
    decision = dot( final_w, transpose( test_data(i, 1:2) ) ) + final_b;
    predict = -1;
    if decision >= 0 % points on the separator line set positive
        predict = 1;
    end
    if predict == test_data(i, 3)
        test_correct = test_correct + 1;
    end
end

accuracy = test_correct / test_data_N