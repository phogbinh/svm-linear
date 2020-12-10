w = matfile('train.mat').train_w;
b = matfile('train.mat').train_b;

test_data = [1 1 -1; 2 0 -1; 3 1 1; 4 2 1];
test_data_N = numel( test_data(:, 1) );

test_correct = 0;
for i = 1:test_data_N
    decision = dot( w, transpose( test_data(i, 1:2) ) ) + b;
    predict = -1;
    if decision >= 0 % points on separator line are set positive
        predict = 1;
    end
    if predict == test_data(i, 3)
        test_correct = test_correct + 1;
    end
end

accuracy = test_correct / test_data_N