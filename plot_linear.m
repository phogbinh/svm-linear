% ax + by + c = 0
function [x, y] = plot_linear(a, b, c, xmin, xmax, ymin, ymax)
    if b == 0
        x = (-c / a) * ones(1, ymax - ymin + 1);
        y = ymin:ymax;
    else
        x = xmin:xmax;
        y = (-c - a * x) / b;
    end
end