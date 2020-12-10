function ret = kernel(a, b)
    if Def.KERNEL == 0
        ret = dot(a, b);
    elseif Def.KERNEL == 1 % polynomial kernel
        ret = (dot(a, b) + 1)^2;
    else
        throw( MException(2, 'invalid kernel') );
    end
end