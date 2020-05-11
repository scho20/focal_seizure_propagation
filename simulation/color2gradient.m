function cmap = color2gradient(color, npoints)
    function v = to_column_vector(v)
    v = v(:);
    if size(v,2) > 1
        v = v';
    end
    end
cmap = arrayfun(@(ci) to_column_vector(linspace(ci, 1, npoints)), color, 'uni', 0);
cmap = horzcat(cmap{:});
end
