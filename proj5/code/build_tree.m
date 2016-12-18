function tree_model = build_tree(X, Y, leaf_size)
    n = size(X, 1);
    if  n == 0
        tree_model = {{'End', -1}, -1, -1, -1, -1, -1, 1};
        return;
    end
    if n <= leaf_size
        tree_model = {{'Leaf', X}, Y, -1, -1, -1, -1, 1};
        return;
    end
    if all(Y(:) == Y(1))
        tree_model =  {{'Leaf', X}, Y(1), -1, -1, -1, -1, 1};
        return;
    end
    i = randi(size(X, 2));  % random feature to split tree upon
    split_value = mean(X(randi(n, 1, 2), i));   % take mean of above feature from two random points as split_value
    left_tree_indices = find(X(:, i) <= split_value);
    left_tree = build_tree(X(left_tree_indices, :), Y(left_tree_indices), leaf_size);
    right_tree_indices = setdiff((1:n), left_tree_indices);
    right_tree = build_tree(X(right_tree_indices, :), Y(right_tree_indices), leaf_size);
    
    l_info = left_tree{1};
    r_info = right_tree{1};
    l_size = 0;
    r_size = 0;
    if strcmp(l_info{1}, 'End') == 0
        l_size = left_tree{7};
    end
    if strcmp(r_info{1}, 'End') == 0
        r_size = right_tree{7};
    end
    
    tree_model = {{'Node', i}, split_value, 1, l_size+1, left_tree, right_tree, l_size+r_size+1};
end