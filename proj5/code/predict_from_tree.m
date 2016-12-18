function result = predict_from_tree(tree_model, X)
    % traverse the tree model and predict the output for input X
    t_info = tree_model{1};
    obj_type = t_info{1};
    while strcmp(obj_type, 'Leaf') == 0 && strcmp(obj_type, 'End') == 0
       i = t_info{2};
       split_value = tree_model{2};
       if X(i) <= split_value
           if iscell(tree_model{5})
               tree_model = tree_model{5};
           else
               tree_model = tree_model{6};
           end
       else
           if iscell(tree_model{6})
               tree_model = tree_model{6};
           else
               tree_model = tree_model{5};
           end
       end
       t_info = tree_model{1};
       obj_type = t_info{1};
       result = mean(tree_model{2});
    end
end