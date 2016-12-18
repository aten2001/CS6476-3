% RANSAC Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Find the best fundamental matrix using RANSAC on potentially matching
% points

% 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
% matching points from pic_a and pic_b. Each row is a correspondence (e.g.
% row 42 of matches_a is a point that corresponds to row 42 of matches_b.

% 'Best_Fmatrix' is the 3x3 fundamental matrix
% 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
% of 'matches_a' and 'matches_b') that are inliers with respect to
% Best_Fmatrix.

% For this section, use RANSAC to find the best fundamental matrix by
% randomly sample interest points. You would reuse
% estimate_fundamental_matrix() from part 2 of this assignment.

% If you are trying to produce an uncluttered visualization of epipolar
% lines, you may want to return no more than 30 points for either left or
% right images.

function [ Best_Fmatrix, inliers_a, inliers_b] = ransac_fundamental_matrix(matches_a, matches_b)

N = size(matches_a,1);
num_pairs = 40; % no. of correspondences used for estimating fundamental matrix
best_no_inliers = 0; %initialize best no. of inliers to 0
threshold = 0.000000001; % threshold below which a pair is considered an inlier
max_iterations = 4000;

matches_original_a = matches_a;
matches_original_b = matches_b;

% normalize the coordinates
% compute mean of all points
mean_x_a = mean(matches_a(:, 1));
mean_y_a = mean(matches_a(:, 2));
mean_x_b = mean(matches_b(:, 1));
mean_y_b = mean(matches_b(:, 2));
% translate all points to the mean value; the below implementation was
% faster than repmat
matches_a(:, 1) = matches_a(:, 1) - mean_x_a;
matches_a(:, 2) = matches_a(:, 2) - mean_y_a;
matches_b(:, 1) = matches_b(:, 1) - mean_x_b;
matches_b(:, 2) = matches_b(:, 2) - mean_y_b;
% compute avg squared distance from center of new coordinates
avg_squared_distance_a = mean(sum(matches_a.^2, 2));
avg_squared_distance_b = mean(sum(matches_b.^2, 2));
% normalize so that asd is 2
scale_a = sqrt(2/avg_squared_distance_a);
scale_b = sqrt(2/avg_squared_distance_b);
matches_a = matches_a.*scale_a;
matches_b = matches_b.*scale_b;

for j=1:max_iterations
    indices = randi(N, 1, num_pairs);
    x = matches_a(indices, 1:2);
    x_orig = matches_original_a(indices, 1:2);
    x2 = matches_b(indices, 1:2);
    x2_orig = matches_original_b(indices, 1:2);
    
    F_matrix = estimate_fundamental_matrix(x, x2);
    error = zeros(num_pairs,1);
    for j=1:num_pairs
        error(j) = [x(j,:) 1] * F_matrix * [x2(j,:) 1]';
    end
    % count inliers
    indices = find(error < threshold);
    no_inliers = length(indices);
    if (no_inliers > best_no_inliers)
        Best_Fmatrix = F_matrix;
        best_no_inliers = no_inliers;
        %inliers_a = x(indices, 1:2);
        %inliers_b = x2(indices, 1:2);
        inliers_a = x_orig(indices, 1:2);
        inliers_b = x2_orig(indices, 1:2);
    end
end

% renormalize the fundamental matrix
T_a = [scale_a 0 0; 0 scale_a 0; 0 0 1] * [1 0 -mean_x_a; 0 1 -mean_y_a; 0 0 1];
T_b = [scale_b 0 0; 0 scale_b 0; 0 0 1] * [1 0 -mean_x_b; 0 1 -mean_y_b; 0 0 1];
Best_Fmatrix = T_b' * Best_Fmatrix * T_a;


if (best_no_inliers > 30)
    inliers_a = inliers_a(1:30, :);
    inliers_b = inliers_b(1:30, :);
end

end