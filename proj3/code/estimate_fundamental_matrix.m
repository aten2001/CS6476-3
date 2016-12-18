% Fundamental Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Returns the camera center matrix for a given projection matrix

% 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
% 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
% 'F_matrix' is 3x3 fundamental matrix

% Try to implement this function as efficiently as possible. It will be
% called repeatly for part III of the project

function [ F_matrix ] = estimate_fundamental_matrix(Points_a,Points_b)
n = size(Points_a, 1);
A = zeros(n, 9);

for i=1:n
    x1 = [Points_a(i, 1:2) 1];
    x2 = [Points_b(i, 1:2) 1];
    
    uv = x2' * x1;
    A(i, :) = reshape(uv, 1, []);
end

% Solve for F
[U, S, V] = svd(A);
f = V(:, end);
F_matrix = reshape(f, [], 3);
%Rank reduction
[U, S, V] = svd(F_matrix);
S(3, 3) = 0;
F_matrix = U*S*V';




end

