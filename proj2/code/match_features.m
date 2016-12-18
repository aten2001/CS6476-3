% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the features as additional inputs.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% Placeholder that you can delete. Random matches and confidences

threshold = 0.88; % threshold for nnd for matching. any nnd < threshold is considered a match
f1_size = size(features1, 1);
f2_size = size(features2, 1);

distances = pdist2(features1, features2, 'euclidean'); % pair-wise distances between features in f1 and f2
[sorted_distances, sorted_indices] = sort(distances, 2); % distances to each feature in f2 for f1 sorted
matches = [];
confidences = [];
for i=1:f1_size
    nnd = sorted_distances(i, 1)/sorted_distances(i, 2);
    if nnd < threshold
        matches = [matches; i, sorted_indices(i, 1)]; 
        confidences = [confidences; 1 - nnd];
    end
end

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
matches = matches(ind,:);