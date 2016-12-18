% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or (b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.

% modification, threshold = for corner detection, 0-1
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width, max_no_points)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Placeholder that you can delete -- random points
close all;

kappa = 0.04; % 0.04 - 0.15 sensitivity parameter
threshold = 0.00001;

Dx = [-1 0 1; -1 0 1; -1 0 1];
Dy = Dx';

Ix = conv2(image, Dx, 'same');
Iy = conv2(image, Dy, 'same');
g = fspecial('gaussian', 3, 0.5); % small smoothing kernel
Ix = conv2(Ix, g, 'same');
Iy = conv2(Iy, g, 'same');

I_m2 = Ix.^2 + Iy.^2; %square magnitudes of gradient
W = fspecial('gaussian', feature_width, 2); % weighting kernel

Ixx = Ix .^ 2;
Ixy = Ix .* Iy;
Iyy = Iy .^ 2;

% convolve with a larger Gaussian
Ixx = conv2(Ixx, W, 'same'); 
Ixy = conv2(Ixy, W, 'same');
Iyy = conv2(Iyy, W, 'same');

A = zeros(size(image));
A = (Ixx .* Iyy - Ixy .^ 2) - kappa * (Ixx + Iyy).^2; % Harris matrix measure


%A([1:(feature_width-1), end-feature_width:end], :) = 0;
%A(:,[1:(feature_width-1),end-feature_width:end]) = 0;
%R_maximas = colfilt(A,[feature_width/4, feature_width/4],'sliding',@max);
%non_suppressed_corners = A.*(A == R_maximas);
%[y, x] = find(non_suppressed_corners);


sr = feature_width/2; % suppression radius for anms
mask = ones(sr);
B = ordfilt2(A, sr * sr - 1, mask); % selects the second-last value in the ascending order of elements
max_peaks = (A>B) & (A>threshold); % sets all peaks to 1 and others to 0
[y, x] = find(max_peaks); % Find row,col coords.
indices = sub2ind(size(I_m2), y, x);
confidence = I_m2(indices);

% check if confidence measure is good
[confidence, ind] = sort(confidence, 'descend');
y = y(ind, :);
x = x(ind, :);
numel(x)
num_el = min(max_no_points, numel(x));
x = x(1:num_el);
y = y(1:num_el);

end