% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4. 'cell' in this context
%    nothing to do with the Matlab data structue of cell(). It is simply
%    the terminology used in the feature literature to describe the spatial
%    bins where gradient distributions will be described.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

%Placeholder that you can delete. Empty features.
features = [];
expected_patch_size = feature_width * feature_width;
n = size(x,1);
% testing with normalized patches
% for i=1:n
%     xmin = floor(max(0, x(i) - feature_width * 0.5));
%     ymin = floor(max(0, y(i) - feature_width * 0.5));
%     patch = imcrop(image, [xmin ymin feature_width-1 feature_width-1]);
%     patch_size = numel(patch);
%     patch = reshape(patch, 1, patch_size);
%     if (patch_size < expected_patch_size)
%         patch = padarray(patch, [0 expected_patch_size - patch_size], 0, 'post');
%     end
%     if i>1 && size(features, 2) ~= size(patch, 2)
%         size(patch)
%     end
%     features = [features; patch];
% end
% 
Dx = [-1 0 1; -1 0 1; -1 0 1];
Dy = Dx';
expected_patch_size = [feature_width feature_width];
for i=1:n
    xmin = floor(max(0, x(i) - feature_width * 0.5));
    ymin = floor(max(0, y(i) - feature_width * 0.5));
    patch = imcrop(image, [xmin ymin feature_width-1 feature_width-1]);
    patch_size = size(patch);
    if ~isequal(patch_size, expected_patch_size)
        patch = padarray(patch, expected_patch_size - patch_size, 0, 'post');
    end
    % Compute gradient in the patch of size feature_width-by-feature_width
    Px = conv2(patch, Dx, 'same');
    Py = conv2(patch, Dy, 'same');
    P_o = atan2(Py, Px); % orientation of gradients
    P_m = sqrt(Px.^2 + Py.^2); % magnitude of gradients
    falloff_filter = fspecial('gaussian', feature_width, 8);
    P_m = conv2(P_m, falloff_filter, 'same'); % smoothen the magnitudes by gaussian weighting
    
    % Now bin the gradients into quadrants of size feature_width/4-by-feature_width/4
    quadrant_size = feature_width/4;
    orientations = linspace(-pi, pi, 8); % possible orientations
    
    % initialize descriptor vector to zeros
    descriptor = zeros(feature_width/4, feature_width/4, 8); % each quadrant is split into 8 orientation bins
    for j=1:feature_width
        for k = 1:feature_width
            % find closest orientation
            [dif index] = min(abs(orientations-P_o(j, k)));
            j_ = ceil(j/4);
            k_ = ceil(k/4);
            descriptor(j_, k_, index) = descriptor(j_, k_, index) + P_m(j, k);
            % instead weight it so that the closest ones get the most value
            weights = exp(-abs(orientations-P_o(j,k)));
            %weights = weights/norm(weights); % normalize the weights
            tempzeros = zeros(size(descriptor)); % crappy hack
            tempzeros(j_, k_, 1:8) = P_m(j,k) * weights;
            % perform linear interpolation to neighbor bins
            j__ = j_ + sign(cos(P_o(j,k)));
            k__ = k_ + sign(sin(P_o(j,k)));
            if (j__ > 0 && j__ <= feature_width/4 && k__ > 0 && k__ <= feature_width/4)
                tempzeros(j__, k__, 1:8) = P_m(j,k) * weights * 0.1;
            end
            descriptor = descriptor + tempzeros; % add gaussian weighted value to bin
        end
    end
    descriptor = reshape(descriptor, 1, numel(descriptor));
    descriptor = descriptor/norm(descriptor); % normalize
    %descriptor = min(descriptor, 0.7); % clamp
    %descriptor = descriptor/norm(descriptor); % re-normalize
    features = [features; descriptor];
end

end








