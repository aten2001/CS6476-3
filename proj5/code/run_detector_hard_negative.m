% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function features_hard_neg = .... 
    run_detector_hard_negative(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression. Err
% on the side of having a low confidence threshold (even less than zero) to
% achieve high enough recall.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

ts = feature_params.template_size;
cz = feature_params.hog_cell_size;
cz2 = ts/cz;

negative_threshold = 0.1;
features_hard_neg = [];


for i = 1:length(test_scenes)
      
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);

    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    [iwidth, iheight, ~] = size(img);
    scale_factor = 1;
    max_iterations = 1; % max number of scaling down iterations in case of large images
    i_min_dim = min(iwidth, iheight);
    iter = 0;
    while i_min_dim*scale_factor > ts && iter < max_iterations % smallest dimension must at least be template size
        resized_img = imresize(img, scale_factor);

        hog_features = vl_hog(im2single(resized_img), cz);
        [ia, ib, ic] = size(hog_features);
        
        for j=1:ia-cz2+1
            for k=1:ib-cz2+1
                hog = hog_features(j:j+cz2-1, k:k+cz2-1, 1:ic);
                hog = reshape(hog, 1, []);
                v = hog * w + b;
                if (v >= negative_threshold)
                    features_hard_neg = [features_hard_neg; hog];
                end
            end
        end

        scale_factor = scale_factor * 0.9;
        iter = iter + 1;
    end
end