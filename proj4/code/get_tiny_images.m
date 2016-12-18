% Starter code prepared by James Hays for Computer Vision

%This feature is inspired by the simple tiny images used as features in 
%  80 million tiny images: a large dataset for non-parametric object and
%  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
%  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
%  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

function image_feats = get_tiny_images(image_paths)
% image_paths is an N x 1 cell array of strings where each string is an
%  image path on the file system.
% image_feats is an N x d matrix of resized and then vectorized tiny
%  images. E.g. if the images are resized to 16x16, d would equal 256.

% To build a tiny image feature, simply resize the original image to a very
% small square resolution, e.g. 16x16. You can either resize the images to
% square while ignoring their aspect ratio or you can crop the center
% square portion out of each image. Making the tiny images zero mean and
% unit length (normalizing them) will increase performance modestly.

% suggested functions: imread, imresize

square_width = 16;  % tiny image size
d = square_width * square_width;
make_zero_mean = 1; % set to 1 if you want to make the images zero mean
% and unit length

N = numel(image_paths);
image_feats = zeros(N, d);
if (make_zero_mean~=1)
    for i=1:N
        arg = image_paths(i);
        image_feats(i, :) = reshape(imresize(im2single(imread(arg{1})), [square_width square_width]), 1, []);
    end
else
    for i=1:N
        arg = image_paths(i);
        image = reshape(imresize(im2single(imread(arg{1})), [square_width square_width]), 1, []);
        image = image - mean(image(:));
        image_feats(i, :) = image/std(image);
    end
end







