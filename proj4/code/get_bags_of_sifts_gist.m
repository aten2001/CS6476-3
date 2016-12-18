function image_feats = get_bags_of_sifts_gist(image_paths, vocab)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every run.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram
% ('vocab_size') below.

% This function augments the bag of SIFTs representation with GIST features
% as per the paper http://people.csail.mit.edu/torralba/code/spatialenvelope/
% It uses code provided by the authors here http://people.csail.mit.edu/torralba/code/spatialenvelope/

vocab = single(vocab'); % build_vocabulary transposes it
vocab_size = size(vocab, 2);
N = numel(image_paths);

% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Pre-allocate gist:
Nfeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;
gist = zeros([N Nfeatures]); 

% Load first image and compute gist:
arg = image_paths(1);
img = im2single(imread(arg{1}));
[gist(1, :), param] = LMgist(img, '', param); % first call

sift_feats = zeros(N, vocab_size);
step_size = 10;
for i=1:N
        i % for knowing current iteration in console
        arg = image_paths(i);
        img = im2single(imread(arg{1}));
        [locations, sift_feat] = vl_dsift(img, 'Fast', 'Step', step_size);
        sift_feat = single(sift_feat);
        distance_matrix = vl_alldist2(sift_feat, vocab);
        [A, I] = min(distance_matrix, [], 2);
        [bag_of_sifts, ~, bins] = histcounts(I,'BinLimits', [1, vocab_size], 'BinMethod', 'integers', 'Normalization', 'probability');
        sift_feats(i, :) = bag_of_sifts; %normalized histogram
        if i>1
            gist(i, :) = LMgist(img, '', param);
        end
end
image_feats = [sift_feats gist];
end