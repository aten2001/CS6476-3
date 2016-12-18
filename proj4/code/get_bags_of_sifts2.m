% Modified version of Starter code prepared by James Hays for Computer Vision

function image_feats = get_bags_of_sifts2(image_paths, vocab, M)
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

% M is a positive integer << vocabulary_size used to defining the local
% linear subspace
%

% This function constructs LLC encoded bag of features as per 
% http://www.robots.ox.ac.uk/~vgg/publications/2011/Chatfield11/chatfield11.pdf

%load('vocab.mat');
% vocab = single(vocab'); % build_vocabulary transposes it
d = size(vocab, 2); % feature vector size
vocab_size = size(vocab, 1);
N = numel(image_paths);
image_feats = zeros(N, vocab_size);
step_size = 8;
lambda = 1e-3;
alpha_value = @(x, beta) inv(x'*x + beta*eye(size(x, 2)));
for i=1:N
        i % for knowing current iteration in console
        arg = image_paths(i);
        [locations, sift_feat] = vl_dsift(im2single(imread(arg{1})), 'Fast', 'Step', step_size);
        sift_feat = single(sift_feat);
        num_feats = size(sift_feat, 2);
        
        llc_encoding = LLC_coding_appr(vocab, sift_feat', M, lambda);
        image_feats(i, :) = max(llc_encoding);
        
        %distance_matrix = vl_alldist2(sift_feat, vocab);
        %[sorted_distances, I] = sort(distance_matrix, 2);
        %m_indices = I(:, 1:M);
        %vocab_subset = vocab(:, m_indices);
        %B = reshape(vocab_subset, d, num_feats, M);    % B as per paper
        % original llc encoding implementation
        %llc_encode = zeros(vocab_size, 1);
        %for j=1:num_feats
        %    Binv = pinv(reshape(B(:, j, 1:M), d, M));
        %    alpha = Binv * sift_feat(:, j);
        %    llc_encode(m_indices(j, :), j) = alpha;
        %end
        %llc_encoded_vector = max(llc_encode, [], 2);
        %image_feats(i, :) = llc_encoded_vector./sum(llc_encoded_vector); %normalized llc_encoded_vectors
end

end