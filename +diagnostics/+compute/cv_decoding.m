function results = cv_decoding(X, y, dim, pars)
%CV_DECODING Cross-validated decoding of labels from PCA latent space.
%
%   RESULTS = diagnostics.compute.cv_decoding(X, Y, DIM, PARS) embeds X
%   into a DIM-dimensional PCA space and decodes labels Y using a
%   nearest-centroid classifier. Embedding and any preprocessing are fit
%   on training folds only.
%
%   Inputs
%   ------
%   X   : T x N matrix (T samples, N neurons). No NaN.
%   y   : T x 1 integer or categorical label vector.
%   dim : positive integer, number of latent dimensions.
%   pars: struct (see diagnostics.pars.CVDecoding):
%         .kfold   (default 5)
%         .rngSeed (default 0)
%         .decoder 'nearestCentroid' (default; no toolbox needed)
%
%   Outputs
%   -------
%   results : struct with fields
%       .accMean       - mean accuracy across folds (scalar).
%       .accPerFold    - (1 x kfold) per-fold accuracy.
%       .balAccMean    - mean balanced accuracy across folds.
%       .balAccPerFold - (1 x kfold) per-fold balanced accuracy.
%       .confMat       - (nClasses x nClasses) confusion matrix (sum over folds).
%       .classes       - unique class labels.
%       .dim           - dim used.
%       .kfold         - kfold used.
%       .rngSeed       - rngSeed used.
%
%   Notes
%   -----
%   * Balanced accuracy = mean per-class recall (robust to class imbalance).
%   * Only classification (categorical/integer y) is supported; pass
%     continuous y to obtain Pearson r-based regression scores separately.
%   * Requires only base MATLAB.
%
%   Example
%   -------
%   pars = diagnostics.pars.CVDecoding();
%   res  = diagnostics.compute.cv_decoding(X, y, 5, pars);
%   fprintf('Mean accuracy = %.2f%%\n', res.accMean * 100);
%
%   See also diagnostics.pars.CVDecoding,
%            diagnostics.compute.permutation_test

% --- Input validation ---
if ~ismatrix(X) || ~isnumeric(X)
    error('diagnostics:cv_decoding:badInput', 'X must be a 2-D numeric matrix.');
end
y = y(:);
if size(X, 1) ~= numel(y)
    error('diagnostics:cv_decoding:sizeMismatch', ...
        'X and y must have the same number of rows.');
end
if any(isnan(X(:)))
    error('diagnostics:cv_decoding:nanData', 'X contains NaN values.');
end

if nargin < 4 || isempty(pars)
    pars = diagnostics.pars.CVDecoding();
end
kfold   = pars.kfold;
rngSeed = pars.rngSeed;
decoder = pars.decoder;

if ~isempty(rngSeed)
    rng(rngSeed, 'twister');
end

[T, N]  = size(X);
dim     = min(dim, min(T, N) - 1);
classes = unique(y);
nClass  = numel(classes);

% --- k-fold partition ---
foldIdx = i_kfold_stratified(y, kfold);

accPerFold    = zeros(1, kfold);
balAccPerFold = zeros(1, kfold);
confMatAcc    = zeros(nClass, nClass);

for ff = 1:kfold
    testMask  = (foldIdx == ff);
    trainMask = ~testMask;

    Xtr = X(trainMask, :);  ytr = y(trainMask);
    Xte = X(testMask,  :);  yte = y(testMask);

    % Train-only z-score
    mu = mean(Xtr, 1);
    sd = std(Xtr,  0, 1);
    sd(sd == 0) = 1;
    Xtr = (Xtr - mu) ./ sd;
    Xte = (Xte - mu) ./ sd;

    % Fit PCA on training data
    [W] = i_pca_fit(Xtr, dim);   % N x dim

    % Project both folds
    Ztr = Xtr * W;
    Zte = Xte * W;

    % Decode
    switch decoder
        case 'nearestCentroid'
            yhat = i_nearest_centroid(Ztr, ytr, Zte, classes);
        otherwise
            error('diagnostics:cv_decoding:unknownDecoder', ...
                'decoder ''%s'' is not supported.', decoder);
    end

    % Score
    accPerFold(ff)    = mean(yhat == yte);
    [balAccPerFold(ff), cm] = i_balanced_acc(yte, yhat, classes);
    confMatAcc = confMatAcc + cm;
end

results.accMean       = mean(accPerFold);
results.accPerFold    = accPerFold;
results.balAccMean    = mean(balAccPerFold);
results.balAccPerFold = balAccPerFold;
results.confMat       = confMatAcc;
results.classes       = classes;
results.dim           = dim;
results.kfold         = kfold;
results.rngSeed       = rngSeed;
end

% =========================================================================
%  Local helpers
% =========================================================================

function foldIdx = i_kfold_stratified(y, k)
% Stratified k-fold: attempts equal class representation in each fold.
classes  = unique(y);
T        = numel(y);
foldIdx  = zeros(T, 1);
for cc = 1:numel(classes)
    idx = find(y == classes(cc));
    idx = idx(randperm(numel(idx)));
    n   = numel(idx);
    for ff = 1:k
        lo = round((ff-1)*n/k) + 1;
        hi = round(ff*n/k);
        foldIdx(idx(lo:hi)) = ff;
    end
end
% Handle any zeros left (shouldn't happen with above logic)
unassigned = find(foldIdx == 0);
for ii = 1:numel(unassigned)
    foldIdx(unassigned(ii)) = mod(ii-1, k) + 1;
end
end

function W = i_pca_fit(X, dim)
% Fit PCA on centred X; return loadings W (N x dim).
X = X - mean(X, 1);
C = (X' * X) / max(size(X, 1) - 1, 1);
[V, D] = eig(C);
[~, ord] = sort(diag(D), 'descend');
V = V(:, ord);
W = V(:, 1:dim);
end

function yhat = i_nearest_centroid(Ztrain, ytrain, Ztest, classes)
% Nearest-centroid classifier. No toolboxes required.
nClass = numel(classes);
centroids = zeros(nClass, size(Ztrain, 2));
for cc = 1:nClass
    centroids(cc, :) = mean(Ztrain(ytrain == classes(cc), :), 1);
end
% Compute squared distances to each centroid
nTest = size(Ztest, 1);
dists = zeros(nTest, nClass);
for cc = 1:nClass
    diff = Ztest - centroids(cc, :);
    dists(:, cc) = sum(diff.^2, 2);
end
[~, cidx] = min(dists, [], 2);
yhat = classes(cidx);
end

function [balAcc, cm] = i_balanced_acc(ytrue, ypred, classes)
% Compute balanced accuracy and confusion matrix.
nClass = numel(classes);
cm     = zeros(nClass, nClass);
for ii = 1:nClass
    for jj = 1:nClass
        cm(ii, jj) = sum(ytrue == classes(ii) & ypred == classes(jj));
    end
end
recall  = zeros(nClass, 1);
for ii = 1:nClass
    n = sum(ytrue == classes(ii));
    if n > 0
        recall(ii) = cm(ii, ii) / n;
    end
end
balAcc = mean(recall);
end
