function results = cv_reconstruction(X, dim, pars)
%CV_RECONSTRUCTION Cross-validated PCA reconstruction of neural activity.
%
%   RESULTS = diagnostics.compute.cv_reconstruction(X, DIM, PARS) evaluates
%   how well the latent embedding reconstructs held-out neural activity
%   using k-fold cross-validation. The embedding is PCA; all preprocessing
%   (z-scoring and PCA fit) is done on the training fold only to avoid
%   data leakage.
%
%   Inputs
%   ------
%   X   : T x N matrix (T samples/time-bins, N neurons/features). No NaN.
%   dim : positive integer. Number of latent dimensions to keep.
%   pars: struct with fields (see diagnostics.pars.CVReconstruction):
%         .kfold   (default 5)
%         .rngSeed (default 0)
%         .zscore  (default true)
%
%   Outputs
%   -------
%   results : struct with fields
%       .reconCorrMean    - mean Pearson r across folds (scalar).
%       .reconCorrPerFold - (1 x kfold) Pearson r on held-out data per fold.
%       .R2Mean           - mean R^2 across folds (scalar).
%       .R2PerFold        - (1 x kfold) R^2 per fold.
%       .corrPerNeuron    - (1 x N) mean Pearson r per neuron across folds.
%       .dim              - dim used.
%       .kfold            - kfold used.
%       .rngSeed          - rngSeed used.
%
%   Notes
%   -----
%   * Pearson r is computed on the vectorised held-out data x_test(:).
%   * R^2 = 1 - SS_res / SS_tot (may be negative).
%   * Requires only base MATLAB (no toolboxes).
%
%   Example
%   -------
%   pars = diagnostics.pars.CVReconstruction();
%   res  = diagnostics.compute.cv_reconstruction(X, 5, pars);
%   fprintf('Mean Pearson r = %.3f\n', res.reconCorrMean);
%
%   See also diagnostics.pars.CVReconstruction

% --- Input validation ---
if ~ismatrix(X) || ~isnumeric(X)
    error('diagnostics:cv_reconstruction:badInput', ...
        'X must be a 2-D numeric matrix (T x N).');
end
if any(isnan(X(:)))
    error('diagnostics:cv_reconstruction:nanData', ...
        'X contains NaN values.');
end
if ~isscalar(dim) || dim < 1 || dim ~= round(dim)
    error('diagnostics:cv_reconstruction:badDim', ...
        'dim must be a positive integer scalar.');
end

if nargin < 3 || isempty(pars)
    pars = diagnostics.pars.CVReconstruction();
end
kfold   = pars.kfold;
rngSeed = pars.rngSeed;
doZscore = pars.zscore;

if ~isempty(rngSeed)
    rng(rngSeed, 'twister');
end

[T, N] = size(X);
dim    = min(dim, min(T, N) - 1);

% --- k-fold partition ---
foldIdx = i_kfold_indices(T, kfold);

reconCorrPerFold  = zeros(1, kfold);
R2PerFold         = zeros(1, kfold);
corrPerNeuronAcc  = zeros(kfold, N);

for ff = 1:kfold
    testMask  = (foldIdx == ff);
    trainMask = ~testMask;

    Xtrain = X(trainMask, :);
    Xtest  = X(testMask,  :);

    % --- Train-only z-score (no leakage) ---
    if doZscore
        mu_tr = mean(Xtrain, 1);
        sd_tr = std(Xtrain, 0, 1);
        sd_tr(sd_tr == 0) = 1;   % avoid divide-by-zero for silent neurons
        Xtrain = (Xtrain - mu_tr) ./ sd_tr;
        Xtest  = (Xtest  - mu_tr) ./ sd_tr;
    end

    % --- Fit PCA on training data ---
    [W, Xtrain_proj] = i_pca_fit(Xtrain, dim);   % W: N x dim

    % --- Project and reconstruct test data ---
    Xtest_proj = Xtest * W;                        % Ttest x dim
    Xhat_test  = Xtest_proj * W';                  % Ttest x N

    % --- Reconstruction scores ---
    reconCorrPerFold(ff)    = i_pearson(Xtest(:), Xhat_test(:));
    R2PerFold(ff)           = i_r2(Xtest(:), Xhat_test(:));
    corrPerNeuronAcc(ff, :) = i_pearson_per_col(Xtest, Xhat_test);
end

results.reconCorrMean    = mean(reconCorrPerFold);
results.reconCorrPerFold = reconCorrPerFold;
results.R2Mean           = mean(R2PerFold);
results.R2PerFold        = R2PerFold;
results.corrPerNeuron    = mean(corrPerNeuronAcc, 1);
results.dim              = dim;
results.kfold            = kfold;
results.rngSeed          = rngSeed;
end

% =========================================================================
%  Local helpers
% =========================================================================

function foldIdx = i_kfold_indices(T, k)
% Return a T x 1 vector of fold assignments 1..k (stratified by index).
foldIdx = zeros(T, 1);
idx     = randperm(T);
folds   = floor(T / k);
for ff = 1:k
    if ff < k
        foldIdx(idx((ff-1)*folds + (1:folds))) = ff;
    else
        foldIdx(idx((ff-1)*folds + 1 : end)) = ff;
    end
end
end

function [W, scores] = i_pca_fit(X, dim)
% Fit PCA on X (already z-scored), return loading matrix W (N x dim).
X = X - mean(X, 1);              % centre
[U, ~, ~] = svd(X, 'econ');      % U: T x rank
C = (X' * X) / (size(X, 1) - 1);
[V, ~]    = eig(C);               % V: N x N, eigenvalues ascending
V = fliplr(V);                    % descending order
W = V(:, 1:dim);                  % N x dim
scores = X * W;                   % T x dim
end

function r = i_pearson(a, b)
% Pearson correlation between two vectors.
a = a(:); b = b(:);
a = a - mean(a); b = b - mean(b);
na = norm(a); nb = norm(b);
if na == 0 || nb == 0
    r = 0;
    return;
end
r = (a' * b) / (na * nb);
end

function r2 = i_r2(y, yhat)
% Coefficient of determination R^2.
ss_res = sum((y - yhat).^2);
ss_tot = sum((y - mean(y)).^2);
if ss_tot == 0
    r2 = 0;
else
    r2 = 1 - ss_res / ss_tot;
end
end

function rv = i_pearson_per_col(A, B)
% Pearson r between corresponding columns of A and B.
N  = size(A, 2);
rv = zeros(1, N);
for nn = 1:N
    rv(nn) = i_pearson(A(:, nn), B(:, nn));
end
end
