function results = dim_parallel_analysis(X, dims, pars, label)
%DIM_PARALLEL_ANALYSIS Shuffle-based dimension selection (parallel analysis).
%
%   RESULTS = diagnostics.compute.dim_parallel_analysis(X, DIMS, PARS)
%   selects the number of latent dimensions d* by comparing the PCA
%   eigenspectrum of the real data against a null distribution obtained
%   by shuffling.
%
%   The null preserves each neuron's marginal firing-rate distribution but
%   destroys cross-neuron covariance (mode='neuronwise', recommended) or
%   the temporal structure within each neuron (mode='rowperm').
%
%   Inputs
%   ------
%   X     : T x N matrix (T samples, N neurons/features). Must not contain
%           NaN.
%   dims  : vector of positive integers, e.g. 1:15. Eigenvalues for these
%           component indices are compared against the null.
%   pars  : struct with fields (see diagnostics.pars.ParallelAnalysis):
%           .nShuffle (default 200)
%           .alpha    (default 0.05)
%           .rngSeed  (default 0)
%           .mode     'neuronwise' | 'rowperm' (default 'neuronwise')
%           .verbose  logical (default true) - print progress
%   label : (optional) string label displayed in progress output, e.g.
%           'Animal.Session'.  Defaults to ''.
%
%   Outputs
%   -------
%   results : struct with fields
%       .eigReal      - (1 x numel(dims)) eigenvalues of real data.
%       .eigNull      - (nShuffle x numel(dims)) null eigenvalue matrix.
%       .nullQuantile - (1 x numel(dims)) (1-alpha) quantile of null.
%       .dimsTested   - dims vector used.
%       .dStar        - selected dimension (largest k where real > null).
%       .alpha        - alpha used.
%       .nShuffle     - nShuffle used.
%       .rngSeed      - rngSeed used.
%       .mode         - shuffle mode used.
%
%   Notes
%   -----
%   * X is z-scored once before eigendecomposition (using all-data
%     statistics). This is appropriate for dimension selection; the
%     class-method wrapper performs train-only z-scoring for CV tasks.
%   * Requires only base MATLAB (no Statistics Toolbox).
%
%   Example
%   -------
%   X = randn(200, 50);  % 200 time bins, 50 neurons
%   pars = diagnostics.pars.ParallelAnalysis();
%   res  = diagnostics.compute.dim_parallel_analysis(X, 1:10, pars);
%   fprintf('Selected dimension: %d\n', res.dStar);
%
%   See also diagnostics.pars.ParallelAnalysis

% --- Input validation ---
if ~ismatrix(X) || ~isnumeric(X)
    error('diagnostics:dim_parallel_analysis:badInput', ...
        'X must be a 2-D numeric matrix (T x N).');
end
if any(isnan(X(:)))
    error('diagnostics:dim_parallel_analysis:nanData', ...
        'X contains NaN values. Remove or impute before calling this function.');
end

dims = dims(:)';
if any(dims < 1) || any(dims ~= round(dims))
    error('diagnostics:dim_parallel_analysis:badDims', ...
        'dims must be a vector of positive integers.');
end

% Defaults
if nargin < 3 || isempty(pars)
    pars = diagnostics.pars.ParallelAnalysis();
end
if nargin < 4 || isempty(label)
    label = '';
end
nShuffle = pars.nShuffle;
alpha    = pars.alpha;
rngSeed  = pars.rngSeed;
mode     = pars.mode;
verbose  = isfield(pars,'verbose') && pars.verbose;

% --- RNG ---
if ~isempty(rngSeed)
    rng(rngSeed, 'twister');
end

[T, N] = size(X);

% Global z-score (appropriate for dimension selection)
X = zscore(X, 0, 1);  % zero-mean unit-variance per column

% Cap dims at min(T,N)-1.
% After mean-centering, the rank of X is at most min(T,N)-1, so at most
% that many non-zero eigenvalues exist.
maxDim = min(T, N) - 1;
dims   = dims(dims <= maxDim);
if isempty(dims)
    error('diagnostics:dim_parallel_analysis:badDims', ...
        'All requested dims exceed the data rank (min(T,N)-1 = %d).', maxDim);
end

% --- Real eigenvalues ---
eigReal = i_pca_eigs(X, max(dims));
eigReal = eigReal(dims);

% --- Null distribution ---
if verbose
    if ~isempty(label)
        fprintf(1, '\n  Parallel analysis [%s]: shuffle %d/%d', label, 0, nShuffle);
    else
        fprintf(1, '\n  Parallel analysis: shuffle %d/%d', 0, nShuffle);
    end
end
eigNull = zeros(nShuffle, numel(dims));
for ss = 1:nShuffle
    Xshuf = i_shuffle(X, mode);
    eigs_all = i_pca_eigs(Xshuf, max(dims));
    eigNull(ss, :) = eigs_all(dims);
    if verbose
        fprintf(1, '\b\b\b\b\b\b\b\b\b\b\b\b\b%d/%d', ss, nShuffle);
    end
end
if verbose
    fprintf(1, ' done.\n');
end

% --- Null quantile and dimension selection ---
nullQuantile = quantile(eigNull, 1 - alpha, 1);

% d* = largest k where real eigenvalue > null quantile
exceed = eigReal > nullQuantile;
if any(exceed)
    dStar = dims(find(exceed, 1, 'last'));
else
    dStar = 0;
end

% --- Pack results ---
results.eigReal      = eigReal;
results.eigNull      = eigNull;
results.nullQuantile = nullQuantile;
results.dimsTested   = dims;
results.dStar        = dStar;
results.alpha        = alpha;
results.nShuffle     = nShuffle;
results.rngSeed      = rngSeed;
results.mode         = mode;
end

% =========================================================================
%  Local helpers
% =========================================================================

function eigs = i_pca_eigs(X, maxK)
% Return eigenvalues (descending) for the first maxK components.
% Uses economy SVD of centred X for numerical stability; no toolbox needed.
[T, N] = size(X);
maxK   = min(maxK, min(T, N) - 1);
X      = X - mean(X, 1);               % centre columns
[~, S, ~] = svd(X, 'econ');            % singular values descending
sv     = diag(S);
eigs   = (sv(1:maxK).^2 / (T - 1))';  % convert to eigenvalues
end

function Xout = i_shuffle(X, mode)
% Shuffle X according to the specified mode.
[T, N] = size(X);
switch mode
    case 'neuronwise'
        Xout = zeros(T, N);
        for nn = 1:N
            Xout(:, nn) = X(randperm(T), nn);
        end
    case 'rowperm'
        Xout = X(randperm(T), :);
    otherwise
        error('diagnostics:dim_parallel_analysis:badMode', ...
            'mode must be ''neuronwise'' or ''rowperm''.');
end
end
