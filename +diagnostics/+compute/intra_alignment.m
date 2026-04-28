function results = intra_alignment(Z, pars, label)
%INTRA_ALIGNMENT Within-session latent-space stability via random trial splits.
%
%   RESULTS = diagnostics.compute.intra_alignment(Z, PARS) estimates the
%   stability of the latent geometry within a single session by repeatedly
%   splitting trials into two random halves, fitting a PCA embedding on
%   each half independently, and computing the Procrustes disparity
%   between the two sub-embeddings.
%
%   A low median disparity indicates that the manifold geometry is stable
%   across subsets of trials (i.e. it is reproducible within the session).
%
%   Inputs
%   ------
%   Z     : T x d latent matrix (trials already projected; rows = time bins,
%           columns = latent dims).  Alternatively, pass a cell array
%           {nTrials x 1} where each cell is (d x nBins) — the same format
%           as OBJ.E.
%
%         When a cell array is provided, the trials are split at the *trial*
%         level (each random split assigns whole trials to each half), and
%         PCA is refit from scratch on each half's concatenated data.
%
%   pars  : struct with fields (see diagnostics.pars.IntraAlignment):
%           .nSplit    (default 100) — number of random half-splits.
%           .dim       (default []) — PCA dimensionality to use when
%                       refitting. If empty, uses size(Z,2) (no refit).
%           .allowScale (default false) — isotropic Procrustes scale.
%           .rngSeed   (default 0)
%           .verbose   (default true) — print progress.
%   label : (optional) progress label string, e.g. 'Animal.Session'.
%
%   Outputs
%   -------
%   results : struct with fields
%       .disparity     - (1 x nSplit) Procrustes disparity per split.
%       .disparityMean - mean disparity across splits.
%       .disparityMedian - median disparity across splits.
%       .disparityStd  - std of disparity across splits.
%       .principalAngles - {1 x nSplit} principal angles per split.
%       .meanPrincipalAngle - (1 x nSplit) mean principal angle per split.
%       .distCorr      - (1 x nSplit) Mantel distance correlation per split.
%       .nSplit, .dim, .allowScale, .rngSeed - parameters used.
%
%   Notes
%   -----
%   * When Z is a T x d matrix (pre-projected), the rows are split evenly
%     (rows 1:T/2 vs T/2+1:T after shuffling) — effectively a random
%     time-block split.  Use the cell-array form for a proper trial-level
%     split with PCA refit.
%   * PCA refit is performed via economy SVD (no toolbox).
%   * For large T, distance-correlation metrics become expensive; they
%     are skipped (set to NaN) when each half has > 1000 time-bins.
%
%   Example
%   -------
%   pars = diagnostics.pars.IntraAlignment();
%   res  = diagnostics.compute.intra_alignment(E_cell, pars, 'Rat1.S01');
%   fprintf('Median intra-session disparity = %.4f\n', res.disparityMedian);
%
%   See also diagnostics.compute.align_procrustes,
%            diagnostics.compute.alignment_metrics,
%            diagnostics.pars.IntraAlignment

if nargin < 2 || isempty(pars)
    pars = diagnostics.pars.IntraAlignment();
end
if nargin < 3 || isempty(label)
    label = '';
end

nSplit    = pars.nSplit;
dim       = pars.dim;
allowScale = pars.allowScale;
rngSeed   = pars.rngSeed;
verbose   = isfield(pars,'verbose') && pars.verbose;

if ~isempty(rngSeed)
    rng(rngSeed, 'twister');
end

% --- Normalise input to a cell array of (d x nBins) matrices --------
isCellInput = iscell(Z);
if isCellInput
    % Validate: each cell must be a 2-D numeric matrix
    nTrials = numel(Z);
    if nTrials < 4
        error('diagnostics:intra_alignment:tooFewTrials', ...
            'Need at least 4 trials for a meaningful random split (got %d).', nTrials);
    end
else
    % Z is T x d matrix — treat each time-bin as a "trial"
    if ~ismatrix(Z) || ~isnumeric(Z)
        error('diagnostics:intra_alignment:badInput', ...
            'Z must be a T x d matrix or a cell array of (d x nBins) matrices.');
    end
    % Wrap each row as a 1-element cell (d x 1 column vectors)
    nTrials = size(Z, 1);
    if nTrials < 4
        error('diagnostics:intra_alignment:tooFewTrials', ...
            'Need at least 4 rows for a meaningful random split (got %d).', nTrials);
    end
    Z = mat2cell(Z', size(Z,2), ones(1, nTrials));   % cell of (d x 1)
    Z = Z(:);
end

% --- Pre-allocate outputs ---
disparity_vec  = zeros(1, nSplit);
pAngles_cell   = cell(1, nSplit);
meanPAngle_vec = zeros(1, nSplit);
distCorr_vec   = zeros(1, nSplit);

if verbose
    if ~isempty(label)
        fprintf(1, '\n  IntraAlignment [%s]: split 0/%d', label, nSplit);
    else
        fprintf(1, '\n  IntraAlignment: split 0/%d', nSplit);
    end
end

halfN = floor(nTrials / 2);
prevLen = 0;
for sp = 1:nSplit
    % Random trial assignment
    perm   = randperm(nTrials);
    idxA   = perm(1:halfN);
    idxB   = perm(halfN+1 : 2*halfN);   % equal-size halves

    % Concatenate trials in each half → (d x T_half) then transpose
    ZA = cell2mat(Z(idxA)');   % d x T_A  (each cell is d x nBins)
    ZB = cell2mat(Z(idxB)');

    ZA = ZA';   % T_A x d
    ZB = ZB';

    % Optional PCA refit on each half
    if ~isempty(dim) && dim > 0 && dim < size(ZA,2)
        ZA = i_pca_project(ZA, dim);
        ZB = i_pca_project(ZB, dim);
    end

    % Equalise lengths for Procrustes
    T_common = min(size(ZA,1), size(ZB,1));
    ZA_c = ZA(1:T_common, :);
    ZB_c = ZB(1:T_common, :);

    % Align B to A
    procRes = diagnostics.compute.align_procrustes(ZA_c, ZB_c, allowScale);
    metRes  = diagnostics.compute.alignment_metrics(ZA_c, procRes.Z2_aligned);

    disparity_vec(sp)  = procRes.disparity;
    pAngles_cell{sp}   = metRes.principalAngles;
    meanPAngle_vec(sp) = metRes.meanPrincipalAngle;
    distCorr_vec(sp)   = metRes.distCorr;

    if verbose
        msg = sprintf('%d/%d', sp, nSplit);
        fprintf(1, '%s%s', repmat(char(8), 1, prevLen), msg);
        prevLen = numel(msg);
    end
end

if verbose
    fprintf(1, ' done.\n');
end

results.disparity          = disparity_vec;
results.disparityMean      = mean(disparity_vec);
results.disparityMedian    = median(disparity_vec);
results.disparityStd       = std(disparity_vec);
results.principalAngles    = pAngles_cell;
results.meanPrincipalAngle = meanPAngle_vec;
results.distCorr           = distCorr_vec;
results.nSplit             = nSplit;
results.dim                = dim;
results.allowScale         = allowScale;
results.rngSeed            = rngSeed;
end

% =========================================================================
%  Local helper
% =========================================================================
function Zproj = i_pca_project(Z, dim)
% Fit economy PCA on Z and project to first DIM dimensions.
Zc  = Z - mean(Z, 1);
[~, ~, V] = svd(Zc, 'econ');
W    = V(:, 1:dim);
Zproj = Zc * W;
end
