function results = selectDimension(obj, dims, pars)
%SELECTDIMENSION Shuffle-based dimension selection via parallel analysis.
%
%   RESULTS = SELECTDIMENSION(OBJ) runs parallel analysis on the smoothed,
%   preprocessed data stored in the NeuralEmbedding object and returns a
%   struct with the selected dimensionality (dStar) and diagnostic output.
%
%   RESULTS = SELECTDIMENSION(OBJ, DIMS) tests the specified component
%   indices (default 1:15).
%
%   RESULTS = SELECTDIMENSION(OBJ, DIMS, PARS) additionally overrides
%   default parameters with those in the struct PARS.  See
%   diagnostics.pars.ParallelAnalysis for all available fields.
%
%   Multi-session usage
%   -------------------
%   RESULTS = SELECTDIMENSION(OBJS, ...) where OBJS is a vector / array of
%   NeuralEmbedding objects returns a (1 x nSessions) struct array with
%   one entry per session.
%
%   Inputs
%   ------
%   obj  : scalar or array of NeuralEmbedding objects.
%   dims : positive integer vector of component indices to test
%          (default 1:15).
%   pars : struct overriding diagnostics.pars.ParallelAnalysis defaults.
%          Fields: nShuffle (200), alpha (0.05), rngSeed (0),
%                  mode ('neuronwise' | 'rowperm').
%
%   Outputs
%   -------
%   results : struct (or struct array for multiple sessions) with fields
%       .eigReal         - real eigenvalues for tested dims.
%       .eigNull         - (nShuffle x numel(dims)) null eigenvalue matrix.
%       .nullQuantile    - (1-alpha) null quantile per dim.
%       .dimsTested      - dims vector used.
%       .dStar           - selected dimension.
%       .alpha, .nShuffle, .rngSeed, .mode - parameters used.
%       .animal, .session        - metadata from the NeuralEmbedding object.
%       .embeddingMethod - embedding method used (from obj.currentEmbeddingMethod).
%
%   Projection pipeline
%   -------------------
%   The null eigenspectrum is built using the same projection pipeline as
%   obj.findEmbedding:
%     * For PCA / SmoothPCA  — covariance PCA (centre only, no extra
%       z-score).  obj.S already applies any z-scoring configured on the
%       object, so double-standardising is avoided.  This matches
%       embedding.PCA.reduce / MATLAB's pca() exactly.
%     * For GPFA / CCA / other — a warning is issued and PCA on the
%       smoothed data is used as an approximation.  For rigorous dimension
%       selection with those methods use crossValReconstruct.
%
%   Example
%   -------
%   % Single session
%   res = NE.selectDimension(1:20);
%   fprintf('%s.%s  dStar = %d\n', res.animal, res.session, res.dStar);
%
%   % Multiple sessions
%   results = selectDimension(NEobjs, 1:20);
%   [results.dStar]
%
%   See also diagnostics.compute.dim_parallel_analysis,
%            diagnostics.pars.ParallelAnalysis,
%            NeuralEmbedding.crossValReconstruct,
%            NeuralEmbedding.crossValDecode

% --- Multi-session dispatch ---
if ~isscalar(obj)
    nSess = numel(obj);
    results = cell(1, nSess);
    for ss = 1:nSess
        if nargin < 2
            results{ss} = selectDimension(obj(ss));
        elseif nargin < 3
            results{ss} = selectDimension(obj(ss), dims);
        else
            results{ss} = selectDimension(obj(ss), dims, pars);
        end
    end
    results = [results{:}];  % struct array
    return;
end

% --- Defaults ---
if nargin < 2 || isempty(dims)
    dims = 1:15;
end
defaultPars = diagnostics.pars.ParallelAnalysis();
if nargin < 3 || isempty(pars)
    pars = defaultPars;
else
    % Fill missing fields with defaults (user pars take priority)
    pars = NeuralEmbedding.mergestructs(defaultPars, pars);
end

% --- Build label for progress output ---
label = sprintf('%s.%s', obj.Animal, obj.Session);
if pars.verbose
    fprintf(1, '\nSelectDimension [%s]', label);
end

% --- Extract data (concatenate all trials along time axis) ---
% obj.S already applies z-scoring / preprocessing set on the object, so we
% do NOT z-score again inside dim_parallel_analysis (projFcn is used).
X = i_get_data(obj);

% --- Build projection function matching the object's embedding method ----
% This ensures the null distribution is built with exactly the same
% pipeline as findEmbedding, making the comparison statistically valid.
embMethod = char(obj.currentEmbeddingMethod);
projFcn   = i_build_proj_fcn(embMethod, label);

% --- Run parallel analysis ---
results = diagnostics.compute.dim_parallel_analysis(X, dims, pars, label, projFcn);

% --- Attach metadata ---
results.animal          = obj.Animal;
results.session         = obj.Session;
results.embeddingMethod = embMethod;

% --- Store in M_ ---
obj.i_storeM(results, 'ParallelAnalysis');
end

% =========================================================================
%  Local helpers
% =========================================================================
function X = i_get_data(obj)
% Concatenate preprocessed (smoothed + obj-z-scored) trials (from obj.S)
% into T x N matrix.  Uses the first area column (respects current aMask).
% NOTE: obj.S already applies z-scoring if obj.zscore=true, so dim_parallel_analysis
% should NOT z-score again — achieved by passing a projFcn.
S = obj.S;
% S is a cell (nTrials x nAreas); use first area column
S = S(:, 1);
% Each cell is nUnits x nTimeBins; transpose to nTimeBins x nUnits, then vertcat
X = cell2mat(cellfun(@(s) s', S, 'UniformOutput', false));
end

function projFcn = i_build_proj_fcn(embMethod, label)
% Build the eigenvalue-extraction function that matches the embedding method.
%
% For PCA / SmoothPCA: use covariance PCA (centre only, no z-score), which
% matches embedding.PCA.reduce / MATLAB's pca() behaviour.
%
% For other methods (GPFA, CCA, …): parallel analysis in the PCA sense is
% not directly applicable.  We warn the user and fall back to covariance
% PCA on the smoothed data as a useful approximation.  For rigorous
% dimensionality selection with non-PCA methods, use crossValReconstruct.
switch upper(strtrim(embMethod))
    case {'PCA', 'SMOOTHPCA', ''}
        % Covariance PCA — matches embedding.PCA.reduce (pca() centers only)
    otherwise
        warning('NeuralEmbedding:selectDimension:methodMismatch', ...
            ['Parallel analysis uses PCA eigenvalues, but the current ' ...
             'embedding method for [%s] is ''%s''. ' ...
             'Results are still meaningful as a baseline, but consider ' ...
             'crossValReconstruct for dimension selection with non-PCA ' ...
             'embeddings.'], label, embMethod);
end
% Covariance PCA (centre only) — matches embedding.PCA.reduce in all cases
projFcn = @(X, maxK) i_pca_eigs_cov(X, maxK);
end

function eigs = i_pca_eigs_cov(X, maxK)
% Covariance-PCA eigenvalues (centre only, no z-score).
% Equivalent to MATLAB's pca() and embedding.PCA.reduce, but toolbox-free.
% X must already be preprocessed (e.g. obj.S applies z-scoring if needed).
[T, N] = size(X);
maxK   = min(maxK, min(T, N) - 1);
X      = X - mean(X, 1);          % centre columns (pca() default)
[~, S, ~] = svd(X, 'econ');       % economy SVD, singular values descending
sv     = diag(S);
eigs   = (sv(1:maxK).^2 / (T - 1))';  % covariance eigenvalues
end
