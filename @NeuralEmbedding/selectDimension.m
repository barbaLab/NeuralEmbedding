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
%       .eigReal      - real eigenvalues for tested dims.
%       .eigNull      - (nShuffle x numel(dims)) null eigenvalue matrix.
%       .nullQuantile - (1-alpha) null quantile per dim.
%       .dimsTested   - dims vector used.
%       .dStar        - selected dimension.
%       .alpha, .nShuffle, .rngSeed, .mode - parameters used.
%       .animal, .session - metadata from the NeuralEmbedding object.
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

% --- Extract data (concatenate all trials along time axis) ---
X = i_get_data(obj);

% --- Run parallel analysis ---
results = diagnostics.compute.dim_parallel_analysis(X, dims, pars);

% --- Attach metadata ---
results.animal  = obj.Animal;
results.session = obj.Session;

% --- Store in M_ ---
obj.i_storeM(results, 'ParallelAnalysis');
end

% =========================================================================
%  Local helper
% =========================================================================
function X = i_get_data(obj)
% Concatenate smoothed, z-scored trials (from obj.S) into T x N matrix.
% Uses the first area column (respects current aMask setting).
S = obj.S;
% S is a cell (nTrials x nAreas); use first area column
S = S(:, 1);
% Each cell is nUnits x nTimeBins; transpose to nTimeBins x nUnits, then vertcat
X = cell2mat(cellfun(@(s) s', S, 'UniformOutput', false));
end
