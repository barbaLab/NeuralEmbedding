function results = crossValAlignment(obj, pars)
%CROSSVALALIGNMENT Within-session latent-space stability via random trial splits.
%
%   RESULTS = CROSSVALALIGNMENT(OBJ) estimates how stable the latent
%   manifold geometry is across randomly chosen subsets of trials within
%   the same recording session. This is a purely *within-session* test:
%   it does not require a second session, and its result can be used as a
%   baseline for the cross-session Procrustes disparity produced by
%   ALIGNSESSIONS.
%
%   Algorithm
%   ---------
%   For each of PARS.nSplit random replicates:
%     1. Randomly assign trials to two equal-size groups (half-sessions).
%     2. (Optionally) refit a PARS.dim-dimensional PCA on each half
%        independently, projecting each half's time-bins into latent space.
%     3. Align the two latent matrices with orthogonal Procrustes
%        (see diagnostics.compute.align_procrustes).
%     4. Compute disparity, principal angles, and distance correlation
%        (see diagnostics.compute.alignment_metrics).
%
%   A low median disparity (relative to the cross-session disparity from
%   alignSessions) indicates that the geometry is stable within the session.
%   The result is stored in M_ under type 'IntraAlignment'.
%
%   RESULTS = CROSSVALALIGNMENT(OBJS) where OBJS is a vector of
%   NeuralEmbedding objects returns a (1 x nSessions) struct array.
%
%   RESULTS = CROSSVALALIGNMENT(OBJ, PARS) overrides default parameters.
%   See diagnostics.pars.IntraAlignment.
%
%   Inputs
%   ------
%   obj  : scalar or array of NeuralEmbedding objects.
%          findEmbedding must have been called (OBJ.E must be non-empty)
%          unless PARS.dim > 0 (in which case PCA is refit on the raw
%          smoothed data OBJ.S).
%   pars : struct with fields (see diagnostics.pars.IntraAlignment):
%          .nSplit     (default 100)
%          .dim        (default 0 = use current embedding directly)
%          .allowScale (default false)
%          .rngSeed    (default 0)
%          .verbose    (default true)
%
%   Outputs
%   -------
%   results : struct (or struct array for multiple sessions) with fields
%       .disparity          - (1 x nSplit) per-split Procrustes disparity.
%       .disparityMean      - mean disparity.
%       .disparityMedian    - median disparity.
%       .disparityStd       - std of disparity.
%       .principalAngles    - {1 x nSplit} principal-angle vectors.
%       .meanPrincipalAngle - (1 x nSplit) mean principal angle (rad).
%       .distCorr           - (1 x nSplit) Mantel distance correlation.
%       .nSplit, .dim, .allowScale, .rngSeed - parameters used.
%       .animal, .session   - metadata.
%
%   Example
%   -------
%   NE.findEmbedding('PCA');
%   res = NE.crossValAlignment();
%   fprintf('Intra-session disparity = %.4f ± %.4f\n', ...
%       res.disparityMean, res.disparityStd);
%
%   % Compare with cross-session alignment
%   res_cross = alignSessions([NE1, NE2]);
%   fprintf('Cross-session disparity = %.4f\n', res_cross.disparity(1,2));
%
%   See also diagnostics.compute.intra_alignment,
%            diagnostics.pars.IntraAlignment,
%            NeuralEmbedding.alignSessions

% --- Multi-session dispatch ---
if ~isscalar(obj)
    nSess = numel(obj);
    results = cell(1, nSess);
    for ss = 1:nSess
        if nargin < 2
            results{ss} = crossValAlignment(obj(ss));
        else
            results{ss} = crossValAlignment(obj(ss), pars);
        end
    end
    results = [results{:}];
    return;
end

% --- Defaults ---
defaultPars = diagnostics.pars.IntraAlignment();
if nargin < 2 || isempty(pars)
    pars = defaultPars;
else
    pars = NeuralEmbedding.mergestructs(defaultPars, pars);
end

% --- Build progress label ---
label = sprintf('%s.%s', obj.Animal, obj.Session);

% --- Build per-trial cell array of latent data ---
if pars.dim > 0
    % Refit PCA from scratch on each half — pass smoothed data as cells
    Z = i_get_S_cells(obj);    % cell {nTrials x 1}, each (nUnits x nBins)
    % Transpose to (nBins x nUnits) so intra_alignment can split trials
    Z = cellfun(@(s) s', Z, 'UniformOutput', false);
    % intra_alignment will do PCA refit internally
else
    % Use current embedding (E) directly — cell {nTrials x 1}, each (d x nBins)
    Z = i_get_E_cells(obj);
end

% --- Run intra-alignment stability analysis ---
results = diagnostics.compute.intra_alignment(Z, pars, label);

% --- Attach metadata ---
results.animal  = obj.Animal;
results.session = obj.Session;

% --- Store in M_ ---
obj.i_storeM(results, 'IntraAlignment');
end

% =========================================================================
%  Local helpers
% =========================================================================

function Z = i_get_E_cells(obj)
% Return per-trial embedding cells (first area column), filtered to
% non-empty trials only.  Each cell is (d x nBins).
E = obj.E;
E = E(:, 1);  % first area column
Z = E(~cellfun(@(e) isempty(e) || size(e,2)==0, E));
end

function S = i_get_S_cells(obj)
% Return per-trial smoothed data cells (first area column).
% Each cell is (nUnits x nBins).
Sc = obj.S;
Sc = Sc(:, 1);  % first area column
S  = Sc(~cellfun(@(s) isempty(s) || size(s,2)==0, Sc));
end
