function results = crossValDecode(obj, y, dim, pars)
%CROSSVALDECODE Cross-validated decoding + permutation test.
%
%   RESULTS = CROSSVALDECODE(OBJ, Y, DIM) decodes labels Y from the
%   DIM-dimensional PCA embedding of the preprocessed data using k-fold
%   cross-validation and a nearest-centroid classifier. A permutation test
%   provides an empirical p-value for the decoding accuracy.
%
%   RESULTS = CROSSVALDECODE(OBJ, Y, DIM, PARS) overrides default
%   parameters via the struct PARS. See diagnostics.pars.CVDecoding.
%
%   Multi-session usage
%   -------------------
%   RESULTS = CROSSVALDECODE(OBJS, Y, DIM, ...) where OBJS is a vector /
%   array of NeuralEmbedding objects and Y is either:
%     - a cell array {y1, y2, ...} (one vector per session), or
%     - a numeric/categorical vector of length sum(nTrials) (concatenated).
%   Returns a (1 x nSessions) struct array.
%
%   Inputs
%   ------
%   obj : scalar or array of NeuralEmbedding objects.
%   y   : T x 1 integer or categorical label vector (T = total time-bins
%         when trials are concatenated). For multi-session input, pass a
%         cell array with one vector per session.
%   dim : positive integer, number of latent dimensions.
%   pars: struct overriding diagnostics.pars.CVDecoding defaults.
%         Fields: kfold (5), nPerm (500), rngSeed (0),
%                 decoder ('nearestCentroid'),
%                 permMode ('global' | 'blocked' | 'timeshift').
%
%   Outputs
%   -------
%   results : struct (or struct array) with fields
%       .accMean, .accPerFold      - accuracy (mean & per fold).
%       .balAccMean, .balAccPerFold- balanced accuracy.
%       .confMat, .classes         - confusion matrix and class labels.
%       .pValue                    - permutation-test p-value.
%       .effectZ                   - effect-size z-score.
%       .statNull                  - (1 x nPerm) null distribution.
%       .dim, .kfold, .nPerm, .rngSeed, .permMode - parameters used.
%       .animal, .session          - metadata.
%
%   Example
%   -------
%   y   = NE.Conditions;  % trial-level labels, one per trial
%   res = NE.crossValDecode(y, 5);
%   fprintf('Acc=%.1f%%  p=%.4f  z=%.2f\n', ...
%       res.accMean*100, res.pValue, res.effectZ);
%
%   See also diagnostics.compute.cv_decoding,
%            diagnostics.compute.permutation_test,
%            diagnostics.pars.CVDecoding,
%            NeuralEmbedding.selectDimension

% --- Multi-session dispatch ---
if ~isscalar(obj)
    nSess = numel(obj);
    % Unpack per-session y if cell array
    if iscell(y)
        yCell = y;
    else
        % Split concatenated y by session (assuming nTimeBins per trial)
        % Build split based on time-bin count per session
        yCell = i_split_y(obj, y);
    end
    results = cell(1, nSess);
    for ss = 1:nSess
        if nargin < 4
            results{ss} = crossValDecode(obj(ss), yCell{ss}, dim);
        else
            results{ss} = crossValDecode(obj(ss), yCell{ss}, dim, pars);
        end
    end
    results = [results{:}];
    return;
end

% --- Defaults ---
defaultPars = diagnostics.pars.CVDecoding();
if nargin < 4 || isempty(pars)
    pars = defaultPars;
else
    pars = NeuralEmbedding.mergestructs(defaultPars, pars);
end

% --- Extract data ---
X = i_get_data(obj);

% Expand trial-level labels to time-bin level if needed
y = i_expand_labels(y, obj);

% --- Validate y ---
if numel(y) ~= size(X, 1)
    error('NeuralEmbedding:crossValDecode:labelMismatch', ...
        ['y has %d elements but X has %d time-bins. ' ...
         'Provide one label per time-bin (after trial concatenation), ' ...
         'or one label per trial (will be expanded automatically).'], ...
        numel(y), size(X, 1));
end

% --- CV decoding ---
decPars        = pars;
decodingResult = diagnostics.compute.cv_decoding(X, y, dim, decPars);

% --- Permutation test ---
% Build the permuter function based on permMode
switch pars.permMode
    case 'global'
        permuterFcn = @(Xp, yp) diagnostics.shufflers.global_permute(Xp, yp);
    case 'blocked'
        % Use trial index as block
        blockId     = i_trial_block_ids(y, obj);
        permuterFcn = @(Xp, yp) diagnostics.shufflers.blocked_permute(Xp, yp, blockId);
    case 'timeshift'
        permuterFcn = @(Xp, yp) diagnostics.shufflers.circular_shift(Xp, yp);
    otherwise
        error('NeuralEmbedding:crossValDecode:badPermMode', ...
            'permMode must be ''global'', ''blocked'', or ''timeshift''.');
end

statFcn = @(Xp, yp) diagnostics.compute.cv_decoding(Xp, yp, dim, decPars).accMean;

permResult = diagnostics.compute.permutation_test(statFcn, permuterFcn, ...
    X, y, pars.nPerm, pars.rngSeed);

% --- Merge results ---
results           = decodingResult;
results.pValue    = permResult.pValue;
results.effectZ   = permResult.effectZ;
results.statNull  = permResult.statNull;
results.nPerm     = pars.nPerm;
results.permMode  = pars.permMode;
results.animal    = obj.Animal;
results.session   = obj.Session;
end

% =========================================================================
%  Local helpers
% =========================================================================

function X = i_get_data(obj)
S = obj.S;
S = S(:, 1);  % first area column
X = cell2mat(cellfun(@(s) s', S, 'UniformOutput', false));
end

function y = i_expand_labels(y, obj)
% If y has one element per trial, replicate to per-time-bin.
S = obj.S;
S = S(:, 1);  % first area column
nBinsPerTrial = cellfun(@(s) size(s, 2), S);
T_total = sum(nBinsPerTrial);
nTr = numel(S);
y = y(:);
if numel(y) == nTr
    % Expand trial labels to time bins
    yExp = zeros(T_total, 1);
    offset = 0;
    for tt = 1:nTr
        yExp(offset + (1:nBinsPerTrial(tt))) = y(tt);
        offset = offset + nBinsPerTrial(tt);
    end
    y = yExp;
end
end

function blockId = i_trial_block_ids(~, obj)
% Return a T-length vector where each element is the trial index of that
% time bin — used as block IDs for blocked_permute.
S = obj.S;
S = S(:, 1);  % first area column
nBinsPerTrial = cellfun(@(s) size(s, 2), S);
blockId = zeros(sum(nBinsPerTrial), 1);
offset = 0;
nTr = numel(S);
for tt = 1:nTr
    blockId(offset + (1:nBinsPerTrial(tt))) = tt;
    offset = offset + nBinsPerTrial(tt);
end
end

function yCell = i_split_y(objs, y)
% Split a concatenated y vector into per-session cell array.
nSess = numel(objs);
yCell = cell(1, nSess);
offset = 0;
y = y(:);
for ss = 1:nSess
    S = objs(ss).S;
    S = S(:, 1);  % first area column
    T = sum(cellfun(@(s) size(s, 2), S));
    if numel(y) == sum(arrayfun(@(o) o.nTrial, objs))
        % Trial-level labels: take the right slice
        T = objs(ss).nTrial;
    end
    yCell{ss} = y(offset + (1:T));
    offset = offset + T;
end
end
