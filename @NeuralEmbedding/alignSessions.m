function results = alignSessions(objs, pars)
%ALIGNSESSIONS Align latent spaces across sessions using Procrustes.
%
%   RESULTS = ALIGNSESSIONS(OBJS) aligns the latent embeddings of all
%   sessions in the NeuralEmbedding array OBJS to the first session
%   (reference session = 1) using orthogonal Procrustes analysis.
%
%   RESULTS = ALIGNSESSIONS(OBJS, PARS) overrides default alignment
%   parameters via the struct PARS. See diagnostics.pars.ProcrustesAlignment.
%
%   Each session must have already had findEmbedding called. The latent
%   space used is the embedded data matrix E (first numPC components,
%   concatenated across trials).
%
%   Inputs
%   ------
%   objs : (1 x nSessions) or (nSessions x 1) NeuralEmbedding array.
%          At least 2 objects required.
%   pars : struct with optional fields (see diagnostics.pars.ProcrustesAlignment):
%          .allowScale  (default false) - allow isotropic scaling.
%          .refSession  (default 1)     - index of reference session.
%
%   Outputs
%   -------
%   results : struct with fields
%       .Z           - cell array {1 x nSessions} of original latent matrices.
%       .Z_aligned   - cell array {1 x nSessions} of aligned latent matrices.
%       .R           - cell array {1 x nSessions} of rotation matrices
%                      (identity for the reference session).
%       .scale       - (1 x nSessions) scale factors.
%       .disparity   - (1 x nSessions) Procrustes disparity per session.
%       .principalAngles - cell {1 x nSessions} principal-angle vectors.
%       .meanPrincipalAngle - (1 x nSessions) mean principal angle (rad).
%       .distCorr    - (1 x nSessions) Mantel distance correlation.
%       .refSession  - index of reference session used.
%       .sessions    - string array of session labels.
%       .animals     - string array of animal labels.
%
%   Notes
%   -----
%   * Alignment is performed on the concatenated (trial-stacked) embedded
%     data from the first (alphabetically ordered) matching conditions.
%   * Sessions must share the same embedding dimensionality (numPC).
%   * If sessions differ in the number of time bins, the shorter session
%     is padded / truncated to match the reference for metric computation
%     only; the full Z_aligned is always returned at native length.
%
%   Example
%   -------
%   NEs = [NE1, NE2, NE3];
%   NE1.findEmbedding('PCA');
%   NE2.findEmbedding('PCA');
%   NE3.findEmbedding('PCA');
%   res = alignSessions(NEs);
%   disp(res.disparity)   % per-session Procrustes disparity
%
%   See also diagnostics.compute.align_procrustes,
%            diagnostics.compute.alignment_metrics,
%            diagnostics.pars.ProcrustesAlignment,
%            NeuralEmbedding.findEmbedding

% --- Validation ---
if numel(objs) < 2
    error('NeuralEmbedding:alignSessions:tooFewSessions', ...
        'At least 2 NeuralEmbedding objects are required.');
end

defaultPars = diagnostics.pars.ProcrustesAlignment();
if nargin < 2 || isempty(pars)
    pars = defaultPars;
else
    pars = NeuralEmbedding.mergestructs(defaultPars, pars);
end

nSess   = numel(objs);
refSess = pars.refSession;

% --- Extract latent matrices ---
Z = cell(1, nSess);
for ss = 1:nSess
    Z{ss} = i_get_latent(objs(ss));
end

% --- Align to reference ---
Zref = Z{refSess};
d    = size(Zref, 2);

Z_aligned   = cell(1, nSess);
R_all       = cell(1, nSess);
scale_all   = zeros(1, nSess);
disparity   = zeros(1, nSess);
pAngles     = cell(1, nSess);
meanPAngle  = zeros(1, nSess);
distCorr    = zeros(1, nSess);

for ss = 1:nSess
    if ss == refSess
        Z_aligned{ss}  = Zref - mean(Zref, 1);
        R_all{ss}      = eye(d);
        scale_all(ss)  = 1.0;
        disparity(ss)  = 0;
        pAngles{ss}    = zeros(1, d);
        meanPAngle(ss) = 0;
        distCorr(ss)   = 1;
        continue;
    end

    % Truncate or pad to min common length for Procrustes fit
    T1 = size(Zref, 1);
    T2 = size(Z{ss}, 1);
    Tcommon = min(T1, T2);
    Zref_common = Zref(1:Tcommon, :);
    Zsess_common = Z{ss}(1:Tcommon, :);

    % Align common portion
    procResult = diagnostics.compute.align_procrustes(Zref_common, ...
        Zsess_common, pars.allowScale);

    R_all{ss}     = procResult.R;
    scale_all(ss) = procResult.scale;
    disparity(ss) = procResult.disparity;

    % Apply transform to full session
    Zc = Z{ss} - mean(Z{ss}, 1);
    Z_aligned{ss} = scale_all(ss) * Zc * R_all{ss};

    % Alignment metrics on common portion
    metResult      = diagnostics.compute.alignment_metrics(Zref_common, ...
        procResult.Z2_aligned);
    pAngles{ss}    = metResult.principalAngles;
    meanPAngle(ss) = metResult.meanPrincipalAngle;
    distCorr(ss)   = metResult.distCorr;
end

% --- Pack results ---
results.Z                  = Z;
results.Z_aligned          = Z_aligned;
results.R                  = R_all;
results.scale              = scale_all;
results.disparity          = disparity;
results.principalAngles    = pAngles;
results.meanPrincipalAngle = meanPAngle;
results.distCorr           = distCorr;
results.refSession         = refSess;
results.sessions           = string({objs.Session});
results.animals            = string({objs.Animal});
end

% =========================================================================
function Z = i_get_latent(obj)
% Concatenate embedded trials into T x d matrix (first area column).
E = obj.E;
E = E(:, 1);  % first area column
Z = cell2mat(cellfun(@(e) e', E, 'UniformOutput', false));
end
