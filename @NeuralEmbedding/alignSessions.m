function results = alignSessions(objs, pars)
%ALIGNSESSIONS Align latent spaces across sessions using Procrustes, and
%              store the aligned subspaces within each NeuralEmbedding object.
%
%   RESULTS = ALIGNSESSIONS(OBJS) aligns the latent embeddings of every
%   session in the NeuralEmbedding array OBJS to the first session
%   (reference session = 1) using orthogonal Procrustes analysis.
%   Alignment is performed *independently* for every area present in the
%   reference session (including "AllNeurons").
%
%   After this call:
%     - Each object's E_aligned_ and W_aligned_ private properties are
%       populated with the rotation-transformed embeddings.
%     - Setting OBJ.useAlignment = true makes get.E and get.W return the
%       aligned subspace instead of the original one.
%     - Each object's M_ is updated with an 'Alignment' entry (same logic
%       as computeMetrics; re-running replaces the previous result).
%
%   RESULTS = ALIGNSESSIONS(OBJS, PARS) overrides default alignment
%   parameters via the struct PARS. See diagnostics.pars.ProcrustesAlignment.
%
%   Inputs
%   ------
%   objs : (1 x nSessions) or (nSessions x 1) NeuralEmbedding array.
%          At least 2 objects required.  Each object must have had
%          findEmbedding called prior to alignment.
%   pars : struct with optional fields (see diagnostics.pars.ProcrustesAlignment):
%          .allowScale  (default false) - allow isotropic scaling.
%          .refSession  (default 1)     - index of reference session.
%
%   Outputs
%   -------
%   results : struct with fields
%       .Z              - {nAreas x nSessions} original latent matrices.
%       .Z_aligned      - {nAreas x nSessions} rotation-only aligned latents.
%       .R              - {nAreas x nSessions} rotation matrices.
%       .scale          - (nAreas x nSessions) isotropic scale factors.
%       .disparity      - (nAreas x nSessions) Procrustes disparity.
%       .principalAngles - {nAreas x nSessions} principal-angle vectors.
%       .meanPrincipalAngle - (nAreas x nSessions) mean principal angle (rad).
%       .distCorr       - (nAreas x nSessions) Mantel distance correlation.
%       .refSession     - index of reference session used.
%       .areas          - string array of area labels (rows of per-area fields).
%       .sessions       - string array of session labels.
%       .animals        - string array of animal labels.
%
%   Object side-effects
%   -------------------
%   Per object OBJ = OBJS(ss):
%     - OBJ.E_aligned_  is populated with the rotation-transformed embedding.
%     - OBJ.W_aligned_  is populated with the rotation-transformed loadings.
%     - OBJ.M_ receives an 'Alignment' entry via i_storeM.
%   To activate the aligned subspace set OBJ.useAlignment = true.
%   To deactivate set OBJ.useAlignment = false.
%
%   Notes
%   -----
%   * The rotation is applied without global re-centring so that the
%     per-trial mean structure of E is preserved.
%   * Only the first numPC components of E (and W) are rotated; higher
%     components (if stored) are left unchanged.
%   * If two sessions do not share the same area label, that area is
%     skipped for the non-matching session.
%
%   Example
%   -------
%   NEs = [NE1, NE2, NE3];
%   NE1.findEmbedding('PCA');  NE2.findEmbedding('PCA');
%   NE3.findEmbedding('PCA');
%   res = alignSessions(NEs);
%   disp(res.disparity)          % nAreas x nSessions matrix
%   NE2.useAlignment = true;     % activate aligned subspace for session 2
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

% Area list from reference session (includes "AllNeurons")
refAreas = objs(refSess).UArea;   % (nAreas+1 x 1) string
nAreas   = numel(refAreas);

% --- Initialise aligned storage (deep-copy of current E_ / W_) ---
for ss = 1:nSess
    objs(ss).E_aligned_ = objs(ss).E_;   % value-copy of cell array
    objs(ss).W_aligned_ = objs(ss).W_;
end

% --- Pre-allocate per-area per-session results ---
R_all        = cell(nAreas, nSess);
scale_all    = ones(nAreas, nSess);
disparity    = zeros(nAreas, nSess);
pAngles      = cell(nAreas, nSess);
meanPAngle   = zeros(nAreas, nSess);
distCorr     = ones(nAreas, nSess);
Z_all        = cell(nAreas, nSess);
Z_aligned_all = cell(nAreas, nSess);

% Fill identity values for reference session now
for aa = 1:nAreas
    R_all{aa, refSess}     = [];   % filled below once d is known
    scale_all(aa, refSess) = 1;
    disparity(aa, refSess) = 0;
    pAngles{aa, refSess}   = [];
    meanPAngle(aa, refSess) = 0;
    distCorr(aa, refSess)  = 1;
end

% --- Per-area alignment ---
for aa = 1:nAreas
    areaStr = refAreas(aa);

    % Locate area column in reference session E_
    aaRef = find(ismember(objs(refSess).UArea, areaStr));
    if isempty(aaRef), continue; end

    % Extract all non-empty trial cells for this area from reference
    E_ref_all = objs(refSess).E_(:, aaRef);   % nTrial_ref x 1
    valid_ref  = ~cellfun(@(e) isempty(e) || size(e,2)==0, E_ref_all);
    if ~any(valid_ref), continue; end

    E_ref_cells = E_ref_all(valid_ref);
    d_ref = min(objs(refSess).numPC, size(E_ref_cells{1}, 1));

    % Concatenate reference latent: T_ref x d
    Zref = cell2mat(cellfun(@(e) e(1:d_ref,:)', ...
        E_ref_cells, 'UniformOutput', false));   % T_ref x d_ref
    Z_all{aa, refSess}        = Zref;
    Z_aligned_all{aa, refSess} = Zref;
    R_all{aa, refSess}         = eye(d_ref);

    for ss = 1:nSess
        if ss == refSess, continue; end

        % Locate matching area in this session
        aaSess = find(ismember(objs(ss).UArea, areaStr));
        if isempty(aaSess)
            % Area not present in this session: no alignment possible
            R_all{aa, ss} = eye(d_ref);
            continue;
        end

        E_ss_all  = objs(ss).E_(:, aaSess);
        valid_ss   = ~cellfun(@(e) isempty(e) || size(e,2)==0, E_ss_all);
        if ~any(valid_ss)
            R_all{aa, ss} = eye(d_ref);
            continue;
        end

        E_ss_cells = E_ss_all(valid_ss);
        d_ss = min([objs(ss).numPC, size(E_ss_cells{1}, 1), d_ref]);

        % Concatenate session latent: T_ss x d_ss
        Z_ss = cell2mat(cellfun(@(e) e(1:d_ss,:)', ...
            E_ss_cells, 'UniformOutput', false));
        Z_all{aa, ss} = Z_ss;

        % Common length for Procrustes fit
        Tcommon = min(size(Zref, 1), size(Z_ss, 1));
        Zref_c  = Zref(1:Tcommon, 1:d_ss);
        Zss_c   = Z_ss(1:Tcommon, :);

        % Compute Procrustes alignment
        procResult = diagnostics.compute.align_procrustes(Zref_c, Zss_c, ...
            pars.allowScale);
        R  = procResult.R;    % d_ss x d_ss orthogonal
        sc = procResult.scale;

        R_all{aa, ss}     = R;
        scale_all(aa, ss) = sc;

        % ----- Apply rotation to ALL trials of this session (in-place) -----
        % E cell is (d_full x T); rotate first d_ss rows: E_rot = sc * R' * E(1:d_ss,:)
        % R is orthogonal, so R' = inv(R).
        % Working on the private E_aligned_ (already initialised as copy of E_).
        E_tmp = objs(ss).E_aligned_;
        for tr = 1:size(E_tmp, 1)
            Ec = E_tmp{tr, aaSess};
            if isempty(Ec) || size(Ec,2) == 0, continue; end
            nrows = min(d_ss, size(Ec, 1));
            % Slice R' to nrows x nrows to handle d_full < d_ss edge case
            Ec(1:nrows, :) = sc * (R(1:nrows, 1:nrows)' * Ec(1:nrows, :));
            E_tmp{tr, aaSess} = Ec;
        end
        objs(ss).E_aligned_ = E_tmp;

        % Apply rotation to W for this area
        Wss = objs(ss).W_aligned_{aaSess};
        if ~isempty(Wss)
            nrows_W = min(d_ss, size(Wss, 1));
            % Slice R' to nrows_W x nrows_W to handle dim mismatch
            Wss(1:nrows_W, :) = sc * (R(1:nrows_W, 1:nrows_W)' * Wss(1:nrows_W, :));
            objs(ss).W_aligned_{aaSess} = Wss;
        end

        % Aligned Z at full session length (rotation-only, no centring)
        Z_aligned_all{aa, ss} = cell2mat(cellfun(@(e) ...
            (sc * (R' * e(1:d_ss,:)))', E_ss_cells, 'UniformOutput', false));

        % Alignment metrics on the common (Procrustes-fitted) portion
        metResult         = diagnostics.compute.alignment_metrics(Zref_c, ...
            procResult.Z2_aligned);
        disparity(aa, ss)    = metResult.disparity;
        pAngles{aa, ss}      = metResult.principalAngles;
        meanPAngle(aa, ss)   = metResult.meanPrincipalAngle;
        distCorr(aa, ss)     = metResult.distCorr;
    end
end

% --- Pack top-level results ---
results.Z                  = Z_all;
results.Z_aligned          = Z_aligned_all;
results.R                  = R_all;
results.scale              = scale_all;
results.disparity          = disparity;
results.principalAngles    = pAngles;
results.meanPrincipalAngle = meanPAngle;
results.distCorr           = distCorr;
results.refSession         = refSess;
results.areas              = refAreas;
results.sessions           = string({objs.Session});
results.animals            = string({objs.Animal});

% --- Store per-session result in each object's M_ ---
for ss = 1:nSess
    sessData.R                  = R_all(:, ss);
    sessData.scale              = scale_all(:, ss);
    sessData.disparity          = disparity(:, ss);
    sessData.principalAngles    = pAngles(:, ss);
    sessData.meanPrincipalAngle = meanPAngle(:, ss);
    sessData.distCorr           = distCorr(:, ss);
    sessData.refSession         = refSess;
    sessData.areas              = refAreas;
    objs(ss).i_storeM(sessData, 'Alignment');
end
end
