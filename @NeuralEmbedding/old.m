


function [C,projectedData,Latent] = DimReduction(D,type,pars)
if isempty(D)
    C={};projectedData={};Latent={};
    return;
end
% default values
pars_.prekern = 250;

pars_.postkern = 0;

pars_.use_sqrt = false;
pars_.binWidth = 5;
pars_.numPC = 3;
pars.VarExp = .8;

pars_.splitUnits = [0 size(D(1).data,1)];
pars_.t = 1:size(D(1).data,2);
pars_.Reproject = false;

pars_.UseGpu = false;

pars_.FRateLim = 1;
pars_.acceptanceRatio = .05;
pars_.zscore = true;

pars_.seqTest = false(1,numel(D));

pars_.endLeg_range = [1:250 size(D(1).data,2)-250:size(D(1).data,2)];
pars_.interest_range = (-250:250)+0.5*size(D(1).data,2);
pars_.ccaRefSig = [];

pars_.SuperviseByConditions = false;

pars_.D_ref = {nan};

for f = fieldnames(pars)'
    pars_.(f{1}) = pars.(f{1});
end

N_Ref = length(pars_.D_ref);
N_Areas = length(pars_.splitUnits)-1;
if N_Ref ~= N_Areas 
    [pars_.D_ref{N_Ref+1:N_Areas}] = deal(nan);
end

% select and execute the routine
switch deblank(type)
    case {'SmoothPCA','PCA','pca'}
        [projectedData,C,Latent] = smoothAndPCA(D,pars_);
    case {'GPFA','gpfa'}
        if pars_.postkern == 0
            pars_.postkern = pars_.prekern;
        end
        [projectedData,C,Latent] = GPFA(D, pars_);
    case {'CCA','cca'}
        [projectedData,C,Latent] = smoothCCA(D,pars_);
    case {'justCCA','justcca'}
        [projectedData,C,Latent] = smoothjustCCA(D,pars_);
    case {'umap','UMAP'}
        [projectedData] = smoothUMAP(D,pars_);
        C = cell(size(projectedData));
        Latent = cell(size(projectedData));
    case {'t-SNE','tsne','tSNE','t-sne'}      
        projectedData = smoothTSNE(D,pars_);
        C = cell(size(projectedData));
        Latent = cell(size(projectedData));
    case {'I','identity','noproject','noProject'}
        projectedData = smoothIdentity(D,pars_);
        C = cell(size(projectedData));
        Latent = cell(size(projectedData));
    otherwise
        projectedData = nan;
        C = nan;
        Latent = nan;
end


end

%% Helper functions
function [newD,C,lat] = smoothIdentity(D, pars)

splitUnits = pars.splitUnits;
if isempty(splitUnits)
    splitUnits = [0 size(D(1).data,1)];
end
if isempty(D)
    expl = repmat({nan},[1,numel(splitUnits)]);
    return;
end

newD = cell(1,numel(splitUnits)-1);
C    = cell(1,numel(splitUnits)-1);
lat  = cell(1,numel(splitUnits)-1);

for uu = 1:numel(splitUnits)-1
    if isempty(splitUnits(uu)+1:splitUnits(uu+1))
        continue;
    end
    [D_,D__] = removeInactiveNeuronsAndCutData(D,pars);
    for tt = 1:numel(D)
        D_(tt).data = full(D_(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:));
        D__(tt).data = full(D__(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:));
    end

    newD{uu} = D_;
    if pars.Reproject
        newD{uu} = D__;
    end

    for tt = 1:numel(newD{uu})
        newD{uu}(tt).data  = smoother(newD{uu}(tt).data ,pars.postkern,pars.binWidth);
    end
end
end


function [newD,C,lat] = smoothAndPCA(D,pars)

splitUnits = pars.splitUnits;
if isempty(splitUnits)
    splitUnits = [0 size(D(1).data,1)];
end
if isempty(D)
    expl = repmat({nan},[1,numel(splitUnits)]);
    return;
end

newD = cell(1,numel(splitUnits)-1);
C    = cell(1,numel(splitUnits)-1);
lat  = cell(1,numel(splitUnits)-1);

for uu = 1:numel(splitUnits)-1
    if isempty(splitUnits(uu)+1:splitUnits(uu+1))
        continue;
    end
    [D_,D__] = removeInactiveNeuronsAndCutData(D,pars);

    for tt = 1:numel(D)
        D_(tt).data  = smoother(D_(tt).data (splitUnits(uu)+1:splitUnits(uu+1),:) ,pars.prekern,pars.binWidth);
        D__(tt).data = smoother(D__(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:) ,pars.prekern,pars.binWidth);
    end

    [newD{uu},C{uu},lat{uu}] = PCAreduce(D_,pars.numPC);
    if pars.Reproject
        newD{uu} = PCAproject(D__,C{uu});
    end

    for tt = 1:numel(newD{uu})
        newD{uu}(tt).data  = smoother(newD{uu}(tt).data ,pars.postkern,pars.binWidth);
    end

end


    function [newD ,C ,lat] = PCAreduce(D,dims)

        % PCAREDUCE Internal function for PCA
        %   PCAREDUCE(D,DIMS) returns a structure of the same form as D, except
        %   the data has been reduced with PCA. All conditions and trials are
        %   considered together to get the best joint reduction.

        % Agglomerate all of the conditions, and perform PCA
        alldata = [D.data];
        [u,sc,lat] = pca(alldata');

        % For each condition, store the reduced version of each data vector
        index = 0;
        for i=1:length(D)
            D(i).data = sc(index + (1:size(D(i).data,2)),1:dims)';
            index = index + size(D(i).data,2);
        end
        newD = D;
        C = u(:,1:dims);
        lat = cumsum(lat) ./ sum(lat);  % eigenvalues

    end

    function [newD] = PCAproject(D,C)

        % Agglomerate all of the conditions, and perform PCA
        alldata = [D.data];
        sc = (alldata - mean(alldata,2))'*C;

        % For each condition, store the reduced version of each data vector
        index = 0;
        for i=1:length(D)
            D(i).data = sc(index + (1:size(D(i).data,2)),:)';
            index = index + size(D(i).data,2);
        end
        newD = D;
    end

end

function [newD] = smoothTSNE(D,pars)
splitUnits = pars.splitUnits;
if isempty(splitUnits)
    splitUnits = [0 size(D(1).data,1)];
end
if isempty(D)
    newD = D;
    return;
end


newD = cell(1,numel(splitUnits)-1);

for uu = 1:numel(splitUnits)-1
if isempty(splitUnits(uu)+1:splitUnits(uu+1))
    continue;
end
    [D__,D_] = removeInactiveNeuronsAndCutData(D,pars);

    for tt = 1:numel(D)
        D_(tt).data  = smoother(D_(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:) ,pars.prekern,pars.binWidth);
        D__(tt).data = smoother(D__(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:),pars.prekern,pars.binWidth);
    end

    if pars.UseGpu
        data = gpuArray([D_.data]');
    else
        data = [D_.data]';
    end

    newD{uu} = tsne(data,'NumDimensions',pars.numPC);


    for tt = 1:numel(newD{uu})
        newD{uu}(tt).data  = smoother(newD{uu}(tt).data ,pars.postkern,pars.binWidth);
    end

end

end

function [newD] = smoothUMAP(D,pars)

splitUnits = pars.splitUnits;
if isempty(splitUnits)
    splitUnits = [0 size(D(1).data,1)];
end
if isempty(D)
    newD = D;
    return;
end

newD = cell(1,numel(splitUnits)-1);
for uu = 1:numel(splitUnits)-1
if isempty(splitUnits(uu)+1:splitUnits(uu+1))
    continue;
end
    [Dth,DthFull] = removeInactiveNeuronsAndCutData(D,pars);

    for tt = 1:numel(D)
        Dth(tt).data  = smoother(Dth(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:) ,pars.prekern,pars.binWidth);
        DthFull(tt).data = smoother(DthFull(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:),pars.prekern,pars.binWidth);
    end

    if pars.SuperviseByConditions
       [nNeurons,nT] = size(Dth(1).data);
        conds = {Dth.condition};
        uConds = unique(conds);
        nCond = numel(uConds);
        condIdxMat = cellfun(@(s) strcmpi(conds,s),uConds,'UniformOutput',false);
        condIdxMat = cat(1,condIdxMat{:});
        condIdxMat = sum(condIdxMat.*[1:nCond]',1);
        condAr = arrayfun(@(m)ones(1,nT)*m,condIdxMat,'UniformOutput',false);
        condAr = cat(2,condAr{:});
        dat = [Dth.data;condAr]';
        lblCol = nNeurons+1;
    else
        dat = [Dth.data]';
        lblCol = 0;
    end

    newD{uu} = UMAPreduce(Dth,dat,pars.numPC,lblCol);
    if pars.Reproject
        dat = [DthFull.data]';
        newD{uu} = UMAPproject(DthFull,dat,...
            fullfile(getenv("Temp"),'umapTemplate.mat'));
    end

    newD{uu} = flip_sign(newD{uu},pars.D_ref{uu});

    for tt = 1:numel(newD{uu})
        newD{uu}(tt).data  = smoother(newD{uu}(tt).data ,pars.postkern,pars.binWidth);
    end

end


    function newD = UMAPreduce(D,dat,dims,lblCol)
        parameter_names = ...
            arrayfun(@(n)sprintf('unit_%.3d',n),1:size(dat,2),...
            'UniformOutput',false);

        [reduction,umap] = run_umap(dat,...
            'n_components',dims,...
            'method','mex',...
            'NSMethod','nn_descent',...
            'fast_approximation',true,...
            'label_column',(lblCol),...
            'cluster_output','none',...
            'verbose','text',...
            'parameter_names',parameter_names);

        tmpDir = getenv("Temp");
        save(fullfile(tmpDir,'umapTemplate'),'umap','-v7.3');

        index = 0;
        for i=1:length(D)
            D(i).data = reduction(index + (1:size(D(i).data,2)),1:dims)';
            index = index + size(D(i).data,2);
        end
        newD = D;
    end

    function newD = UMAPproject(D,dat,template_file)
        parameter_names = ...
            arrayfun(@(n)sprintf('unit_%.3d',n),1:size(dat,2),...
            'UniformOutput',false);

        [reduction] = run_umap(dat,...
            'method','mex',...
            'fast_approximation',true,...
            'cluster_output','none',...
            'verbose','text',...
            'template_file',template_file,...
            'parameter_names',parameter_names);

        index = 0;
        for i=1:length(D)
            D(i).data = reduction(index + (1:size(D(i).data,2)),:)';
            index = index + size(D(i).data,2);
        end
        newD = D;
    end

end

function [newD,C,lat] = GPFA(D, pars)

seqTest = pars.seqTest;
splitUnits = pars.splitUnits;
if isempty(splitUnits)
    splitUnits = [0 size(D(1).data,1)];
end
if isempty(D)
    expl = repmat({nan},[1,numel(splitUnits)]);
    return;
end

newD = cell(1,numel(splitUnits)-1); 
C    = cell(1,numel(splitUnits)-1);
lat  = cell(1,numel(splitUnits)-1);

for uu = 1:numel(splitUnits)-1
if isempty(splitUnits(uu)+1:splitUnits(uu+1))
    continue;
end
    [D_,D__] = removeInactiveNeuronsAndCutData(D,pars);
    for tt = 1:numel(D)
        D_(tt).data = full(D_(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:));
        D__(tt).data = full(D__(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:));
    end

    [newD{uu},C{uu},lat{uu}] = GPFAreduce(D_(~seqTest),D_(seqTest),pars,'verbose',false);
    if pars.Reproject
        newD{uu} = GPFAproject(D__,C{uu});
    end

    for tt = 1:numel(newD{uu})
        newD{uu}(tt).data  = smoother(newD{uu}(tt).data ,pars.postkern,pars.binWidth);
    end
end


    function [newD,C,lat,alldata] = GPFAreduce(D, seqTest, pars,varargin)
        %
        % gpfaEngine(seqTrain, seqTest, fname, ...)
        %
        % Extract neural trajectories using GPFA.
        %
        % INPUTS:
        %
        % seqTrain      - training data structure, whose nth entry (corresponding to
        %                 the nth experimental trial) has fields
        %                   trialId (1 x 1)   -- unique trial identifier
        %                   y (# neurons x T) -- neural data
        %                   T (1 x 1)         -- number of timesteps
        % seqTest       - test data structure (same format as seqTrain)
        %
        % OPTIONAL ARGUMENTS:
        %
        % xDim          - state dimensionality (default: 3)
        % binWidth      - spike bin width in msec (default: 20)
        % startTau      - GP timescale initialization in msec (default: 100)
        % startEps      - GP noise variance initialization (default: 1e-3)
        %
        % @ 2009 Byron Yu         byronyu@stanford.edu
        %        John Cunningham  jcunnin@stanford.edu


        % Agglomerate all of the conditions, and perform PCA
        alldata = repmat(struct(),size(D));
        [alldata.y] = deal(D.data);
        [alldata.trialId] = deal(D.condition);
        for nn = 1:numel(alldata),alldata(nn).T = size(D(nn).data,2);end

        % For each condition, store the reduced version of each data vector

        XDim          = pars.numPC;
        binWidth      = pars.binWidth; % in msec
        startTau      = 100; % in msec
        startEps      = 1e-3;
        extraOpts     = assignopts(who, varargin);

        % For compute efficiency, train on equal-length segments of trials
        seqTrainCut = cutTrials(alldata, extraOpts{:});
        if isempty(seqTrainCut)
            fprintf('WARNING: no segments extracted for training.  Defaulting to segLength=Inf.\n');
            seqTrainCut = cutTrials(alldata, 'segLength', Inf);
        end

        % ==================================
        % Initialize state model parameters
        % ==================================
        startParams.covType = 'rbf';
        % GP timescale
        % Assume binWidth is the time step size.
        startParams.gamma = (binWidth / startTau)^2 * ones(1, XDim);
        % GP noise variance
        startParams.eps   = startEps * ones(1, XDim);

        % ========================================
        % Initialize observation model parameters
        % ========================================
        fprintf('Initializing parameters using factor analysis...\n');

        yAll             = [seqTrainCut.y];
        [faParams, faLL] = fastfa(yAll, XDim, extraOpts{:});

        startParams.d = mean(yAll, 2);
        startParams.C = faParams.L;
        startParams.R = diag(faParams.Ph);

        % Define parameter constraints
        startParams.notes.learnKernelParams = true;
        startParams.notes.learnGPNoise      = false;
        startParams.notes.RforceDiagonal    = true;

        currentParams = startParams;

        % =====================
        % Fit model parameters
        % =====================
        fprintf('\nFitting GPFA model...\n');

        [estParams, seqTrainCut, LL, iterTime] =...
            em(currentParams, seqTrainCut, extraOpts{:});

        % Extract neural trajectories for original, unsegmented trials
        % using learned parameters
        [gpfa_traj, LLorig] = exactInferenceWithLL(alldata, estParams);

        % ========================================
        % Leave-neuron-out prediction on test data
        % ========================================
        if ~isempty(seqTest) % check if there are any test trials
            if estParams.notes.RforceDiagonal
                seqTest = cosmoother_gpfa_viaOrth_fast(seqTest, estParams, 1:XDim);
            else
                seqTest = cosmoother_gpfa_viaOrth(seqTest, estParams, 1:XDim);
            end
        end

        newD = D;
        [newD.data] = deal(gpfa_traj.xsm);
        [~, C] = orthogonalize([gpfa_traj.xsm], estParams.C);
        [~,lat] = pcacov(estParams.C * estParams.C');
        lat = cumsum(lat(1:XDim))./sum(lat);
    end

    function [newD] = GPFAproject(D, C)
        % Agglomerate all of the conditions, and perform PCA
        alldata = [D.data];

        % remove nans
        %         sc = (alldata - mean(alldata,2))'*C;
        sc = alldata'*C;

        % For each condition, store the reduced version of each data vector
        index = 0;
        for i = 1:length(D)
            D(i).data = sc(index + (1:size(D(i).data,2)),:)';
            index = index + size(D(i).data,2);
        end
        newD = D;
    end

    function [Xorth, Lorth, TT] = orthogonalize(X, L)
        %
        % [Xorth, Lorth, TT] = orthogonalize(X, L)
        %
        % Orthonormalize the columns of the loading matrix and
        % apply the corresponding linear transform to the latent variables.
        %
        %   yDim: data dimensionality
        %   xDim: latent dimensionality
        %
        % INPUTS:
        %
        % X        - latent variables (xDim x T)
        % L        - loading matrix (yDim x xDim)
        %
        % OUTPUTS:
        %
        % Xorth    - orthonormalized latent variables (xDim x T)
        % Lorth    - orthonormalized loading matrix (yDim x xDim)
        % TT       - linear transform applied to latent variables (xDim x xDim)
        %
        % @ 2009 Byron Yu -- byronyu@stanford.edu

        xDim = size(L, 2);

        if xDim == 1
            mag   = sqrt(L' * L);
            Lorth = L / mag;
            TT = mag;
            Xorth = mag * X;

        else
            [UU, DD, VV] = svd(L);
            % TT is transform matrix
            TT = diag(diag(DD)) * VV';

            Lorth = UU(:, 1:xDim);
            Xorth = TT * X;
        end
    end

    function [estParams, seq, LL, iterTime] = em(currentParams, seq, varargin)
        %
        % [estParams, seq, LL] = em(currentParams, seq, ...)
        %
        % Fits GPFA model parameters using expectation-maximization (EM) algorithm.
        %
        %   yDim: number of neurons
        %   xDim: state dimensionality
        %
        % INPUTS:
        %
        % currentParams - GPFA model parameters at which EM algorithm is initialized
        %                   covType (string) -- type of GP covariance ('rbf')
        %                   gamma (1 x xDim) -- GP timescales in milliseconds are
        %                                       'stepSize ./ sqrt(gamma)'
        %                   eps (1 x xDim)   -- GP noise variances
        %                   d (yDim x 1)     -- observation mean
        %                   C (yDim x xDim)  -- mapping between low- and high-d spaces
        %                   R (yDim x yDim)  -- observation noise covariance
        % seq           - training data structure, whose nth entry (corresponding to
        %                 the nth experimental trial) has fields
        %                   trialId      -- unique trial identifier
        %                   T (1 x 1)    -- number of timesteps
        %                   y (yDim x T) -- neural data
        %
        % OUTPUTS:
        %
        % estParams     - learned GPFA model parameters returned by EM algorithm
        %                   (same format as currentParams)
        % seq           - training data structure with new fields
        %                   xsm (xDim x T)        -- posterior mean at each timepoint
        %                   Vsm (xDim x xDim x T) -- posterior covariance at each timepoint
        %                   VsmGP (T x T x xDim)  -- posterior covariance of each GP
        % LL            - data log likelihood after each EM iteration
        % iterTime      - computation time for each EM iteration
        %
        % OPTIONAL ARGUMENTS:
        %
        % emMaxIters    - number of EM iterations to run (default: 500)
        % tol           - stopping criterion for EM (default: 1e-8)
        % minVarFrac    - fraction of overall data variance for each observed dimension
        %                 to set as the private variance floor.  This is used to combat
        %                 Heywood cases, where ML parameter learning returns one or more
        %                 zero private variances. (default: 0.01)
        %                 (See Martin & McDonald, Psychometrika, Dec 1975.)
        % freqLL        - data likelihood is computed every freqLL EM iterations.
        %                 freqLL = 1 means that data likelihood is computed every
        %                 iteration. (default: 5)
        % verbose       - logical that specifies whether to display status messages
        %                 (default: false)
        %
        % @ 2009 Byron Yu         byronyu@stanford.edu
        %        John Cunningham  jcunnin@stanford.edu

        emMaxIters   = 500;
        tol          = 1e-8;
        minVarFrac   = 0.01;
        verbose      = false;
        freqLL       = 10;
        extra_opts   = assignopts(who, varargin);

        N            = length(seq(:));
        T            = [seq.T];
        [yDim, xDim] = size(currentParams.C);
        LL           = [];
        LLi          = 0;
        iterTime     = [];
        varFloor     = minVarFrac * diag(cov([seq.y]'));

        % Loop once for each iteration of EM algorithm
        for i = 1:emMaxIters
            if verbose
                fprintf('\n');
            end
            rand('state', i);
            randn('state', i);
            tic;

            if verbose
                fprintf('EM iteration %3d of %d', i, emMaxIters);
            end
            if (rem(i, freqLL) == 0) || (i<=2)
                getLL = true;
            else
                getLL = false;
            end

            % ==== E STEP =====
            if ~isnan(LLi)
                LLold = LLi;
            end
            [seq, LLi] = exactInferenceWithLL(seq, currentParams, 'getLL', getLL);
            LL         = [LL LLi];

            % ==== M STEP ====
            sum_Pauto   = zeros(xDim, xDim);
            for n = 1:N
                sum_Pauto = sum_Pauto + ...
                    sum(seq(n).Vsm, 3) + seq(n).xsm * seq(n).xsm';
            end
            Y           = [seq.y];
            Xsm         = [seq.xsm];
            sum_yxtrans = Y * Xsm';
            sum_xall    = sum(Xsm, 2);
            sum_yall    = sum(Y, 2);

            term = [sum_Pauto sum_xall; sum_xall' sum(T)]; % (xDim+1) x (xDim+1)
            Cd   = ([sum_yxtrans sum_yall]) / term;   % yDim x (xDim+1)

            currentParams.C = Cd(:, 1:xDim);
            currentParams.d = Cd(:, end);

            % yCent must be based on the new d
            % yCent = bsxfun(@minus, [seq.y], currentParams.d);
            % R = (yCent * yCent' - (yCent * [seq.xsm]') * currentParams.C') / sum(T);
            if currentParams.notes.RforceDiagonal
                sum_yytrans = sum(Y .* Y, 2);
                yd          = sum_yall .* currentParams.d;
                term        = sum((sum_yxtrans - currentParams.d * sum_xall') .*...
                    currentParams.C, 2);
                r           = currentParams.d.^2 + (sum_yytrans - 2*yd - term) / sum(T);

                % Set minimum private variance
                r               = max(varFloor, r);
                currentParams.R = diag(r);
            else
                sum_yytrans = Y * Y';
                yd          = sum_yall * currentParams.d';
                term        = (sum_yxtrans - currentParams.d * sum_xall') * currentParams.C';
                R           = currentParams.d * currentParams.d' +...
                    (sum_yytrans - yd - yd' - term) / sum(T);

                currentParams.R = (R + R') / 2; % ensure symmetry
            end

            if currentParams.notes.learnKernelParams
                res = learnGPparams(seq, currentParams, 'verbose', verbose,...
                    extra_opts{:});
                switch currentParams.covType
                    case 'rbf'
                        currentParams.gamma = res.gamma;
                    case 'tri'
                        currentParams.a     = res.a;
                    case 'logexp'
                        currentParams.a     = res.a;
                end
                if currentParams.notes.learnGPNoise
                    currentParams.eps = res.eps;
                end
            end

            tEnd     = toc;
            iterTime = [iterTime tEnd];

            % Display the most recent likelihood that was evaluated
            if verbose
                if getLL
                    fprintf('       lik %g (%.1f sec)\n', LLi, tEnd);
                else
                    fprintf('\n');
                end
            else
%                 if getLL
%                     fprintf('       lik %g\r', LLi);
%                 else
%                     fprintf('\r');
%                 end
            end

            % Verify that likelihood is growing monotonically
            if i<=2
                LLbase = LLi;
            elseif (LLi < LLold)
                fprintf('\nError: Data likelihood has decreased from %g to %g\n',...
                    LLold, LLi);
                keyboard;
            elseif ((LLi-LLbase) < (1+tol)*(LLold-LLbase))
                break;
            end
        end
        fprintf('\n');

        if length(LL) < emMaxIters
            fprintf('Fitting has converged after %d EM iterations.\n', length(LL));
        end

        if any(diag(currentParams.R) == varFloor)
            fprintf('Warning: Private variance floor used for one or more observed dimensions in GPFA.\n');
        end

        estParams = currentParams;
    end

    function [seq, LL] = exactInferenceWithLL(seq, params, varargin)
        %
        % [seq, LL] = exactInferenceWithLL(seq, params,...)
        %
        % Extracts latent trajectories given GPFA model parameters.
        %
        % INPUTS:
        %
        % seq         - data structure, whose nth entry (corresponding to the nth
        %               experimental trial) has fields
        %                 y (yDim x T) -- neural data
        %                 T (1 x 1)    -- number of timesteps
        % params      - GPFA model parameters
        %
        % OUTPUTS:
        %
        % seq         - data structure with new fields
        %                 xsm (xDim x T)        -- posterior mean at each timepoint
        %                 Vsm (xDim x xDim x T) -- posterior covariance at each timepoint
        %                 VsmGP (T x T x xDim)  -- posterior covariance of each GP
        % LL          - data log likelihood
        %
        % OPTIONAL ARGUMENTS:
        %
        % getLL       - logical that specifies whether to compute data log likelihood
        %               (default: false)
        % wbar        - waitbar handle
        %
        % @ 2009 Byron Yu         byronyu@stanford.edu
        %        John Cunningham  jcunnin@stanford.edu

        getLL = false;
        dimreduce_wb = -1;  % waitbar handle for performing dim reduction
        assignopts(who, varargin);

        [yDim, xDim] = size(params.C);

        % Precomputations
        if params.notes.RforceDiagonal
            Rinv     = diag(1./diag(params.R));
            logdet_R = sum(log(diag(params.R)));
        else
            Rinv     = inv(params.R);
            Rinv     = (Rinv+Rinv') / 2; % ensure symmetry
            logdet_R = logdet(params.R);
        end
        CRinv  = params.C' * Rinv;
        CRinvC = CRinv * params.C;

        Tall = [seq.T];
        Tu   = unique(Tall);
        LL   = 0;

        % Overview:
        % - Outer loop on each elt of Tu.
        % - For each elt of Tu, find all trials with that length.
        % - Do inference and LL computation for all those trials together.
        for j = 1:length(Tu)

            if (ishandle(dimreduce_wb)) % user is peforming dim reduction, so show progress
                waitbar(0.5+j/(2*length(Tu)), dimreduce_wb, 'Extracting trajectories...');
            end

            T = Tu(j);

            [K_big, K_big_inv, logdet_K_big] = make_K_big(params, T);

            % There are three sparse matrices here: K_big, K_big_inv, and CRinvC_inv.
            % Choosing which one(s) to make sparse is tricky.  If all are sparse,
            % code slows down significantly.  Empirically, code runs fastest if
            % only K_big is made sparse.
            %
            % There are two problems with calling both K_big_inv and CRCinvC_big
            % sparse:
            % 1) their sum is represented by Matlab as a sparse matrix and taking
            %    its inverse is more costly than taking the inverse of the
            %    corresponding full matrix.
            % 2) term2 has very few zero entries, but Matlab will represent it as a
            %    sparse matrix.  This makes later computations with term2 ineffficient.

            K_big = sparse(K_big);

            blah        = cell(1, T);
            [blah{:}]   = deal(CRinvC);
            %CRinvC_big = blkdiag(blah{:});     % (xDim*T) x (xDim*T)
            [invM, logdet_M] = invPerSymm(K_big_inv + blkdiag(blah{:}), xDim,...
                'offDiagSparse', true);

            % Note that posterior covariance does not depend on observations,
            % so can compute once for all trials with same T.
            % xDim x xDim posterior covariance for each timepoint
            Vsm = nan(xDim, xDim, T);
            idx = 1: xDim : (xDim*T + 1);

            for t = 1:T
                cIdx       = idx(t):idx(t+1)-1;
                Vsm(:,:,t) = invM(cIdx, cIdx);
            end



            % T x T posterior covariance for each GP
            VsmGP = nan(T, T, xDim);
            idx   = 0 : xDim : (xDim*(T-1));

            for i = 1:xDim
                VsmGP(:,:,i) = invM(idx+i,idx+i);
            end

            % Process all trials with length T
            nList    = find(Tall == T);
            dif      = bsxfun(@minus, [seq(nList).y], params.d); % yDim x sum(T)
            term1Mat = reshape(CRinv * dif, xDim*T, []); % (xDim*T) x length(nList)

            % Compute blkProd = CRinvC_big * invM efficiently
            % blkProd is block persymmetric, so just compute top half
            Thalf   = ceil(T/2);
            blkProd = zeros(xDim*Thalf, xDim*T);
            idx     = 1: xDim : (xDim*Thalf + 1);


            for t = 1:Thalf
                bIdx            = idx(t):idx(t+1)-1;
                blkProd(bIdx,:) = CRinvC * invM(bIdx,:);
            end


            blkProd = K_big(1:(xDim*Thalf), :) *...
                fillPerSymm(speye(xDim*Thalf, xDim*T) - blkProd, xDim, T);
            % potentially place a waitbar kill here
            xsmMat  = fillPerSymm(blkProd, xDim, T) * term1Mat; % (xDim*T) x length(nList)

            % potentially place a waitbar kill here

            ctr = 1;
            for n = nList
                seq(n).xsm   = reshape(xsmMat(:,ctr), xDim, T);
                seq(n).Vsm   = Vsm;
                seq(n).VsmGP = VsmGP;

                ctr = ctr + 1;
            end

            if getLL
                % Compute data likelihood
                val = -T * logdet_R - logdet_K_big - logdet_M -...
                    yDim * T * log(2*pi);
                LL  = LL + length(nList) * val - sum(sum((Rinv * dif) .* dif)) +...
                    sum(sum((term1Mat' * invM) .* term1Mat'));
            end

        end

        if getLL
            LL = LL / 2;
        else
            LL = NaN;
        end
    end

end

function [newD,C,r] = smoothCCA(D,pars)
    splitUnits = pars.splitUnits;
    if isempty(splitUnits)
        error('CCA requires anatomical areas information in the Splitunits field of the parameters structure');
    end
    if isempty(D)
        expl = repmat({nan},[1,numel(splitUnits)]);
        return;
    end
    
    nArea = numel(splitUnits)-1;

    [D_,D__] = removeInactiveNeuronsAndCutData(D,pars);
    D_ = repmat({D_},1,nArea);
    D__ = repmat({D__},1,nArea);

    for uu = 1:numel(splitUnits)-1
        for tt = 1:numel(D)
            D_{uu}(tt).data  = smoother(D_{uu}(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:) ,pars.prekern,pars.binWidth);
            D__{uu}(tt).data = smoother(D__{uu}(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:),pars.prekern,pars.binWidth);
        end
    end

    [newD,C,r] = CCAreduce(D_,pars);
    if pars.Reproject
        newD = CCAproject(D__,C);
    end

    function [newD,C,Corr] = CCAreduce(D,pars)
        % PCAREDUCE Internal function for PCA
        %   PCAREDUCE(D,DIMS) returns a structure of the same form as D, except
        %   the data has been reduced with PCA. All conditions and trials are
        %   considered together to get the best joint reduction.

        % Agglomerate all of the conditions, and perform PCA
        if length(D) ~= 2 && ~iscell(D)
            error('Input must be a cell-array of two elements with data from two distinct areas.')
        end
        dims = pars.numPC;
        endLeg_range = pars.endLeg_range;
        interest_range = pars.interest_range;


        newD = D;
        nAreas = numel(D);
        alldata = cell(size(D));

        for ii=1:nAreas
            alldata{ii} = [D{ii}.data]';
        end
        [A,B,Corr] = getCCAWeights(alldata{:});
        C = {A(:,1:dims),B(:,1:dims)};
        [U,V] = checkFlip(alldata{:},C{:},endLeg_range, interest_range);
        sc = {U,V};

        % For each condition, store the reduced version of each data vector
        for ii = 1:nAreas
            index = 0;
            for jj = 1:length(D{ii})
                newD{ii}(jj).data = sc{ii}(index + (1:size(D{ii}(jj).data,2)),1:dims)';
                index = index + size(D{ii}(jj).data,2);
            end
        end

        function [A,B,r] = getCCAWeights(X,Y)
            %{
DESCRIPTION: CCA calculation to get canonical weights of input data

INPUT: ______________________________________________________
X: first dataset
Y: second dataset

OUTPUT: _____________________________________________________
A & B: corresponding sets of canonical weight matrices for X and Y
respectively
r: vector containing canonical correlations for each canonical variate
(this calculation is performed independently of the signs of the CCA
weights which is why we can calculate it now).
            %}

            if nargin < 2
                error(message('stats:canoncorr:TooFewInputs'));
            end

            [n,p1] = size(X);
            if size(Y,1) ~= n
                error(message('stats:canoncorr:InputSizeMismatch'));
            elseif n == 1
                error(message('stats:canoncorr:NotEnoughData'));
            end
            p2 = size(Y,2);

            % Center the variables
            X = X - mean(X,1);
            Y = Y - mean(Y,1);

            % Factor the inputs, and find a full rank set of columns if necessary
            [Q1,T11,perm1] = qr(X,0);
            rankX = sum(abs(diag(T11)) > eps(abs(T11(1)))*max(n,p1));
            if rankX == 0
                error(message('stats:canoncorr:BadData', 'X'));
            elseif rankX < p1
                warning(message('stats:canoncorr:NotFullRank', 'X'));
                Q1 = Q1(:,1:rankX); T11 = T11(1:rankX,1:rankX);
            end
            [Q2,T22,perm2] = qr(Y,0);
            rankY = sum(abs(diag(T22)) > eps(abs(T22(1)))*max(n,p2));
            if rankY == 0
                error(message('stats:canoncorr:BadData', 'Y'));
            elseif rankY < p2
                warning(message('stats:canoncorr:NotFullRank', 'Y'));
                Q2 = Q2(:,1:rankY); T22 = T22(1:rankY,1:rankY);
            end

            % Compute canonical coefficients and canonical correlations.  For rankX >
            % rankY, the economy-size version ignores the extra columns in L and rows
            % in D. For rankX < rankY, need to ignore extra columns in M and D
            % explicitly. Normalize A and B to give U and V unit variance.
            d = min(rankX,rankY);
            [L,thisD,M] = svd(Q1' * Q2,0);
            A = T11 \ L(:,1:d) * sqrt(n-1);
            B = T22 \ M(:,1:d) * sqrt(n-1);
            r = min(max(diag(thisD(:,1:d))', 0), 1); % remove roundoff errs

            % Put coefficients back to their full size and their correct order
            A(perm1,:) = [A; zeros(p1-rankX,d)];
            B(perm2,:) = [B; zeros(p2-rankY,d)];

        end

        function [U,V] = checkFlip(X,Y,A,B,endLeg_range, interest_range)

            %{
DESCRIPTION: standardized means to determine if weights need to be flipped 
(i.e., if corresponding traces need to be flipped). This will be done by
taking the average firing rate of the resulting top CV (or CCA trace) at 
it ends and seeing if the average firing rate around the region of interest
is lower than it. If it is, then flip the weights.

INPUT: ______________________________________________________
RFA_unitData
S1_unitData
RFA_CCAweights
S1_CCAweights

endleg_range: 1x4 vector detailing 2 pairs of time stamps that define
'baseline' time range

interest_range: 1x4 vector detailing 2 pairs of time stamps that define
time range of interest (e.g., around the time of task onset)

OUTPUT: _____________________________________________________
U & V: corresponding sets of canonical vectors if flip_Status = 0.
            %}

            flipRFA = 0;
            flipS1 = 0;


            % Center the variables
            X = X - mean(X,1);
            Y = Y - mean(Y,1);

            % project onto canonical space
            U = X * A;
            V = Y * B;

%             U = reshape(U, binNum, [], size(U,2));
%             V = reshape(V, binNum, [], size(V,2));

            %{
subplot(1,2,1)
plot(U(:,:,1),'b')
hold on
plot(mean(U(:,:,1),2),'r', LineWidth=5)
xline(50,'r', LineWidth=5)
subplot(1,2,2)
plot(V(:,:,1),'b')
hold on
plot(mean(V(:,:,1),2),'r', LineWidth=5)
xline(50,'r', LineWidth=5)
            %}


            %Flip canonical weights and rerun CCA
            if mean(mean(mean(U([endLeg_range(1):endLeg_range(2),endLeg_range(3):endLeg_range(4)],:,1),2),1)) > mean(mean(mean(U(interest_range(1):interest_range(2),:,1),2),1))
                flipRFA = 1;
                A = -A;
            end
            if mean(mean(mean(V([endLeg_range(1):endLeg_range(2),endLeg_range(3):endLeg_range(4)],:,1),2),1)) > mean(mean(mean(V(interest_range(1):interest_range(2),:,1),2),1))
                flipS1 = 1;
                B = -B;
            end

            if flipRFA || flipS1
                U = X * A;
                V = Y * B;
                
%                 U = reshape(U, binNum, [], size(U,2));
%                 V = reshape(V, binNum, [], size(V,2));
            end
        end

    end

    function [newD] = CCAproject(D,C)
        if length(D) ~= 2 && ~iscell(D)
            error('Input must be a cell-array of two elements with data from two distinct areas.')
        end

        newD = D;
        nAreas = numel(D);
        sc = cell(size(D));

         % Agglomerate all of the conditions, and perform PCA
        for ii = 1:nAreas
            alldata = [D{ii}.data]';
            sc{ii} = (alldata - mean(alldata,2))*C{ii};
        end %ii

        % For each condition, store the reduced version of each data vector
        for ii = 1:nAreas
            index = 0;
            for jj = 1:length(D{ii})
                newD{ii}(jj).data = sc{ii}(index + (1:size(D{ii}(jj).data,2)),:)';
                index = index + size(D{ii}(jj).data,2);
            end %jj
        end %ii
    end %function

end

function [newD,C,r] = smoothjustCCA(D,pars)
    splitUnits = pars.splitUnits;
    if isempty(splitUnits)
        error('CCA requires anatomical areas information in the Splitunits field of the parameters structure');
    end
    if isempty(D)
        expl = repmat({nan},[1,numel(splitUnits)]);
        return;
    end
    
    nArea = numel(splitUnits)-1;

    [D_,D__] = removeInactiveNeuronsAndCutData(D,pars);
    D_ = repmat({D_},1,nArea);
    D__ = repmat({D__},1,nArea);

    for uu = 1:numel(splitUnits)-1
        for tt = 1:numel(D)
            D_{uu}(tt).data  = smoother(D_{uu}(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:) ,pars.prekern,pars.binWidth);
            D__{uu}(tt).data = smoother(D__{uu}(tt).data(splitUnits(uu)+1:splitUnits(uu+1),:),pars.prekern,pars.binWidth);
        end
    end

    [newD,C,r] = CCAreduce(D_,pars);
    
    if pars.Reproject
        newD = CCAproject(D__,C);
    end

    function [newD,C,Corr] = CCAreduce(D,pars)
        % PCAREDUCE Internal function for PCA
        %   PCAREDUCE(D,DIMS) returns a structure of the same form as D, except
        %   the data has been reduced with PCA. All conditions and trials are
        %   considered together to get the best joint reduction.

        % Agglomerate all of the conditions, and perform PCA
        if length(D) ~= 2 && ~iscell(D)
            error('Input must be a cell-array of two elements with data from two distinct areas.')
        end
        dims = pars.numPC;
        endLeg_range = pars.endLeg_range;
        interest_range = pars.interest_range;


        newD = D;
        nAreas = numel(D);
        alldata = cell(size(D));

        for ii=1:nAreas
            alldata{ii} = [D{ii}.data]';
        end
        [A,B,Corr,U,V]=canoncorr(alldata{:});
        C = {A(:,1:dims),B(:,1:dims)};
        sc = {U,V};

        % For each condition, store the reduced version of each data vector
        for ii = 1:nAreas
            index = 0;
            for jj = 1:length(D{ii})
                newD{ii}(jj).data = sc{ii}(index + (1:size(D{ii}(jj).data,2)),1:dims)';
                index = index + size(D{ii}(jj).data,2);
            end
        end
        [newD,wasFlipped] = adjustPolarization(newD,pars);
    end

    function [newD] = CCAproject(D,C)
        if length(D) ~= 2 && ~iscell(D)
            error('Input must be a cell-array of two elements with data from two distinct areas.')
        end

        newD = D;
        nAreas = numel(D);
        sc = cell(size(D));

         % Agglomerate all of the conditions, and perform PCA
        for ii = 1:nAreas
            alldata = [D{ii}.data]';
            sc{ii} = (alldata - mean(alldata,2))*C{ii};
        end %ii

        % For each condition, store the reduced version of each data vector
        for ii = 1:nAreas
            index = 0;
            for jj = 1:length(D{ii})
                newD{ii}(jj).data = sc{ii}(index + (1:size(D{ii}(jj).data,2)),:)';
                index = index + size(D{ii}(jj).data,2);
            end %jj
        end %ii
    end %function

    function [ccaData,wasFlipped] = adjustPolarization(ccaData,pars)
        % check correlation sign and use it to flip or not the signal
        wasFlipped = cell(size(ccaData));
        if isempty(pars.ccaRefSig)
            [wasFlipped{1:numel(ccaData)}] = deal(false);
            return;
        end

        for ii=1:numel(ccaData)
            thisData = mean(cat(3,ccaData{ii}.data),3);
            refData  = mean(cat(3,pars.ccaRefSig{ii}.data),3);
            c = diag(corr(thisData',refData'));
            tmpData = arrayfun(@(d)d.data.*sign(c),ccaData{1},'UniformOutput',false);
            [ccaData{ii}.data] = tmpData{:};
            wasFlipped{ii} = c<0;
        end

    end
end

function seq = getSeq(dat, binWidth, varargin)
%
% seq = getSeq(dat, binWidth, ...)  
%
% Converts 0/1 spike trains into spike counts.
%
% INPUTS:
%
% dat         - structure whose nth entry (corresponding to the nth experimental
%               trial) has fields
%                 trialId -- unique trial identifier
%                 spikes  -- 0/1 matrix of the raw spiking activity across
%                            all neurons.  Each row corresponds to a neuron.
%                            Each column corresponds to a 1 msec timestep.
% binWidth    - spike bin width in msec
%
% OUTPUTS:
%
% seq         - data structure, whose nth entry (corresponding to
%               the nth experimental trial) has fields
%                 trialId      -- unique trial identifier
%                 T (1 x 1)    -- number of timesteps
%                 y (yDim x T) -- neural data
%
% OPTIONAL ARGUMENTS:
%
% useSqrt     - logical specifying whether or not to use square-root transform
%               on spike counts (default: true)
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  useSqrt = true;
  assignopts(who, varargin);

  seq = [];
  for n = 1:length(dat)
    yDim = size(dat(n).spikes, 1);
    T    = floor(size(dat(n).spikes, 2) / binWidth);

    seq(n).trialId = dat(n).trialId;
    seq(n).T       = T;
    seq(n).y       = nan(yDim, T);
    
    for t = 1:T
      iStart = binWidth * (t-1) + 1;
      iEnd   = binWidth * t;
      
      seq(n).y(:,t) = sum(dat(n).spikes(:, iStart:iEnd), 2);
    end
    
    if useSqrt
      seq(n).y = sqrt(seq(n).y);
    end
  end
  
  % Remove trials that are shorter than one bin width
  if ~isempty(seq)
    trialsToKeep = ([seq.T] > 0);
    seq = seq(trialsToKeep);
  end
end

function [Dth,DthFull] = removeInactiveNeuronsAndCutData(D,pars)

Dth = (D);
DthFull = (D);
nTrial = numel(D);

thSpikeRate = arrayfun(@(d) (sum(d.data,2)./max(d.times)) < pars.FRateLim,D,...
    'UniformOutput',false);
thSpikeRate = sum([thSpikeRate{:}],2) > (nTrial * pars.acceptanceRatio);

nanidx = false(size(D(1).data,1),1);
for nn = 1:numel(Dth)
    Dth(nn).data = Dth(nn).data(:,pars.t);
    Dth(nn).data(thSpikeRate,:) = nan;
    Dth(nn).times = Dth(nn).times(pars.t);
    Dth(nn).trialId = nn;

    DthFull(nn).data(thSpikeRate,:) = nan;
    DthFull(nn).trialId = nn;
    nanidx = nanidx | any(isnan(Dth(nn).data),2);
end

[Dth.spikes] = deal(Dth.data);
tmp = getSeq(Dth, pars.binWidth, 'useSqrt', pars.use_sqrt);
[Dth.data] = deal(tmp.y);

[DthFull.spikes] = deal(DthFull.data);
tmp = getSeq(DthFull, pars.binWidth, 'useSqrt', pars.use_sqrt);
[DthFull.data] = deal(tmp.y);


% remove nans
% subs = setdiff(1:numel(nanidx),find(nanidx));
% subs = subs(randperm(numel(subs),sum(nanidx)));
for nn = 1:numel(Dth)
    %          alldata(nn).y(nanidx,:) = alldata(nn).y(subs,:);
    Dth(nn).data(nanidx,:) = zeros(sum(nanidx),size(Dth(nn).data,2));
    l = size(Dth(nn).data,2);
    n = ceil(max(Dth(nn).times) * 2 * pars.FRateLim);
    randomSpk = arrayfun(@(x)sparse(1,randperm(l,n),1,1,l),1:sum(nanidx),'UniformOutput',false);
    randomSpk = cat(1,randomSpk{:});
    Dth(nn).data(nanidx,:) = randomSpk;

    DthFull(nn).data(nanidx,:) = zeros(sum(nanidx),size(DthFull(nn).data,2));
    l = size(DthFull(nn).data,2);
    n = ceil(max(DthFull(nn).times) * 2 * pars.FRateLim);
    randomSpk = arrayfun(@(x)sparse(1,randperm(l,n),1,1,l),1:sum(nanidx),'UniformOutput',false);
    randomSpk = cat(1,randomSpk{:});
    DthFull(nn).data(nanidx,:) = randomSpk;
    
    % if pars.zscore
    %     mu = mean(Dth(nn).data,2);
    %     ss = std(Dth(nn).data,[],2);
    %     Dth(nn).data = (Dth(nn).data - mu)./ss;
    % 
    %     DthFull(nn).data = (DthFull(nn).data - mu)./ss;
    % end
end

if pars.zscore
    mu = mean([Dth.data],2);
    ss = std([Dth.data],[],2);
    tmp = arrayfun(@(x)(x.data-mu)./ss,Dth,'UniformOutput',false);
    tmpfull = arrayfun(@(x)(x.data-mu)./ss,DthFull,'UniformOutput',false);
    [Dth.data] = deal(tmp{:});
    [DthFull.data] = deal(tmpfull{:});
end

Dth = rmfield(Dth, {'spikes','trialId'});
DthFull = rmfield(DthFull, {'spikes','trialId'});


end

function D_correct = flip_sign(D,D_ref)
    D_correct = D;
    if isempty(D_ref) || (isnumeric(D_ref) && isnan(D_ref) )
        return;
    end
    sig = mean(cat(3,D.data),3);
    ref = mean(cat(3,D_ref.data),3);
    flip = diag(sign(corr(sig',ref')));
    if any(flip<0)
        for ii = 1:numel(D)
            D_correct(ii).data = flip .* D(ii).data;
        end
    end
end

function yOut = smoother(yIn, kernSD, stepSize, varargin)
%
% yOut = smoother(yIn, kernSD, stepSize)
%
% Gaussian kernel smoothing of data across time.
%
% INPUTS:
%
% yIn      - input data (yDim x T)
% kernSD   - standard deviation of Gaussian kernel, in msec
% stepSize - time between 2 consecutive datapoints in yIn, in msec
%
% OUTPUT:
%
% yOut     - smoothed version of yIn (yDim x T)
%
% OPTIONAL ARGUMENT:
%
% causal   - logical indicating whether temporal smoothing should
%            include only past data (true) or all data (false)
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

% Aug 21, 2011: Added option for causal smoothing

  causal = false;
  assignopts(who, varargin);

  if (kernSD == 0) || (size(yIn, 2)==1)
    yOut = yIn;
    return
  end

  if issparse(yIn)
      yIn = full(yIn);
  end

  % Filter half length
  % Go 3 standard deviations out
  fltHL = ceil(3 * kernSD / stepSize);

  % Length of flt is 2*fltHL + 1
  flt = normpdf(-fltHL*stepSize : stepSize : fltHL*stepSize, 0, kernSD);

  if causal
    flt(1:fltHL) = 0;
  end

  [yDim, T] = size(yIn);
  yOut      = nan(yDim, T);

  % Normalize by sum of filter taps actually used
  nm = conv(flt, ones(1, T));
  
  for i = 1:yDim
    ys = conv(flt, yIn(i,:)) ./ nm;
    % Cut off edges so that result of convolution is same length 
    % as original data
    yOut(i,:) = ys(fltHL+1:end-fltHL);
  end
end