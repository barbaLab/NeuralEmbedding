function [E,C,VarExp] = reduce(seqTrain, seqTest, varargin)
%  
% originally gpfaEngine(seqTrain, seqTest, fname, ...) 
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
% fname         - filename of where results are saved
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

  xDim          = 3;
  binWidth      = 20; % in msec
  startTau      = 100; % in msec
  startEps      = 1e-3;
  extraOpts     = assignopts(who, varargin);
    
  % For compute efficiency, train on equal-length segments of trials
  seqTrainCut = cutTrials(seqTrain, extraOpts{:});
  if isempty(seqTrainCut)
    fprintf('WARNING: no segments extracted for training.  Defaulting to segLength=Inf.\n');
    seqTrainCut = cutTrials(seqTrain, 'segLength', Inf);
  end
  
  % ==================================
  % Initialize state model parameters
  % ==================================
  startParams.covType = 'rbf';
  % GP timescale
  % Assume binWidth is the time step size.
  startParams.gamma = (binWidth / startTau)^2 * ones(1, xDim);
  % GP noise variance
  startParams.eps   = startEps * ones(1, xDim);

  % ========================================
  % Initialize observation model parameters
  % ========================================
  fprintf('Initializing parameters using factor analysis...\n');
  
  yAll             = [seqTrainCut.y];
  [faParams, faLL] = fastfa(yAll, xDim, extraOpts{:});
  
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
  [seqTrain, LLorig] = exactInferenceWithLL(seqTrain, estParams);

  % ========================================
  % Leave-neuron-out prediction on test data
  % ========================================
  if ~isempty(seqTest) % check if there are any test trials
    if estParams.notes.RforceDiagonal
      seqTest = cosmoother_gpfa_viaOrth_fast(seqTest, estParams, 1:xDim);
    else
      seqTest = cosmoother_gpfa_viaOrth(seqTest, estParams, 1:xDim);
    end
  end

  % =============
  % return results
  % =============
  E = cell(size(seqTrain));
  C = cell(1);
  VarExp = cell(1);

  [E{:}] = deal(seqTrain.xsm);
  [~, C{1}] = orthogonalize([seqTrain.xsm], estParams.C);
  [~,lat] = pcacov(estParams.C * estParams.C');
  VarExp{1} = cumsum(lat(1:xDim))./sum(lat);
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

function seqOut = cutTrials(seqIn, varargin)
%
% seqOut = cutTrials(seqIn, ...)
%
% Extracts trial segments that are all of the same length.  Uses
% overlapping segments if trial length is not integer multiple
% of segment length.  Ignores trials with length shorter than
% one segment length.
%
% INPUTS:
%
% seqIn       - data structure, whose nth entry (corresponding to
%               the nth experimental trial) has fields
%                 trialId      -- unique trial identifier
%                 T (1 x 1)    -- number of timesteps in trial
%                 y (yDim x T) -- neural data
%
% OUTPUTS:
%
% seqOut      - data structure, whose nth entry (corresponding to
%               the nth segment) has fields
%                 trialId      -- identifier of trial from which
%                                 segment was taken
%                 segId        -- segment identifier within trial
%                 T (1 x 1)    -- number of timesteps in segment
%                 y (yDim x T) -- neural data
%
% OPTIONAL ARGUMENTS:
%
% segLength   - length of segments to extract, in number of timesteps.
%               If infinite, entire trials are extracted, i.e., no
%               segmenting. (default: 20)
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

segLength = 20; % number of timesteps in each segment
assignopts(who, varargin);

if isinf(segLength)
    seqOut = seqIn;
    return
end

seqOut = [];
for n = 1:length(seqIn)
    T = seqIn(n).T;

    % Skip trials that are shorter than segLength
    if T < segLength
        fprintf('Warning: trialId %4d shorter than one segLength...skipping\n',...
            seqIn(n).trialId);
        continue
    end

    numSeg = ceil(T/segLength);

    if numSeg == 1
        cumOL      = 0;
    else
        totalOL    = (segLength*numSeg) - T;
        probs      = ones(1,numSeg-1)/(numSeg-1);
        % mnrnd is very sensitive to sum(probs) being even slightly
        % away from 1 due to floating point round-off.
        probs(end) = 1-sum(probs(1:end-1));
        randOL     = mnrnd(totalOL, probs);
        cumOL      = [0 cumsum(randOL)];
    end

    seg.trialId = seqIn(n).trialId;
    seg.T       = segLength;

    for s = 1:numSeg
        tStart = -cumOL(s) + segLength * (s-1) + 1;

        seg.segId   = s;
        seg.y       = seqIn(n).y(:,tStart:(tStart+segLength-1));

        seqOut = [seqOut seg];
    end
end


end

function remain = assignopts (opts, varargin)
% assignopts - assign optional arguments (matlab 5 or higher)
%
%   REM = ASSIGNOPTS(OPTLIST, 'VAR1', VAL1, 'VAR2', VAL2, ...)
%   assigns, in the caller's workspace, the values VAL1,VAL2,... to
%   the variables that appear in the cell array OPTLIST and that match
%   the strings 'VAR1','VAR2',... .  Any VAR-VAL pairs that do not
%   match a variable in OPTLIST are returned in the cell array REM.
%   The VAR-VAL pairs can also be passed to ASSIGNOPTS in a cell
%   array: REM = ASSIGNOPTS(OPTLIST, {'VAR1', VAL1, ...});
%
%   By default ASSIGNOPTS matches option names using the strmatch
%   defaults: matches are case sensitive, but a (unique) prefix is
%   sufficient.  If a 'VAR' string is a prefix for more than one
%   option in OPTLIST, and does not match any of them exactly, no
%   assignment occurs and the VAR-VAL pair is returned in REM.
%
%   This behaviour can be modified by preceding OPTLIST with one or
%   both of the following flags:
%      'ignorecase' implies case-insensitive matches.
%      'exact'      implies exact string matches.
%   Both together imply case-insensitive, but otherwise exact, matches.
%
%   ASSIGNOPTS useful for processing optional arguments to a function.
%   Thus in a function which starts:
%		function foo(x,y,varargin)
%		z = 0;
%		assignopts({'z'}, varargin{:});
%   the variable z can be given a non-default value by calling the
%   function thus: foo(x,y,'z',4);  When used in this way, a list
%   of currently defined variables can easily be obtained using
%   WHO.  Thus if we define:
%		function foo(x,y,varargin)
%		opt1 = 1;
%               opt2 = 2;
%		rem = assignopts('ignorecase', who, varargin);
%   and call foo(x, y, 'OPT1', 10, 'opt', 20); the variable opt1
%   will have the value 10, the variable opt2 will have the
%   (default) value 2 and the list rem will have the value {'opt',
%   20}.
%
%   See also WARNOPTS, WHO.
%
% Copyright (C) by Maneesh Sahani

ignorecase = 0;
exact = 0;

% check for flags at the beginning
while (~iscell(opts))
    switch(lower(opts))
        case 'ignorecase',
            ignorecase = 1;
        case 'exact',
            exact = 1;
        otherwise,
            error(['unrecognized flag :', opts]);
    end

    opts = varargin{1};
    varargin = varargin{2:end};
end

% if passed cell array instead of list, deal
if length(varargin) == 1 & iscell(varargin{1})
    varargin = varargin{1};
end

if rem(length(varargin),2)~=0,
    error('Optional arguments and values must come in pairs')
end

done = zeros(1, length(varargin));

origopts = opts;
if ignorecase
    opts = lower(opts);
end

for i = 1:2:length(varargin)

    opt = varargin{i};

    if ignorecase
        opt = lower(opt);
    end

    % look for matches

    if exact
        match = strmatch(opt, opts, 'exact');
    else
        match = strmatch(opt, opts);
    end

    % if more than one matched, try for an exact match ... if this
    % fails we'll ignore this option.

    if (length(match) > 1)
        match = strmatch(opt, opts, 'exact');
    end

    % if we found a unique match, assign in the corresponding value,
    % using the *original* option name

    if length(match) == 1
        assignin('caller', origopts{match}, varargin{i+1});
        done(i:i+1) = 1;
    end
end

varargin(find(done)) = [];
remain = varargin;
end