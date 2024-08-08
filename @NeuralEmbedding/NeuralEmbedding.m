classdef NeuralEmbedding < handle & ...
                        matlab.mixin.CustomDisplay 

    
    properties(Dependent,SetAccess=private)
        D   cell                                                            % Spiking original data [nUnits]x[T]
        P   cell                                                            % preprocessed data ready to project
        S   cell                                                            % Smoothed data
        E   cell                                                            % Embedded data
        M   struct
        TrialTime   double                                                  % Trial time vector
        TrialL      int64                                                   % Trial length in bins
        
        UArea
        UConditions                                                          % (String) String array listing all conditinos
    end

    properties(Dependent,Access=public)
        tMask       logical                                                 % Time vector mask, ie what time bins to use to embed data
        aMask       string                                                  % Area vector mask, ie what area to use during computation
        cMask       string                                                  % Condition vector mask, ie whar condition to use during computation
        
        Colors                                                              % (double) Matrix containing colors for each condition
    end

    % Parameters
    properties(Access=public)
        % removeInactiveNeurons
        FRateLim                = 1;
        acceptanceRatio         = .05;

        % binData
        binWidth                = 5;
        useSqrt                 = false;

        % smoothData
        causalSmoothing         = false;
        prekern                 = 250;

        % zscore
        zscore                  = true;

        % Projecting 
        useTMask                = true;
        postkern                = 0;
        Reproject               = false
        numPC                   = 3;
        VarExp                  = .8;

        % General
        useGpu                  = false
        useParallel             = false
        useaMask                = true
        usecMask                = true

        % Metrics computation
        appendM                 = false
    end

    properties (GetAccess = public,SetAccess = private)
        ProjMatrix                                                          % (Double) Projection Matrix to embedded data
        VarExplained                                                        % (Double) Variance explained by each embedded dimension

        Area    string                                                      % String array with area classification of each unit/channel
        Conditions string                                                   % String array with area classification of each unit/channel

        nUnits double
        nTrial double
        nArea  double
    end

    properties (Access = private)
        D_ cell                                                             % Raw data
        TrialTime_                                                          % original trial time
        tMask_      logical                                                 % original size mask
        tMaskSub    logical                                                 % stored subsampled mask for speed 

        aMask_      string = "AllNeurons"                                   % stored area mask
        cMask_      string = "AllConditions"                                % stored condition mask

        P_ cell                                                             % prepro data, ie binned and pruned for inactive neurons
        S_ cell                                                             % Smoothed data
        E_ cell                                                             % Embedded data
        M_ = ...
            struct('type',[],'date',[],'condition',...                      % Metrics struct storing quality matrics.
            [],'data',[],'Area',[]);

        mu  double = 0                                                     % per unit mean 
        ss  double = 1                                                     % per unit variance 
        subsampling double = 1                                             % subsampling, it is updated after binning
    end
    
    %% Constructor
    methods
        function obj = NeuralEmbedding(D,opts)
            %% NEURALEMBEDDING Construct an instance of NeuralEmbedding class.
            % D input neural data. D can be a:
            % 1x nTrial struct array with at least a data field nNueronsxT
            % nNueronsxTxnTrial double (or sparse) array
            arguments
                D
                opts.time           (1,:) {mustBeNumeric} = []
                opts.fs             (1,:) {mustBeNumeric} = 1e3
                opts.area           (1,:) {mustBeText}    = string.empty
                opts.condition      (1,:) {mustBeText}    = string.empty
                
            end


            if datareader.is.Struct(D,opts)
                [D,TrialTime,nUnits,nTrial,Condition,Area] = ...
                    datareader.convert.Struct(D,opts);

            elseif datareader.is.Double(D,opts)
                [D,TrialTime,nUnits,nTrial,Condition,Area] = ...
                    datareader.convert.Double(D,opts);
            end
            
            obj.D_ = D;
            obj.nArea = numel(unique(Area));
            obj.E_  = cell(nTrial,1 + obj.nArea);
            obj.TrialTime_ = TrialTime;

            obj.nUnits = nUnits;
            obj.nTrial = nTrial;
            obj.Conditions = Condition;
            obj.Area = Area;
            
            obj.tMask = true(size(obj.TrialTime_));
            
            obj.removeInactiveNeurons();
            obj.binData();
            obj.smoothData();
            obj.zscoreData();

        end
    end

    %% Dependent methods
    methods
        % Returns raw sparse data
        function value = get.D(obj)
            value = obj.D_;
        end
            
        % Returnes preprocess matrix, pruned of incative neurons and
        % binned. If useSqrt is active, returns sqrt(X)
        function value = get.P(obj)
            if obj.useSqrt
                value = cellfun(@(x)sqrt(x),...
                    obj.P_,...
                    'UniformOutput',false);
            else
                value = obj.P_;
            end
        end

        % Returns smoothed spikecount matrix (Instantaneous Firing Rate).
        % If zscore is active returns zscore(IFR)
        function value = get.S(obj)
            amask = obj.aMask;
            cmask = obj.cMask;

            value = cell(obj.nTrial,size(amask,1));
            for amIdx = 1:size(amask,1)
                if obj.zscore
                    value(:,amIdx) = cellfun(@(x)(x(amask(amIdx,:),obj.tMask)...
                        - obj.mu(amask(amIdx,:)))...
                        ./obj.ss(amask(amIdx,:)),...
                        obj.S_(cmask),...
                        'UniformOutput',false);
                else
                    value(amIdx) = cellfun(@(x)x(amask,obj.tMask),...
                        obj.S_(cmask),...
                        'UniformOutput',false);
                end
            end
        end

        % Returns embedded data
        function value = get.E(obj)
            amask = ismember(obj.UArea,obj.aMask_);
            cmask = obj.cMask;
            value = obj.E_(cmask,amask);
        end

        function value = get.M(obj)
            value = struct2table(obj.M_);
        end
        
        % Returns unique experimental conditions.
        function value = get.UConditions(obj)
            value = [string(unique(obj.Conditions)), "AllConditions"];
        end

        % Returns unique experimental conditions.
        function value = get.UArea(obj)
            value = [string(unique(obj.Area)), "AllNeurons"];
        end
       
        % Returns updated TrialTime wrt subsampling and tMask.
        function value = get.TrialTime(obj)
            % idx = find(obj.tMask);
            value = obj.TrialTime_(1:obj.subsampling:end);
            value = value(obj.tMask);
        end

        % Returns up to date TrialL wrt subsampling.
        function value = get.TrialL(obj)
            value = floor(sum(obj.tMask_) ./ obj.subsampling);
        end

        % Returns up to date tMask wrt subsampling.
        function value = get.tMask(obj)
            if obj.useTMask
                value = true(1,obj.TrialL);
            else
                value = obj.tMaskSub;
            end
        end
        function set.tMask(obj,val)
            if islogical(val) && ...
                    length(val) == length(obj.TrialTime_)
                obj.tMask_ = val;

                T = length(obj.TrialTime_);
                blk = repmat({ones(obj.subsampling,1)},...
                    T/obj.subsampling,1);
                A = blkdiag(blk{:});
                obj.tMaskSub = logical(round(obj.tMask_ * A./obj.subsampling));
            else
                error('Input is either not logical or has a length mismatch.')
            end
        end

        % Returns up to date aMask .
        function value = get.aMask(obj)
            if obj.useaMask
                str = obj.aMask_;
                value = obj.Area == str';
                if strcmp(str,"AllNeurons")
                    value = value | 1;
                end
            else
                str = "AllNeurons";
                value = true(1,obj.nUnits);
            end
            if obj.calledByBase,fprintf(1,'aMask (i.e. area mask) set to %s.\n',str);end
        end
        function set.aMask(obj,val)
            if isstring(val) &&...
                    any(ismember(obj.UArea,val))
                obj.aMask_ = val;
            elseif isstring(val) &&...
                    val == "" || strcmpi(val,"none") || strcmpi(val,"all")
                obj.aMask_ = "AllNeurons";
            else
                error('Input is either not strinf or does not match areas provided during initialization.')
            end
        end

        % Returns up to date aMask .
        function value = get.cMask(obj)
            if obj.usecMask
                str = obj.cMask_;
                value = strcmp(obj.Conditions,str);
                if strcmp(str,"AllConditions")
                    value = value | 1;
                end
            else
                str = "AllConditions";
                value = true(1,obj.nTrial);
            end
            if obj.calledByBase,fprintf(1,'cMask (i.e. condition mask) set to %s.\n',str);end;
        end
        function set.cMask(obj,val)
            if isstring(val) &&...
                    any(strcmp(obj.UConditions,val))
                obj.cMask_ = val;
            elseif isstring(val) &&...
                    val == "" || strcmpi(val,"none") || strcmpi(val,"all")
                obj.cMask_ = "AllConditions";
            else
                error('Input is either not string or does not match conditions provided during initialization.')
            end
        end
    end 
    %% Preprocessing methods
    methods (Access = private)
        function loadDefaultPars(obj)
            %LOADDEFUALTPARS load defualt parameters
           
            pars_.numPC = 3;

            pars_.splitUnits = [0 size(D(1).data,1)];
            obj.t = obj.TrialTime;

            pars_.seqTest = false(1,numel(D));

            pars_.endLeg_range = [1:250 size(D(1).data,2)-250:size(D(1).data,2)];
            pars_.interest_range = (-250:250)+0.5*size(D(1).data,2);
            pars_.ccaRefSig = [];

            pars_.SuperviseByConditions = false;

            pars_.D_ref = {nan};

            N_Ref = length(pars_.D_ref);
            N_Areas = length(pars_.splitUnits)-1;
            if N_Ref ~= N_Areas
                [pars_.D_ref{N_Ref+1:N_Areas}] = deal(nan);
            end

        end
    
        function removeInactiveNeurons(obj)

            obj.P_ = obj.D_;

            TT = abs(obj.TrialTime(end) - obj.TrialTime(1));
            thSpikeRate = cellfun(@(d) (sum(d,2) ./ TT) < obj.FRateLim,...
                obj.D_,...
                'UniformOutput',false);
            thSpikeRate = sum([thSpikeRate{:}],2) > (obj.nTrial * obj.acceptanceRatio);

            nanidx = false(size(obj.P_{1},1),1);
            for nn = 1:numel(obj.P_)

                obj.P_{nn}(thSpikeRate,:) = nan;
                nanidx = nanidx | any(isnan(obj.P_{nn}),2);
            end

            % remove nans
            for nn = 1:obj.nTrial
                obj.P_{nn}(nanidx,:) = 0;
                l = size(obj.P_{nn},2);
                n = ceil(max(obj.TrialTime) * 2 * obj.FRateLim);
                randomSpk = arrayfun(@(x)sparse(1,randperm(l,n),1,1,l),1:sum(nanidx),'UniformOutput',false);
                randomSpk = cat(1,randomSpk{:});
                obj.P_{nn}(nanidx,:) = randomSpk;

            end
        end

        function binData(obj)
            
            T = length(obj.TrialTime);

            % Binning as block diagonal matrix multiplication
            blk = repmat({ones(obj.binWidth,1)},...
                T/obj.binWidth,1);
            A = blkdiag(blk{:});

            obj.P_ = cellfun(@(x) x * sparse(A),obj.P_,...
                'UniformOutput',false);
            obj.subsampling = obj.binWidth;

            obj.tMaskSub = logical(round(obj.tMask_ * A./obj.subsampling));
        end

        function smoothData(obj)
        % Performs Gaussian kernel smoothing of data on binned data

            obj.S_ = cellfun(@(x)NeuralEmbedding.smoother(x,...
                obj.prekern,obj.causalSmoothing,obj.subsampling,obj.useGpu),...
                obj.P_,'UniformOutput',false);

        end

        function zscoreData(obj)

            obj.mu = mean([obj.S_{:}],2);
            obj.ss = std([obj.S_{:}],[],2);

        end
    
        function pars = assignEPars(obj,names,method)
            names = string(names);
            genericPars = struct();
            for nn = names(:)'
                genericPars.(nn) = obj.(nn);
            end
            methodPars = embedding.(method).loadParams();

            pars = NeuralEmbedding.mergestructs(methodPars,genericPars);
        end
        function pars = assignMPars(obj,names,method)
            names = string(names);
            genericPars = struct();
            for nn = names(:)'
                if nn == "",continue;end
                genericPars.(nn) = obj.(nn);
            end
            methodPars = metrics.pars.(method);

            pars = NeuralEmbedding.mergestructs(methodPars,genericPars);
        end
        function str = initMstruct(obj,data,type)
            str = struct('type',type,...
                'date',datetime ,...
                'condition',obj.cMask_,...
                'data',data,...
                'Area',obj.aMask_);
        end
    end

    %% Compute embeddings
    methods (Access=public)
         flag = findEmbedding(obj,type,Area)
    end

    %% Compute manifold metrices
    methods (Access=public)
         flag = computeMetrics(obj,type)
    end

    %% Plot data
    methods

    end

    %% Class data preview
    methods (Access = protected)
        function propgrp = getPropertyGroups(obj)
            if ~isscalar(obj)
                propgrp = getPropertyGroups@matlab.mixin.CustomDisplay(obj);
            else
                propList = struct('Units',obj.nUnits,...
                    'Trials',obj.nTrial,...
                    'Trial_names',obj.UConditions,...
                    'Areas',obj.UArea,...
                    'Area_mask',obj.aMask_,...
                    'Condition_mask',obj.cMask_);
                propgrp = matlab.mixin.util.PropertyGroup(propList);
            end
        end
    end
    %% Usefull generic methods
    methods(Static)
        % Gaussian kernel smoothing of data across time
        function Xs = smoother(X,kern,causal,binsize,gpu)
            %
            % Gaussian kernel smoothing of data across time.
            %
            % INPUTS:
            %
            % yIn      - input data (yDim x T)
            % obj.prekern   - standard deviation of Gaussian kernel, in msec
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
            % Based on @ 2009 Byron Yu -- byronyu@stanford.edu

            if (kern == 0)
                return;
            end

            % Filter half length
            % Go 3 standard deviations out
            fltHL = ceil(3 * kern / binsize);

            % Length of flt is 2*fltHL + 1
            flt = normpdf(-fltHL*binsize : binsize : fltHL*binsize, 0, kern);

            if causal
                flt(1:fltHL) = 0;
            end

            if gpu
                flt = gpuArray(flt);
            end
            [n,T]         = size(X);

            % Normalize by sum of filter taps actually used
            nm = ones(n,1) * conv(ones(1, T),flt,"same");
            Xs = conv2(full(X), flt, "same") ./ nm;
        end
        
        % Merges structures. If fields are present in both, only Y is kept
        function Z = mergestructs(x,y)
                       Xnames = fieldnames(x);
            Ynames = fieldnames(y);
            XinY = ismember(Xnames,Ynames);                         % field of A present also in B
            
            Xvals = struct2cell(x);
            Yvals = struct2cell(y);

            Z = cell2struct(...
                [Xvals(~XinY) ;Yvals ],...
                [Xnames(~XinY);Ynames]);
        end
        
        % Returns a matrix of explained variances
        function R2 = explainedVar(varargin)
            %EXPLAINEDVAR returns a matrix of explained variances of dimension NxN
            %where N is the number of inputs. Inputs must have the same size.
            sz1 = size(varargin{1});
            if sz1(1) < sz1(2)
                varargin{1} = varargin{1}';
                sz1 = fliplr(sz1);
            end %fi
            for ii = 2:nargin
                if ~all(diag(...
                        sz1 == size(varargin{ii})' | fliplr(sz1) == size(varargin{ii})' ...
                        ))
                    error('Input sizes must be consistent');
                elseif all(diag(fliplr(sz1) == size(varargin{ii})'))
                    varargin{ii} = varargin{ii}';
                end
            end %ii
            R2 = zeros(nargin);
            for ii = 1:nargin-1
                X = varargin{ii};
                for jj = ii+1:nargin
                    Y = varargin{jj};
                    S = cov(X);
                    Srec = cov(Y);
                    R2(ii,jj) = trace(Srec)/trace(S);
                    if R2(ii,jj) > 1
                        R2(jj,ii) = trace(S)/trace(Srec);
                        R2(ii,jj) = 1;
                    else
                        R2(jj,ii) = 1;
                    end%fi
                end% jj
            end%ii
        end %explainedVar

        % Returns true if current context is two level below base
        function value = calledByBase()
            % calledByBase Returns true if current context is two level
            % below base. This means that if it is called within a function
            % it will be true if that function was called by base.
            stack = dbstack('-completenames');
            value = numel(stack) < 3;
        end

    end
end

