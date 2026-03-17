classdef NeuralEmbedding < handle & ...
                        matlab.mixin.CustomDisplay 

    
    properties(Dependent,SetAccess=private)
        D   cell                                                            % Spiking original data [nUnits]x[T]
        P   cell                                                            % preprocessed data ready to project
        S   cell                                                            % Smoothed data
        E   cell                                                            % Embedded data
        M   struct

        W   cell                                                            % Projection Matrix
        Winv cell                                                           % inverse projection matrix

        TrialTime   double                                                  % Trial time vector
        TrialL      int64                                                   % Trial length in bins
        
        UArea
        UConditions                                                          % (String) String array listing all conditions

        Events  struct                                                       % Behavioral events struct array (fields: Ts, name, trial, data)
    end

    properties(Dependent,Access=public)
        tMask       cell                                                 % Time vector mask, ie what time bins to use to embed data
        aMask       string                                                  % Area vector mask, ie what area to use during computation
        cMask       string                                                  % Condition vector mask, ie whar condition to use during computation
        
        Colors                                                              % (double) Matrix containing colors for each condition
    
        PreKern
        PostKern
        BinWidth

        VarExplained                                                        %(Double) Variance explained by each embedded dimension
        CanonCorr                                                           %(Cell) Canonical correlation (where relevant)
        NumPC                   

    end

    % Parameters
    properties(Access=public)
        % removeInactiveNeurons
        FRateLim                = 1;
        acceptanceRatio         = .05;

        % binData
        binwidth                = 5;
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
        VarExp                  = .8;

        % MCCA regularization parameter
        mcca_k                  = 0.9;  % Add this line

        % General
        useGpu                  = false
        useParallel             = false
        useaMask                = true
        usecMask                = true

        % Metrics computation
        appendM                 = false
        baseline_idx            = []
        signal_idx              = []

        % Metadata
        Meta                    = struct("AnName","","ExpGroup","");
    end

    properties (GetAccess = public,SetAccess = private)
        Animal
        Session

        ProjMatrix                                                          % (Double) Projection Matrix to embedded data

        Area    string                                                      % String array with area classification of each unit/channel
        Conditions string                                                   % String array with area classification of each unit/channel

        nUnits double
        nTrial double
        nArea  double
        nCondition double

        currentEmbeddingMethod string = ""
    end

    properties (Access = private)
        D_ cell                                                             % Raw data
        TrialTime_                                                          % original trial time
        tMask_      cell                                                    % original size mask
        tMaskSub    cell                                                    % stored subsampled mask for speed 

        aMask_      string = "AllNeurons"                                   % stored area mask
        cMask_      string = "AllConditions"                                % stored condition mask

        M_ = ...
            struct('type',[],'date',[],'condition',...                      % Metrics struct storing quality matrics.
            [],'data',[],'Area',[]);
        Events_ = struct('Ts',{},'name',{},'trial',{},'data',{});          % Events struct array (Ts, name, trial, data)
        W_ cell                                                            % projection matrix
        Winv_ cell                                                         % inverse projection matrix
        mu  double = 0                                                     % per unit mean 
        ss  double = 1                                                     % per unit variance 
        subsampling double = 1                                             % subsampling, it is updated after binning
        homogeneous logical = false                                        % flag for trial homogenuity. If all trials all equally long, this is 0

        VarExplained_ double                                               % cell storing variance explained values
        CanonCorrelation_ cell                                             % cell storing canonical correlation iof applicable
        numPC double = 6
    end

    % Transient stores
    properties (Access = private, Transient = true)
        P_ cell                                                             % prepro data, ie binned and pruned for inactive neurons
        S_ cell                                                             % Smoothed data
        E_ cell                                                             % Embedded data
    end
    
    %% Constructor
    methods
        function obj = NeuralEmbedding(D,opts)
       % NEURALEMBEDDING Construct an instance of NeuralEmbedding class.
        %   OBJ = NeuralEmbedding(D) creates an instance of NeuralEmbedding
        %   class, where D is either a 1x nTrial struct array with at least a
        %   data field nNueronsxT or nNueronsxTxnTrial double (or sparse) array.
        %   All the optional parameters are passed through the opts structure.
        %   The available options are:
        %       time : (1,:) time vector, if empty it is assumed to be
        %       fs   : (1,:) sampling frequency (in Hz)
        %       area : (1,:) area labels for each unit/channel
        %       condition : (1,:) condition labels for each trial
            arguments
                D
                opts.time           (1,:) {mustBeVector} = []
                opts.fs             (1,:) {mustBeNumeric} = 1e3
                opts.area           (1,:) {mustBeText}    = string.empty
                opts.condition      (1,:) {mustBeText}    = string.empty
                
                opts.pars           (1,:) struct = struct.empty

                opts.animal         (1,:) string = ""
                opts.session        (1,:) string = ""
            end


            % Determine the type of data
            if datareader.is.Struct(D,opts)
                % If D is a struct convert it to the standard format
                [D,TrialTime,nUnits,nTrial,Condition,Area,Dishomogeneous] = ...
                    datareader.convert.Struct(D,opts);
            elseif datareader.is.Double(D,opts)
                % If D is a numeric array convert it to the standard format
                [D,TrialTime,nUnits,nTrial,Condition,Area,Dishomogeneous] = ...
                    datareader.convert.Double(D,opts);
            elseif datareader.is.Cell(D,opts)
                % If D is a cell array convert it to the standard format
                [D,TrialTime,nUnits,nTrial,Condition,Area,Dishomogeneous] = ...
                    datareader.convert.Cell(D,opts);
            end
            
            % Store the raw data
            obj.D_ = D;

            obj.homogeneous = ~Dishomogeneous;
            % Store the number of areas
            obj.nArea = numel(unique(Area));
            obj.nCondition = numel(unique(Condition));
            % Pre-allocate the embedded data
            obj.E_  = cell(nTrial,1 + obj.nArea);
            obj.W_  = cell(1,1 + obj.nArea);
            obj.Winv_  = cell(1,1 + obj.nArea);

            % Store the original trial time
            obj.TrialTime_ = TrialTime(:);            
            % Store the number of units and trials
            obj.nUnits = nUnits;
            obj.nTrial = nTrial;
            % Store the condition labels
            obj.Conditions = Condition(:);
            % Store the area labels
            obj.Area = Area(:);

            % Store the original time mask
            obj.tMask = cellfun(@(t) true(length(t),1),obj.TrialTime_,...
                'UniformOutput',false);
            
            obj.Animal = opts.animal;
            obj.Session = opts.session;


            obj.setPars(opts.pars)

            performPrePro(obj);
        end

        function performPrePro(obj)
            %PERFORMPREPRO Perform all preprocessing operations on the data
            %   This function removes inactive neurons, bins the data, smooths the
            %   data and z-scores the data. The parameters for these operations are
            %   stored in the properties of the object.
            % 
            % Inactive neurons' pruning, <a href="matlab:help NeuralEmbedding.removeInactiveNeurons">removeInactiveNeurons()</a>
            % Pars:
            %   - FRateLim: the minimum spike rate for a neuron to be considered
            %     active
            %   - acceptanceRatio: the fraction of trials that must have a spike rate
            %     above FRateLim for a neuron to be considered active
            % 
            % Data binning, <a href="matlab:help NeuralEmbedding.binData">binData()</a>
            % Pars:
            %   - binwidth: the width of the bins in samples
            % 
            % Data smoothing, <a href="matlab:help NeuralEmbedding.smoothData">smoothData()</a>
            % Pars:
            %   - causalSmoothing: whether to use causal or acausal smoothing
            %   - prekern: the width of the smoothing kernel in samples
            % 
            % Data Z-scoring, <a href="matlab:help NeuralEmbedding.zscoreData">zscoreData()</a>
            % Pars:
            %   - zscore: whether to z-score the data or not

            
            obj.subsampling = 1;
            obj.tMask = obj.tMask_;

            % Remove inactive neurons
            obj.removeInactiveNeurons();

            % Bin the data
            obj.binData();

            % Smooth the data
            obj.smoothData();

            % Z-score the data
            obj.zscoreData();
        end
    
        function addEvents(obj,evts)
        % ADDEVENTS Add behavioral events to the NeuralEmbedding object.
        %   ADDEVENTS(OBJ, EVTS) adds events stored in the struct array EVTS to
        %   the object. EVTS must be a struct array with the following fields:
        %     Ts    - (double) timestamp relative to trial start (0 = trial alignment)
        %     name  - (string or char) event name/label
        %     trial - (double) trial index (1-based)
        %     data  - (optional) any additional data (reserved for future use)
        %
        %   To convert events with absolute timestamps to the required relative
        %   format, use the static utility:
        %     evts = NeuralEmbedding.absoluteToRelativeEvents(evts, ...
        %         'trialStartReference', trialStartTimes)
        %
        %   See also NeuralEmbedding.absoluteToRelativeEvents
            arguments
                obj  NeuralEmbedding
                evts struct
            end

            evts = evts(:);  % normalise to column vector

            ff = fieldnames(evts);
            requiredFields = {'Ts','name','trial'};
            if ~all(ismember(requiredFields, ff))
                error('NeuralEmbedding:addEvents:invalidFields', ...
                    ['Event structure must contain the fields: ' ...
                    strjoin(requiredFields,', ') '.']);
            end

            % Add optional data field if absent
            if ~ismember('data', ff)
                [evts.data] = deal([]);
            end

            % Validate trial indices
            trialIdx = [evts.trial];
            if any(trialIdx < 1) || any(trialIdx > obj.nTrial)
                error('NeuralEmbedding:addEvents:invalidTrial', ...
                    'Trial indices must be integers in [1, %d].', obj.nTrial);
            end

            obj.Events_ = [obj.Events_; evts(:)];
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
            amask = obj.aMask;
            cmask = obj.cMask;
            Pout = cell(sum(any(cmask,2)),size(amask,2));
            for amIdx = 1:size(amask,2)
                if obj.zscore
                    Pout(:,amIdx) = cellfun(@(x,tmask)(x(amask(:,amIdx),tmask)...
                        - obj.mu(amask(:,amIdx)))...
                        ./obj.ss(amask(:,amIdx)),...
                        obj.S_(cmask),obj.tMask(cmask),...
                        'UniformOutput',false);
                else
                    Pout(amIdx) = cellfun(@(x,tmask)x(amask(:,amIdx),tmask),...
                        obj.S_(cmask),obj.tMask(cmask),...
                        'UniformOutput',false);
                end
            end
            if obj.useSqrt
                value = cellfun(@(x)sqrt(x),...
                    Pout,...
                    'UniformOutput',false);
            else
                value = Pout;
            end
        end

        % Returns smoothed spikecount matrix (Instantaneous Firing Rate).
        % If zscore is active returns zscore(IFR)
        function value = get.S(obj)
            amask = obj.aMask;
            cmask = obj.cMask;

            value = cell(sum(any(cmask,2)),size(amask,2));
            for amIdx = 1:size(amask,2)
                if obj.zscore
                    value(:,amIdx) = cellfun(@(x,tmask)(x(amask(:,amIdx),tmask)...
                        - obj.mu(amask(:,amIdx)))...
                        ./obj.ss(amask(:,amIdx)),...
                        obj.S_(cmask),obj.tMask(cmask),...
                        'UniformOutput',false);
                else
                    value(amIdx) = cellfun(@(x,tmask)x(amask(:,amIdx),tmask),...
                        obj.S_(cmask),obj.tMask(cmask),...
                        'UniformOutput',false);
                end
            end
        end

        % Returns embedded data
        function value = get.E(obj)
            amask = ismember(obj.UArea,obj.aMask_);
            cmask = obj.cMask;
            value = obj.E_(cmask,amask);
            value = cellfun(@(e)e(1:obj.numPC,:),value, ...
                'UniformOutput',false,'ErrorHandler',@(S,n)errorFunc(S,obj.numPC));

            function e = errorFunc(S,varargin)
                e = zeros(varargin{1},0);
            end

        end

        function value = get.W(obj)
            amask = ismember(obj.UArea,obj.aMask_);
            value = obj.W_(amask);
            value = cellfun(@(w)w(1:obj.numPC,:),value, ...
                'UniformOutput',false,'ErrorHandler',@(S,n)errorFunc(S,obj.numPC));
            function e = errorFunc(S,varargin)
                e = zeros(varargin{1},0);
            end
        end
        function value = get.Winv(obj)
            amask = ismember(obj.UArea,obj.aMask_);
            if isempty(obj.Winv_{amask})
                obj.Winv_{amask} = pinv(obj.W_{amask});
            end
            value = obj.Winv_(amask);
            value = cellfun(@(w)w(:,1:obj.numPC),value, ...
                'UniformOutput',false,'ErrorHandler',@(S,n)errorFunc(S,obj.numPC));
            function e = errorFunc(S,varargin)
                e = zeros(0,varargin{1});
            end
        end

        function value = get.M(obj)
            this = obj.M_;
            if all(arrayfun(@(t)all(structfun(@isempty,t)),this))
                this = this([]);
            end

            [this.animal] = deal(obj.Animal);
            [this.session] = deal(obj.Session);
            args = {};
            if isscalar(this)
                args = [args,{"AsArray",true}];
            end
            value = struct2table(this,args{:});
            
        end
        
        % Returns unique experimental conditions.
        function value = get.UConditions(obj)
            value = [string(unique(obj.Conditions(:))); "AllConditions"];
        end

        % Returns unique experimental conditions.
        function value = get.UArea(obj)
            value = [string(unique(obj.Area(not(ismissing(obj.Area))))); "AllNeurons"];
        end

        % Returns the events struct array.
        function value = get.Events(obj)
            value = obj.Events_;
        end
       
        % Returns updated TrialTime wrt subsampling and tMask.
        function value = get.TrialTime(obj)
            % idx = find(obj.tMask);
                value = cellfun(@(t)t(1:obj.subsampling:end),...
                obj.TrialTime_(obj.cMask),...
                'UniformOutput',false);
            value = cellfun(@(t,tm)t(tm),...
                value(:),obj.tMask(obj.cMask),...
                'UniformOutput',false);

        end

        % Returns up to date TrialL wrt subsampling.
        function value = get.TrialL(obj)
            value = cellfun(@(tmsub)floor(sum(tmsub)),...
               obj.tMaskSub(obj.cMask),'UniformOutput',false);
        end

        % Returns up to date tMask wrt subsampling.
        function value = get.tMask(obj)
            if obj.useTMask
                value = obj.tMaskSub(:);
            else
                value = cellfun(@(tms)tms | 1,...
                    obj.tMaskSub,...
                    'UniformOutput',false);
            end
        end
        %SET.TMASK Set the time mask of the data
        %   set.tMask(val) sets the time mask of the data to val.
        %   val should be either a logical array or a cell array of logical
        %   arrays. If val is a logical array, it should have the same length
        %   as the number of time points in the data. If val is a cell array of
        %   logical arrays, it should have the same length as the number of
        %   trials in the data.
        function set.tMask(obj,val)
            % Check if the input is a logical array or a cell array of logical
            % arrays
            if islogical(val) && ...
                    obj.homogeneous && ...
                      length(val) == length(obj.TrialTime_{1})
                % If the input is a logical array, replicate it to match the
                % number of trials
                obj.tMask_ = repmat({val},obj.nTrial,1);

            elseif iscell(val) && ...
                    obj.homogeneous && ...
                      length(val{1}) == length(obj.TrialTime_{1})
                % If the input is a cell array of logical arrays, replicate the
                % first element to match the number of trials
                obj.tMask_ = repmat({val{1}(:)},obj.nTrial,1);

            elseif iscell(val) && ...
                    ~obj.homogeneous && ...
                      length(val) == obj.nTrial && ...
                        all(cellfun(@(v,t)length(v) == length(t), val, obj.TrialTime_))
                % If the input is a cell array of logical arrays, check if the
                % length of each element matches the number of time points in
                % the data
                if all(cellfun(@(m,t)length(m)==length(t),...
                        val,obj.TrialTime_))
                    % If the length matches, set the time mask to the input
                    obj.tMask_ = val(:);
                else
                    % If the length does not match, throw an error
                    error('Input is either not logical or has a length mismatch.')
                end

            else
                % If the input is neither a logical array nor a cell array of
                % logical arrays, throw an error
                error('Input is either not logical or has a length mismatch.')
            end

            % Update the subsampled time mask
            if obj.subsampling ~=1
                T     = cellfun(@length,obj.TrialTime_);
                Tdown = floor(T./obj.subsampling);
                Trest = T - Tdown*obj.subsampling;

                % Binning as block diagonal matrix multiplication
                blk = arrayfun(@(t)...
                    repmat({sparse(1:obj.subsampling,1,1)},t,1),...
                    Tdown,'UniformOutput',false);

                A = arrayfun(@(thisblk,tr,t)[blkdiag(thisblk{1}{:});sparse(tr,t)],...
                    blk,Trest,Tdown,...
                    'UniformOutput',false);

                tMaskSub_ = cellfun(@(tm,a)logical(round(tm' * a./obj.subsampling)),...
                    obj.tMask_,A,...
                    'UniformOutput',false);
            else
                tMaskSub_ = obj.tMask_;
            end
            obj.tMaskSub = tMaskSub_(:);
        end

        % Returns up to date aMask .
        %
        %   val = obj.get.aMask() returns the area mask.
        %
        %   The area mask is a logical array that marks the neurons that
        %   satisfy the area mask. If the area mask is not specified, all
        %   neurons are marked as true.
        function value = get.aMask(obj)
            if obj.useaMask
                % Get the area mask as a string
                str = obj.aMask_;
                % Check if the area mask string matches one of the areas
                % provided during initialization
                value = obj.Area == str';
                % If the area mask is "AllNeurons", mark all neurons as true
                if any(strcmp(str,"AllNeurons"))
                    value(:,strcmp(str,"AllNeurons")) = true;
                end
            else
                % If the area mask is not specified, mark all neurons as true
                str = "AllNeurons";
                value = true(1,obj.nUnits);
            end
            if obj.calledByBase,fprintf(1,'aMask (i.e. area mask) set to %s.\n',str);end
        end
        % Sets the area mask to the value specified by the input string.
        %
        %   The input string should match one of the areas provided during
        %   initialization. If the input string is empty, or matches "none" or
        %   "all", all neurons are marked as true.
        function set.aMask(obj,val)

            if isstring(val)
                    uAIdx = ismember(obj.UArea(1:end-1),val);
                    obj.aMask_ = obj.UArea(uAIdx);

                    if any(val == "" | strcmpi(val,"none") | strcmpi(val,"all") | strcmpi(val,"AllNeurons"))
                        obj.aMask_ = [ obj.aMask_;"AllNeurons"];
                    end
            else
                error('Input must be string')
            end
        end

        % Returns up to date cMask .
        function value = get.cMask(obj)
            % GET.CMASK get condition mask
            %
            %   val = get.cMask() returns the condition mask.
            %
            %   The condition mask is a logical array that marks the trials that
            %   satisfy the condition mask. If the condition mask is not specified,
            %   all trials are marked as true.
            if obj.usecMask
                str = obj.cMask_;
                value = ismember(obj.Conditions,str);
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
            % SET.CMASK set condition mask
            %
            %   set.cMask(val) sets the condition mask to the string
            %   val. If the condition mask is not specified, all trials
            %   are marked as true.
            %
            %   val can be
            %
            %   - a string matching one of the conditions in
            %     obj.UConditions
            %   - "" or "none" to set condition mask to all conditions
            %   - "all" to set condition mask to all conditions
            if isstring(val) &&...
                    any(ismember(obj.UConditions,val))
                obj.cMask_ = val;
            elseif isstring(val) &&...
                    val == "" || strcmpi(val,"none") || strcmpi(val,"all")
                obj.cMask_ = "AllConditions";
            else
                error('Input is either not string or does not match conditions provided during initialization.')
            end
        end
   
        function value = get.PreKern(obj)
            value = obj.prekern;
        end
        function set.PreKern(obj,val)
            ts = diff(obj.TrialTime_{1}(1:2));
            obj.prekern = round(val*1e-3/ts);
            obj.performPrePro();
        end

         function value = get.PostKern(obj)
            value = obj.postkern;
        end
        function set.PostKern(obj,val)
            ts = diff(obj.TrialTime_{1}(1:2));
            obj.postkern = round(val*1e-3/ts);
            for ar = obj.UArea'
                obj.aMask = ar;
                obj.findEmbedding(obj.currentEmbeddingMethod,true);
            end
        end

        function set.BinWidth(obj,val)
            ts = diff(obj.TrialTime_{1}(1:2));
            obj.binwidth = round(val*1e-3/ts);
            obj.performPrePro();
        end
        function value = get.BinWidth(obj)
            ts = diff(obj.TrialTime_{1}(1:2));
            value = obj.binwidth*ts*1e3;
        end
    
        function value = get.VarExplained(obj)
            amask = ismember(obj.UArea,obj.aMask_);
            if isempty(obj.VarExplained_) || ...
                    length(obj.VarExplained_) < find(amask,1,'last')
                orignial = cat(2,obj.S{:});
                projected = obj.Winv{:} * cat(2,obj.E{:});
                r2 = obj.explainedVar(orignial,projected);
                obj.VarExplained_(amask) = r2(1,2);
            end
            value = obj.VarExplained_(amask);
        end
        function value = get.CanonCorr(obj)
            amask = ismember(obj.UArea,obj.aMask_);
            if isempty(obj.CanonCorrelation_) || ...
                    length(obj.CanonCorrelation_) < find(amask,1,'last')
                % TODO, not sure if makes sense
            end
            nMask = numel(obj.aMask_);
            value_tmp = cell(1,(nMask-1)*(nMask-2)/2);
            str = repmat("",(nMask-1)*(nMask-2)/2);
            idx = 1;
            for ii = 1:nMask-1
                for jj = ii+1:nMask
                str(idx) = obj.aMask_(ii)+"_"+obj.aMask_(jj);
                value_tmp{idx} = obj.CanonCorrelation_(ii);
                idx = idx + 1;
                end
            end
            value = cell2struct(value_tmp,str);
        end

        function value = get.NumPC(obj)
            value = obj.numPC;
        end
        function set.NumPC(obj,value)
            obj.numPC = ceil(value);
            obj.VarExplained_ = repmat([],1,obj.nArea);
        end
    
    end 
    %% Preprocessing methods
    methods (Access = private)
        function setPars(obj,pars)
            %SETPARS takes care of assigning custom parameters
            
            pNames = fieldnames(pars);
            for pp = 1:numel(pNames)
                pn = pNames{pp};
                obj.(pn) = pars.(pn);
                % set(obj,pn,pars.(pn));
            end
        end
    
        
        function removeInactiveNeurons(obj)
        %% REMOVEINACTIVENEURONS Remove inactive neurons from the data
        % and replaces them with random spikes
        %
        % This function removes neurons with spike rates below a threshold
        % and replaces them with random spikes.
        %
        % Input:
        %   obj - the NeuralEmbedding object
        %
        % Returns:
        %   nothing
        %
        % Parameters:
        %   The threshold is set by the FRateLim property.
        %   The acceptanceRatio property sets the ratio of trials with
        %   spike rates above the threshold that are accepted as active.
        %

            % Copy the data into the preprocessed data structure
            obj.P_ = obj.D_;

            % Find the time difference between the last and first trial
            TT = cellfun(@(t)t(end)-t(1),obj.TrialTime,'UniformOutput',false);

            % Find the neurons with spike rates below the threshold
            thSpikeRate = cellfun(@(d,t) (sum(d,2) ./ t) < obj.FRateLim,...
                obj.D_,TT(:),...
                'UniformOutput',false);
            thSpikeRate = sum([thSpikeRate{:}],2) > (obj.nTrial * obj.acceptanceRatio);

            % Get the indices of the inactive neurons
            nanidx = false(size(obj.P_{1},1),1);
            for nn = 1:numel(obj.P_)

                % Set the inactive neurons to nan
                obj.P_{nn}(thSpikeRate,:) = nan;

                % Get the indices of the inactive neurons
                nanidx = nanidx | any(isnan(obj.P_{nn}),2);
            end

            % Calculate the number of spikes to add to each inactive neuron
            n = cellfun(@(tt)ceil((tt(end) - tt(1)) ...
                    * 2 * obj.FRateLim),...
                    obj.TrialTime);

            % Replace inactive neurons with random spikes
            for nn = 1:obj.nTrial
                obj.P_{nn}(nanidx,:) = 0;

                % Add random spikes to the inactive neurons
                l = size(obj.P_{nn},2);

                randomSpk = arrayfun(@(x)sparse(1,randperm(l,n(nn)),1,1,l),...
                    1:sum(nanidx),'UniformOutput',false);
                randomSpk = cat(1,randomSpk{:});
                obj.P_{nn}(nanidx,:) = randomSpk;

            end
        end

        function binData(obj)
        %% BINDATA Bin the data using a block diagonal matrix multiplication
        % This function bins the data using a block diagonal matrix
        % multiplication method, reducing the number of time points by the
        % bin width.
        %
        % Input:
        %   obj - the NeuralEmbedding object
        %
        % Returns:
        %   nothing
        %
        % Parameters:
        %   The bin width is set by the binwidth property. The resulting
        %   subsampling property reflects the new bin width.

            T     = cellfun(@length,obj.TrialTime_);
            Tdown = floor(T./obj.binwidth);
            Trest = T - Tdown*obj.binwidth;

            % Binning as block diagonal matrix multiplication
            blk = arrayfun(@(t)...
                repmat({ones(obj.binwidth,1)},t,1),...
                Tdown,'UniformOutput',false);

            A = arrayfun(@(thisblk,tr,t)[blkdiag(thisblk{1}{:});sparse(tr,t)],...
                blk,Trest,Tdown,...
                'UniformOutput',false);

            obj.P_ = cellfun(@(x,a) x * sparse(a),...
                obj.P_,A(:),...
                'UniformOutput',false);

            obj.subsampling = obj.binwidth;

            tMaskSub_ = cellfun(@(tm,a)logical(round(tm' * a./obj.subsampling)),...
                obj.tMask_,A,...
                'UniformOutput',false);
            obj.tMaskSub = tMaskSub_(:);
        end


        function smoothData(obj)
            %% SMOOTHEDDATA Smooth the preprocessed data using a Gaussian kernel
            %
            % Input:
            %   obj - the NeuralEmbedding object
            %
            % Returns:
            %   nothing
            %
            % Parameters:
            %   The smoothing kernel width is set by the prekern property.
            %   The causalSmoothing property determines if the smoothing is
            %   causal or acausal. If the useGpu property is true, the smoothing
            %   is performed using a GPU. The resulting smoothed data is stored
            %   in the S_ property.

            obj.S_ = cellfun(@(x)NeuralEmbedding.smoother(x,...
                obj.prekern,obj.causalSmoothing,obj.useGpu),...
                obj.P_,'UniformOutput',false);

        end

        function zscoreData(obj)
        %% ZSCOREDATA Z-score the data using the mean and standard deviation calculated from all trials
        % This function computes the mean and standard deviation from all the trials
        % and uses them to z-score the data.
        %
        % Input:
        %   obj - the NeuralEmbedding object
        %
        % Returns:
        %   nothing
        %
        % The z-scored data is stored in the S_ property, and the calculated mean
        % and standard deviation are stored in the mu and ss properties, respectively.
        % The data is z-scored using the formula: data = (data - mu) ./ ss;

            % Calculate mean and std from all the data
            obj.mu = mean([obj.S_{:}],2);
            obj.ss = std([obj.S_{:}],[],2);

        end
    
        function pars = assignEPars(obj,names,method)
            %% ASSIGNEPARS Assign default parameters to the embedding method.
            %
            % obj.assignEPars(names,method) assigns the default parameters to the
            % embedding method. The names of the parameters are expected to be in the
            % names input. The method is expected to be in the method input. The
            % resulting parameters are stored in the pars output.
            %
            % The parameters are assigned as follows:
            %   1. The parameters are loaded from the method using the loadParams
            %      method.
            %   2. The default parameters are assigned from the properties of the
            %      NeuralEmbedding object.
            %   3. The resulting parameters are merged using the mergestructs
            %      method.
            % Load method parameters
            if not(isscalar(obj))
                pars = assignEPars(obj(1),names,method);
                for ii=2:numel(obj)
                    pars = [pars,assignEPars(obj(ii),names,method)];
                end
                return;
            end
            methodPars = embedding.(method).loadParams();

            % Load default parameters from properties
            names = string(names);
            genericPars = struct();
            for nn = names(:)'
                genericPars.(nn) = obj.(nn);
            end

            % Merge parameters
            pars = NeuralEmbedding.mergestructs(methodPars,genericPars);
        end
        function pars = assignMPars(obj,names,method)
           %% ASSIGNMPARS Assign default parameters to the metric method.
            %
            % obj.assignMPars(names,method) assigns the default parameters to the
            % metric method. The names of the parameters are expected to be in the
            % names input. The method is expected to be in the method input. The
            % resulting parameters are stored in the pars output.
            %
            % The parameters are assigned as follows:
            %   1. The parameters are loaded from the method using the loadParams
            %      method.
            %   2. The default parameters are assigned from the properties of the
            %      NeuralEmbedding object.
            %   3. The resulting parameters are merged using the mergestructs
            %      method.
            
            % Load method parameters
            methodPars = metrics.pars.(method);

            % Load default parameters from properties
            names = string(names);
            genericPars = struct();
            for nn = names(:)'
                if nn == "",continue;end
                genericPars.(nn) = obj.(nn);
            end

            % Merge parameters
            pars = NeuralEmbedding.mergestructs(methodPars,genericPars);
        end
 
        function str = initMstruct(obj,data,type)
           %% INITMSTRUCT Initialize a metrics structure
            %
            % obj.initMstruct(data,type) initializes a metrics structure with the
            % data and type inputs. The resulting structure is stored in the str
            % output.
            %
            % Input:
            %   data - the data to be stored in the structure.
            %   type - the type of the data to be stored in the structure.
            %
            % Returns:
            %   str - the initialized structure.
            str = struct('type',type,...
                'date',datetime ,...
                'condition',obj.cMask_,...
                'data',data,...
                'Area',obj.aMask_);

            if strcmp(type,'DPrime')
                str.Area = strjoin(str.Area,"∪");
            end
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
        function plot3(obj,maxT)
        % PLOT3  Plot embedded neural trajectories in the first three dimensions.
        %   PLOT3(OBJ) plots up to 40 trials (default). Trials are coloured by
        %   time and selected as the closest to the median trajectory.
        %   If events have been added via addEvents, they are overlaid as
        %   markers on the trajectories and a legend is shown.
        %
        %   PLOT3(OBJ, MAXT) uses at most MAXT trajectories.
        %
        %   See also NeuralEmbedding.addEvents
            if nargin < 2
                maxT = 40;
            end

            if not(isscalar(obj))
                arrayfun(@(o)o.plot3(maxT),obj);
                return;
            end

            % Cache embedded data (filtered by current cMask/aMask)
            E_ = obj.E;

            reducedE_ = cellfun(@(x)[x(1:3,:) nan(3,1)], ...
                E_, ...
                'UniformOutput',false);            
            t = cellfun(@(t)[t(:)' nan], ...
                obj.TrialTime, ...
                'UniformOutput',false);
            nT = sum(obj.cMask);
            MaxLines = min(nT,maxT);
            % idx = randperm(nT,MaxLines);
            closestIdx = findClosestN(reducedE_,MaxLines);

            % Map filtered-trial indices to absolute trial indices for event lookup
            cMaskIdx = find(obj.cMask(:));
            selectedTrialIdx = cMaskIdx(closestIdx);

            t = [t{closestIdx}];

            % Prepare event-overlay data (unique names, colours, markers)
            evts = obj.Events_;
            hasEvents = ~isempty(evts);
            if hasEvents
                evtNames  = string({evts.name});
                uEvtNames = unique(evtNames, 'stable');
                nEvtTypes = numel(uEvtNames);
                evtMarkers = {'o','s','^','v','d','p','h','*','+','x'};
                evtColors  = lines(nEvtTypes);
            end

            for aa = 1:size(reducedE_,2)
                reducedE = [reducedE_{closestIdx,aa}];
                fig = figure;
                ax  = axes(fig);
                surface(ax, [reducedE(1,:);reducedE(1,:)], ...
                    [reducedE(2,:);reducedE(2,:)], ...
                    [reducedE(3,:);reducedE(3,:)], ...
                    [t;t], ...
                    'facecol','no',...
                    'edgecol','interp',...
                    'linew',1)

                % Overlay events as 3-D markers
                if hasEvents
                    hold(ax,'on');
                    legendHandles = gobjects(1, nEvtTypes);
                    nValid = 0;
                    for en = 1:nEvtTypes
                        thisName  = uEvtNames(en);
                        nameIdx   = evtNames == thisName;
                        thisEvts  = evts(nameIdx);
                        thisTrIdx = [thisEvts.trial];

                        xPts = zeros(1,0);
                        yPts = zeros(1,0);
                        zPts = zeros(1,0);
                        for tr_i = 1:numel(closestIdx)
                            trAbsIdx  = selectedTrialIdx(tr_i);
                            trEvtMask = thisTrIdx == trAbsIdx;
                            if ~any(trEvtMask), continue; end

                            trialT = obj.TrialTime{closestIdx(tr_i)};
                            trialE = E_{closestIdx(tr_i), aa};
                            if isempty(trialE) || size(trialE,1) < 3
                                continue;
                            end
                            trialE3 = trialE(1:3, :);

                            for ev = reshape(find(trEvtMask),1,[])
                                [~, tIdx] = min(abs(trialT - thisEvts(ev).Ts));
                                xPts(end+1) = trialE3(1, tIdx); %#ok<AGROW>
                                yPts(end+1) = trialE3(2, tIdx); %#ok<AGROW>
                                zPts(end+1) = trialE3(3, tIdx); %#ok<AGROW>
                            end
                        end

                        if ~isempty(xPts)
                            mk = evtMarkers{mod(en-1, numel(evtMarkers)) + 1};
                            nValid = nValid + 1;
                            legendHandles(nValid) = scatter3(ax, xPts, yPts, zPts, ...
                                50, evtColors(en,:), mk, 'filled', ...
                                'DisplayName', thisName, ...
                                'LineWidth', 1.5);
                        end
                    end

                    if nValid > 0
                        legend(ax, legendHandles(1:nValid));
                    end
                end

                title(ax, obj.Animal + " " + obj.Session + " " + obj.aMask_(aa))
                xlabel(ax, 'Dimension 1');
                ylabel(ax, 'Dimension 2');
                zlabel(ax, 'Dimension 3');
                colorbar(ax)
            end

            function idx = findClosestN(traj,N)
                m = median(cat(3,traj{:}),3);
                % [dist,idx] = sort(cellfun(@(l) ...
                %     norm(l(:,1:end-1) - m(:,1:end-1)),traj));
                [~,idx] = sort( ...
                    cellfun(@(l) ...
                        max(sum(l(:,1:end-1) - m(:,1:end-1),2)), ...
                    traj) ...
                              );
                idx = idx(1:N);
            end

            
        end


        function peth(obj)
            if not(isscalar(obj))
                arrayfun(@(o)o.peth,obj);
                return;
            end

            if not(obj.homogeneous)
                warning("PETH is not available for Dishomogeneous data.%sAborting.",newline);
                return;
            end
            f = figure('Units','pixels','Position',[4 42 1100 940],'Color',[1 1 1]);
            BakCond = obj.cMask_;
            BakArea = obj.aMask_;
            ii = 1;
            for aa = 1:numel(obj.UArea)
                thisArea = obj.UArea(aa);
                obj.aMask = thisArea;
                for uu = 1:numel(obj.UConditions)
                    thisCond = obj.UConditions(uu);
                    obj.cMask = thisCond;


                    dat = mean(cat(3,obj.S{:}),3);
                    [~, MaxIdx] = max(dat,[],2);
                    [~,OrderedIdx] = sort(MaxIdx);


                    T = obj.TrialTime{1};
                    ax = subplot(obj.nArea+1,obj.nCondition+1,ii);

                    imagesc(ax,T,1:obj.nUnits,dat(OrderedIdx,:));
                    xlabel(ax,'Time [s]');
                    ylabel(ax,sprintf('Neurons %s',thisArea))
                    ax.YAxis.TickValues = [];

                    if aa==1
                        title(ax,thisCond);
                    end

                    yl = ylim(ax);
                    hold(ax,"on");
                    plot(ax,[0 0],yl .* [.8 1.2],'w','Tag','zscore');
                    ylim(ax,yl)
                    box(ax,'off');
                    % colorbar(ax)

                    ii = ii + 1;
                end
            end
            % linkprop(f.Children,'CLim');

            %     if ii == 1 || ordered
            %         [v,i] = max(Zdata{ii}(1:unitS1-1,:),[],2);
            %         [~,idxRFA] = sort(i);
            %     end
            %     imagesc(ax,t-t(floor(numel(t)/2)),1:sum(~all(0 == Zdata{ii}(idxRFA,:),2)),Zdata{ii}(idxRFA(~all(isnan(Zdata{ii}(idxRFA,:)),2)),:));
            % 
            %     title(uCond{ii});
            %     ax.YTickLabel = cellstr(num2str(idxRFA));
            %     ax.XAxis.Visible = false;
            %     hold on
            %     yl = ylim(ax);
            %     plot([0 0],yl .* [.8 1.2],'w');
            %     ylim(ax,yl)
            %     box off
            %     colorbar
            % 
            % 
            %     ax = subplot(2,numel(uCond),ii+numel(uCond));
            %     if ii == 1 || ordered
            %         [v,i] = max(Zdata{ii}(unitS1:end,:),[],2);
            %         [~,idxS1] = sort(i);
            %         idxS1 = idxS1 + unitS1-1;
            %     end
            %     imagesc(ax,t-t(floor(numel(t)/2)),1:size(Zdata{ii},1),Zdata{ii}(idxS1,:));
            %     xlabel(ax,'Time [ms]');
            %     ylabel(ax,'Neurons S1')
            %     ax.YTickLabel = cellstr(num2str(idxS1));
            %     hold on
            %     yl = ylim(ax);
            %     plot([0 0],yl .* [.8 1.2],'w');
            %     ylim(ax,yl)
            %     box off
            %     colorbar
            % 
            %     
            % 
            % sgtitle(tankObj.Children(aa).Name);

            obj.cMask = BakCond;
            obj.aMask = BakArea;
        end
  
        function animate3(obj,maxT)
            if nargin < 2
                maxT = 40;
            end

            if not(isscalar(obj))
                warning("Array object ont yet supported.")
                return;
            end
            reducedE = cellfun(@(x)[x(1:3,:) nan(3,1)], ...
                obj.E, ...
                'UniformOutput',false);
            t = cellfun(@(t)[t(:)' nan], ...
                obj.TrialTime, ...
                'UniformOutput',false);
            nT = sum(obj.cMask);
            MaxLines = min(nT,maxT);
            % idx = randperm(nT,MaxLines);
            idx = findClosestN(reducedE,MaxLines);
            reducedE = reducedE(idx);
            AxLimits = [min([reducedE{:}],[],2),max([reducedE{:}],[],2)];

            t = t(idx);
            ax = axes(figure);
            ax.set("XLim", AxLimits(1,:),"YLim",AxLimits(2,:),"ZLim",AxLimits(3,:));
            for tr = 1:numel(t)
                anl(tr) = animatedline(ax,reducedE{tr}(1,1),reducedE{tr}(2,1),reducedE{tr}(3,1));
                anl(tr).MaximumNumPoints = 50;
            end

            for tt = 1:numel(t{1})
                for tr = 1:numel(t)
                    anl(tr).addpoints(reducedE{tr}(1,tt),reducedE{tr}(2,tt),reducedE{tr}(3,tt));
                    % title(sprintf("%d",t(tt)));
                    pause(1/1500);
                end
            end
            % title(obj.Animal + " " +obj.Session)
            % xlabel('Dimension 1');ylabel('Dimension 2');zlabel('Dimension 3');
            % colorbar

            function idx = findClosestN(traj,N)
                m = median(cat(3,traj{:}),3);
                % [dist,idx] = sort(cellfun(@(l) ...
                %     norm(l(:,1:end-1) - m(:,1:end-1)),traj));
                [dist,idx] = sort( ...
                    cellfun(@(l) ...
                    max(sum(l(:,1:end-1) - m(:,1:end-1),2)), ...
                    traj) ...
                    );
                idx = idx(1:N);
            end

            
        end

    end

    %% Class data preview
    methods (Access = protected)
        function propgrp = getPropertyGroups(obj)
            if ~isscalar(obj)
                propgrp = getPropertyGroups@matlab.mixin.CustomDisplay(obj);
            else
                propList = struct('Animal',obj.Animal,...
                    'Session',obj.Session,...
                    'Units',obj.nUnits,...
                    'Trials',obj.nTrial,...
                    'Events',numel(obj.Events_),...
                    'Trial_names',obj.UConditions,...
                    'Areas',obj.UArea,...
                    'Area_mask',obj.aMask_,...
                    'Condition_mask',obj.cMask_);
                propgrp = matlab.mixin.util.PropertyGroup(propList);
            end
        end
    
        function sobj = saveobj(obj)
        %     % sobj = struct(obj);
        %     % sobj = rmfield(sobj, ...
        %     %     ["D","P","S","E","M"]);
        %     E_ = obj.E_;S_ = obj.S_; P_ = obj.P_;
        %     [obj.E_,obj.S_,obj.P_] = deal({});
            sobj = (obj);
        %     [obj.E_,obj.S_,obj.P_] = deal(E_,S_,P_);
        end

        
    end
    %% Usefull generic methods
    methods(Static)
        function evts = absoluteToRelativeEvents(evts, varargin)
        % ABSOLUTETORELATIVEEVENTS Convert event timestamps from absolute to relative time.
        %   EVTS = ABSOLUTETORELATIVEEVENTS(EVTS, 'trialStartReference', REF) subtracts
        %   each event's trial-start time from its Ts field, so that the
        %   returned struct has Ts values relative to the beginning of the
        %   trial (0 = trial alignment / trial start).
        %
        %   Inputs:
        %     evts            - struct array with fields Ts (absolute recording time),
        %                       name, trial, and optionally data. Events without a
        %                       valid Ts are discarded.
        %     trialStartReference - either:
        %                       * numeric vector (1 x nTrials) or (nTrials x 1) with
        %                         trial-start timestamps in the same absolute time base
        %                         as evts.Ts, or
        %                       * event name (char/string) present in evts. In this
        %                         modality, for each trial the first event with that
        %                         name is used as trial-start reference.
        %     inferTrialFromBounds - logical flag (default false). If true, and both
        %                       'trialstart' and 'trialend' events are present in evts,
        %                       missing/invalid trial assignments are inferred by
        %                       checking where each event Ts falls within those bounds.
        %
        %   Output:
        %     evts - same struct array with Ts converted to relative time.
        %
        %   Example:
        %     % Trial starts at t = 10, 20, 30 seconds
        %     trialStarts = [10 20 30];
        %     evt.Ts    = 22;   % absolute timestamp
        %     evt.name  = "reward";
        %     evt.trial = 2;    % belongs to trial 2 (started at t=20)
        %     evt.data  = [];
        %     evt = NeuralEmbedding.absoluteToRelativeEvents(evt, ...
        %         'trialStartReference', trialStarts);
        %     % evt.Ts is now 2 (= 22 - 20)
        %
        %   See also NeuralEmbedding.addEvents
            arguments
                evts            struct
            end

            evts = evts(:);  % normalise to column vector

            ff = fieldnames(evts);
            p = inputParser();
            p.FunctionName = 'NeuralEmbedding.absoluteToRelativeEvents';
            addParameter(p,'trialStartReference',[], ...
                @(x) isnumeric(x) || ischar(x) || (isstring(x) && isscalar(x)));
            addParameter(p,'inferTrialFromBounds',false, ...
                @(x) islogical(x) && isscalar(x));
            parse(p,varargin{:});
            trialStartReference = p.Results.trialStartReference;
            inferTrialFromBounds = p.Results.inferTrialFromBounds;

            if ~ismember('Ts', ff)
                evts = evts([]);
                return;
            end

            hasTs = arrayfun(@(e) isnumeric(e.Ts) && isscalar(e.Ts) && ...
                ~isempty(e.Ts) && isfinite(e.Ts), evts);
            evts = evts(hasTs);
            if isempty(evts)
                return;
            end

            ff = fieldnames(evts);
            if ~ismember('trial', ff)
                [evts.trial] = deal([]);
                ff = fieldnames(evts);
            end

            if inferTrialFromBounds
                if ~ismember('name', ff)
                    error('NeuralEmbedding:absoluteToRelativeEvents:invalidFields', ...
                        'Event structure must contain field name when inferTrialFromBounds is true.');
                end
                evtNames = string({evts.name});
                isTrialStart = strcmpi(evtNames,'trialstart');
                isTrialEnd = strcmpi(evtNames,'trialend');
                if any(isTrialStart) && any(isTrialEnd)
                    tStart = [evts(isTrialStart).Ts];
                    tEnd = [evts(isTrialEnd).Ts];
                    nBounds = min(numel(tStart), numel(tEnd));
                    tStart = tStart(1:nBounds);
                    tEnd = tEnd(1:nBounds);
                    validBounds = tEnd >= tStart;
                    tStart = tStart(validBounds);
                    tEnd = tEnd(validBounds);

                    if ~isempty(tStart)
                        for i = 1:numel(evts)
                            trial = evts(i).trial;
                            hasValidTrial = isnumeric(trial) && isscalar(trial) && ...
                                ~isempty(trial) && isfinite(trial) && ...
                                trial >= 1 && mod(trial,1) == 0;
                            if ~hasValidTrial
                                match = find(evts(i).Ts >= tStart & evts(i).Ts <= tEnd, 1, 'first');
                                if ~isempty(match)
                                    evts(i).trial = match;
                                end
                            end
                        end
                    end
                end
            end

            if isempty(trialStartReference)
                error('NeuralEmbedding:absoluteToRelativeEvents:missingTrialStartReference', ...
                    ['Missing trial-start reference. Provide ', ...
                    '''trialStartReference'' as numeric trial-start times ', ...
                    'or as an event name present in evts.']);
            end

            if isnumeric(trialStartReference)
                trialStartTimes = trialStartReference(:)';
            else
                if ~ismember('name', ff)
                    error('NeuralEmbedding:absoluteToRelativeEvents:invalidFields', ...
                        'Event structure must contain fields Ts, name, trial.');
                end

                refName = string(trialStartReference);
                if strlength(refName) == 0
                    error('NeuralEmbedding:absoluteToRelativeEvents:invalidTrialStartReference', ...
                        'trialStartReference event name must be non-empty.');
                end

                validTrialMask = arrayfun(@(e) isnumeric(e.trial) && isscalar(e.trial) && ...
                    ~isempty(e.trial) && isfinite(e.trial) && ...
                    e.trial >= 1 && mod(e.trial,1) == 0, evts);
                if ~all(validTrialMask)
                    error('NeuralEmbedding:absoluteToRelativeEvents:invalidTrial', ...
                        ['All events must have a valid trial index to use ', ...
                        'an event name as trialStartReference.']);
                end

                trialIdx = [evts.trial];
                nTrials = max(trialIdx);
                evtNames = string({evts.name});
                isRefEvent = strcmp(evtNames, refName);
                trialStartTimes = nan(1,nTrials);
                for trial = 1:nTrials
                    idx = find(isRefEvent & trialIdx == trial, 1, 'first');
                    if isempty(idx)
                        error('NeuralEmbedding:absoluteToRelativeEvents:missingTrialStartEvent', ...
                            'No event named %s found for trial %d.', refName, trial);
                    end
                    trialStartTimes(trial) = evts(idx).Ts;
                end
            end

            if ~ismember('trial', ff)
                error('NeuralEmbedding:absoluteToRelativeEvents:invalidFields', ...
                    'Event structure must contain at least the fields: Ts, trial.');
            end

            nTrials = numel(trialStartTimes);
            for i = 1:numel(evts)
                trial = evts(i).trial;
                if ~isnumeric(trial) || ~isscalar(trial) || isempty(trial) || ...
                        ~isfinite(trial) || trial < 1 || mod(trial,1) ~= 0 || trial > nTrials
                    error('NeuralEmbedding:absoluteToRelativeEvents:invalidTrial', ...
                        'Trial index %d is out of range [1, %d].', trial, nTrials);
                end
                evts(i).Ts = evts(i).Ts - trialStartTimes(trial);
            end
        end

        % Gaussian kernel smoothing of data across time
        function Xs = smoother(X,kern,causal,gpu)
        %% SMOOTHER Smooth the data using a Gaussian kernel
        % This function smooths the data using a Gaussian kernel. The
        % parameters for the smoothing are stored in the properties of the object.
        %
        % Input:
        %   X - the data to be smoothed
        %   kern - the standard deviation of the Gaussian kernel in samples
        %   causal - whether to use causal or acausal smoothing
        %   binsize - the width of the bins in samples
        %   gpu - whether to use the GPU for the computation
        %
        % Returns:
        %   Xs - the smoothed data
        % Based on @ 2009 Byron Yu -- byronyu@stanford.edu

        if (kern == 0)
            Xs = X;
            return;
        end

        % Filter half length
        % Go 3 standard deviations out
        fltHL = ceil(3 * kern );

        % Length of flt is 2*fltHL + 1
        flt = normpdf(-fltHL : 1 : fltHL, 0, kern);

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

        function Z = mergestructs(x, y)
        % MERGESTRUCTS    Merges two structures.
        %
        %   Merges two structures X and Y. If a field is present in both
        %   structures, the value from Y is kept.
        %
        %   Z = MERGESTRUCTS(X, Y) returns a new structure Z that is a
        %   combination of X and Y. If X and Y have fields with the same
        %   name, the value from Y is used in Z.
            % Get the field names of the two structures
            Xnames = fieldnames(x);
            Ynames = fieldnames(y);

            % Find the fields of X that are also present in Y
            XinY = ismember(Xnames, Ynames);

            % Convert the structures to cell arrays
            Xvals = struct2cell(x);
            Yvals = struct2cell(y);

            % Create a new structure Z by merging the cell arrays
            Z = cell2struct(...
                [Xvals(~XinY); Yvals], ...
                [Xnames(~XinY); Ynames]);
        end
        
        % Returns a matrix of explained variances
        function R2 = explainedVar(varargin)
            % EXPLAINEDVAR returns a matrix of explained variances of dimension NxN
            % where N is the number of inputs. Inputs must have the same size.
            %
            % R2 = EXPLAINEDVAR(X, Y, Z, ...)
            %
            % Inputs:
            %   X, Y, Z, ... - matrices of same size
            %
            % Outputs:
            %   R2 - a matrix of explained variances of size NxN
            %
            % The R2 matrix is symmetric and R2(i,j) is the explained variance of
            % the data in X_i with respect to X_j. If R2(i,j) > 1, it is set to 1
            % and R2(j,i) is the actual explained variance of X_j with respect to
            % X_i.

            % Check if first input is transposed
            % If it is, transpose it and change the size
            sz1 = size(varargin{1});

            % Check if first dimension is the number of samples
            if sz1(1) < sz1(2)
                varargin{1} = varargin{1}';
                sz1 = fliplr(sz1);
            end %fi

            % Check if all inputs have the same size
            % If not, throw an error
            for ii = 2:nargin
                if ~all(diag(...
                        sz1 == size(varargin{ii})' | fliplr(sz1) == size(varargin{ii})' ...\
                        ))
                    warning('Input sizes must be consistent');
                    R2 = nan(2);
                    return;
                elseif all(diag(fliplr(sz1) == size(varargin{ii})'))
                    varargin{ii} = varargin{ii}';
                end
            end %ii

            % Initialize R2 matrix
            % This matrix is symmetric and R2(i,j) is the explained variance of
            % the data in X_i with respect to X_j. If R2(i,j) > 1, it is set to 1
            % and R2(j,i) is the actual explained variance of X_j with respect to
            % X_i
            R2 = zeros(nargin);

            % Compute explained variances
            % Loop over all pairs of inputs
            for ii = 1:nargin-1
                X = varargin{ii};

                % Loop over all remaining inputs
                for jj = ii+1:nargin
                    Y = varargin{jj};

                    % Compute the covariance matrix of X and Y
                    S = cov(X);
                    Srec = cov(Y);

                    % Compute the explained variance of X with respect to Y
                    % This is the trace of the covariance matrix of Y divided by
                    % the trace of the covariance matrix of X
                    R2(ii,jj) = trace(Srec)/trace(S);

                    % If the explained variance is greater than 1, set it to 1
                    % and set the explained variance of Y with respect to X to the
                    % actual value
                    if R2(ii,jj) > 1
                        R2(jj,ii) = trace(S)/trace(Srec);
                        R2(ii,jj) = 1;
                    else
                        R2(jj,ii) = 1;
                    end%fi
                end% jj
            end%ii

        end %explainedVar

        function value = calledByBase()
            % CALLEDBYBASE Returns true if current context is two level below base
            %
            % This function checks how far up the call stack base is. If
            % base is two calls up, it means that the current function was
            % called by base and it returns true. Otherwise, it returns false.
            %
            % See also: dbstack
            stack = dbstack('-completenames');
            value = numel(stack) < 3;
        end
        
        % Bins spike timestamps aligned to triggers and returns a cell per trial wrapping a sparse matrix.
        function D = makeSparseTrain(spikeTimes, unitIDs, triggers, window, binL)
            % binSpikesToSparse - Bins spike timestamps aligned to triggers and returns a sparse matrix.
            %
            % Syntax:
            %   spkMat = binSpikesToSparse(spikeTimes, unitIDs, triggers, window)
            %   spkMat = binSpikesToSparse(spikeTimes, unitIDs, triggers, window, binL)
            %
            % Inputs:
            %   spikeTimes - Vector of spike timestamps (in seconds).
            %   unitIDs    - Vector of unit identifiers corresponding to each spike.
            %   triggers   - Vector of trigger timestamps (in seconds) to which activity is aligned.
            %   window     - Two-element vector [tStart, tEnd] defining the time window (in seconds)
            %                relative to each trigger.
            %   binL       - (Optional) Bin length in seconds (default: 0.001 sec, i.e. 1 ms).
            %
            % Output:
            %   spkMat - Sparse matrix with nUnits rows and (nTrials*nBins) columns. Each row corresponds
            %            to one unique unit and for each trigger the binned spike count in the specified
            %            window is concatenated horizontally.
            %
            % Example:
            %   % Generate example data:
            %   spikeTimes = rand(1000,1)*10;    % 1000 spikes over 10 seconds
            %   unitIDs    = randi(5, 1000, 1);    % spikes from 5 units
            %   triggers   = 1:0.5:9.5;            % triggers every 0.5 sec from 1 to 9.5 sec
            %   window     = [-0.1 0.3];           % analyze from 100 ms before to 300 ms after trigger
            %   spkMat = binSpikesToSparse(spikeTimes, unitIDs, triggers, window);
            %
            % See also: histcounts, sparse

            if nargin < 5 || isempty(binL)
                binL = 0.001; % default bin length: 1 ms
            end

            % Determine unique units and basic dimensions
            uniqueUnits = unique(unitIDs);
            nUnits  = numel(uniqueUnits);
            nTrials = numel(triggers);
            nBins   = round((window(2) - window(1)) / binL); % number of bins per trial


            % Define bin edges for histograms
            edges = window(1):binL:window(2);

            % Preallocate final cell array
            D = cell(nTrials,1);

            % Loop over trials and units to bin spikes
            for trial = 1:nTrials
                % Get the current trigger time
                tTrigger = triggers(trial);

                % For efficiency, find spikes that occur roughly in the whole window (plus margin)
                % This can reduce searching time if spikeTimes is large.
                trialStart = tTrigger + window(1);
                trialEnd   = tTrigger + window(2);
                trialSpkIdx = spikeTimes >= trialStart & spikeTimes < trialEnd;

                % If no spikes fall in this trial window, skip to next trial.
                if ~any(trialSpkIdx)
                    spkMat = zeros(nUnits, nBins);
                    D{trial} = sparse(spkMat);
                    continue;
                end

                % Get the subset of spikes for this trial
                relSpikeTimes = spikeTimes(trialSpkIdx) - tTrigger;
                relUnitIDs    = unitIDs(trialSpkIdx);

                % Preallocate sparse matrix:
                % total columns = nTrials * nBins, rows = nUnits.
                spkMat = zeros(nUnits, nBins);


                % Process each unit separately
                for u = 1:nUnits
                    % Logical index for spikes from the current unit in this trial
                    idx = (relUnitIDs == uniqueUnits(u));
                    if any(idx)
                        % Compute the histogram for this unit's spikes
                        counts = histcounts(relSpikeTimes(idx), edges);
                        % Place counts into the corresponding columns for this trial.
                        spkMat(u, :) = counts;
                    end
                end
                D{trial} = sparse(spkMat);
            end
        end

        function obj = loadobj(obj)
            fprintf(1,'Loading %s.%s: loading data ',obj.Animal,obj.Session);
            obj.E_  = cell(obj.nTrial,1 + obj.nArea);
            if isempty(obj.W_),obj.W_ = cell(1,1 + obj.nArea);end
            if isempty(obj.Winv_),obj.Winv_ = cell(1,1 + obj.nArea);end
            fprintf(1,repmat('\b',1,13));
            fprintf(1,'preprocessing');
            performPrePro(obj);
            
            fprintf(1,repmat('\b',1,13));
            fprintf(1,'projecting');
            currentAmask = unique(obj.Area(obj.aMask));
            for ar = obj.UArea(:)'
                obj.aMask = ar;
                E = ...
                    embedding.(obj.currentEmbeddingMethod).project(obj.S,obj.W);
                % standardize data
                E_s = cellfun(@(x)(x - mean(x,2)./std(x,[],2)),...
                    E,'UniformOutput',false);

                fprintf(1,repmat('\b',1,10));
                fprintf(1,'smoothing');
                amask = ismember(obj.UArea,obj.aMask_);
                cmask = obj.cMask;
                % Smooth the data using the smoother
                obj.E_(cmask,amask) = cellfun(@(x)NeuralEmbedding.smoother(x,...
                    obj.postkern,obj.causalSmoothing,obj.useGpu),...
                    E_s,'UniformOutput',false);
            end
            obj.aMask = currentAmask;
            fprintf(1,repmat('\b',1,9));
            fprintf(1,'done!');
            fprintf(1,'\n');

        end
    end
end

