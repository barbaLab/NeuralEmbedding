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
        tMask       cell                                                 % Time vector mask, ie what time bins to use to embed data
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
        tMask_      cell                                                    % original size mask
        tMaskSub    cell                                                    % stored subsampled mask for speed 

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
        homogeneous logical = false                                    % flag for trial homogenuity. If all trials all equally long, this is 0
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
            % Pre-allocate the embedded data
            obj.E_  = cell(nTrial,1 + obj.nArea);
            % Store the original trial time
            obj.TrialTime_ = TrialTime(:);            
            % Store the number of units and trials
            obj.nUnits = nUnits;
            obj.nTrial = nTrial;
            % Store the condition labels
            obj.Conditions = Condition;
            % Store the area labels
            obj.Area = Area;

            % Store the original time mask
            obj.tMask = cellfun(@(t) true(length(t),1),obj.TrialTime_,...
                'UniformOutput',false);
            
            % Remove inactive neurons
            obj.removeInactiveNeurons();
            
            performPrePro(obj);
        end

        function performPrePro(obj)
            % Bin the data
            obj.binData();
            % Smooth the data
            obj.smoothData();
            % Z-score the data
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

            value = cell(obj.nTrial,size(amask,2));
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
        end

        function value = get.M(obj)
            value = struct2table(obj.M_);
        end
        
        % Returns unique experimental conditions.
        function value = get.UConditions(obj)
            value = [string(unique(obj.Conditions(:))); "AllConditions"];
        end

        % Returns unique experimental conditions.
        function value = get.UArea(obj)
            value = [string(unique(obj.Area(:))); "AllNeurons"];
        end
       
        % Returns updated TrialTime wrt subsampling and tMask.
        function value = get.TrialTime(obj)
            % idx = find(obj.tMask);
            value = cellfun(@(t)t(1:obj.subsampling:end),...
                obj.TrialTime_,...
                'UniformOutput',false);
            value = cellfun(@(t,tm)t(tm),...
                value(:),obj.tMask,...
                'UniformOutput',false);

        end

        % Returns up to date TrialL wrt subsampling.
        function value = get.TrialL(obj)
            value = cellfun(@(tmsub)floor(sum(tmsub)),...
               obj.tMaskSub,'UniformOutput',false);
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
                obj.tMask_ = repmat(val(1),obj.nTrial,1);

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
                blk = arrayfun(@(t,tr)...
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
                if strcmp(str,"AllNeurons")
                    value = value | 1;
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
            if isstring(val) &&...
                    any(ismember(obj.UArea,val))
                % If the input string matches one of the areas, set the area
                % mask to that area
                obj.aMask_ = val;
            elseif isstring(val) &&...
                    (val == "" || strcmpi(val,"none") || strcmpi(val,"all"))
                % If the input string is empty or matches "none" or "all",
                % set the area mask to all neurons
                obj.aMask_ = "AllNeurons";
            else
                error('Input is either not strinf or does not match areas provided during initialization.')
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
            %   This function loads default parameters for the neural
            %   embedding algorithm. The parameters loaded here are
            %   used if the user does not specify them during
            %   initialization.
           
            % number of principal components to keep
            pars_.numPC = 3;

            % split units
            pars_.splitUnits = [0 size(D(1).data,1)];
            % time vector
            obj.t = obj.TrialTime;

            % sequence test
            pars_.seqTest = false(1,numel(D));

            % end leg range
            pars_.endLeg_range = [1:250 size(D(1).data,2)-250:size(D(1).data,2)];
            % interest range
            pars_.interest_range = (-250:250)+0.5*size(D(1).data,2);
            % cca reference signal
            pars_.ccaRefSig = [];

            % supervise by conditions
            pars_.SuperviseByConditions = false;

            % reference data
            pars_.D_ref = {nan};

            % number of reference data sets
            N_Ref = length(pars_.D_ref);
            % number of areas
            N_Areas = length(pars_.splitUnits)-1;
            % if there are more areas than reference data sets, make
            % sure there are enough reference data sets
            if N_Ref ~= N_Areas
                [pars_.D_ref{N_Ref+1:N_Areas}] = deal(nan);
            end

        end
    
        %% REMOVEINACTIVENEURONS Remove inactive neurons from the data
        % and replaces them with random spikes
        %
        % This function removes neurons with spike rates below a threshold
        % and replaces them with random spikes.
        %
        % Parameters:
        %   obj - the NeuralEmbedding object
        %
        % Returns:
        %   nothing
        function removeInactiveNeurons(obj)
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
            n = cellfun(@(tt)ceil(tt(end) - tt(1) ...
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
            %% BINDATA Bin the data using a block diagonal matrix multiplication.
            %
            % This function multiplies the data with a sparse matrix that
            % represents the binning operation. The matrix is a block diagonal
            % matrix where each block is a matrix of ones with size equal to
            % the bin width. The result is a new set of data with the same
            % number of trials but with the number of time points reduced by
            % the bin width. The subsampling property is then updated to
            % reflect the new bin width.
            %
            % The tMaskSub property is also updated to reflect the new
            % subsampling. This property contains the indices of the time
            % points that are not masked (i.e. not NaN).

            T     = cellfun(@length,obj.TrialTime);
            Tdown = floor(T./obj.binWidth);
            Trest = T - Tdown*obj.binWidth;

            % Binning as block diagonal matrix multiplication
            blk = arrayfun(@(t,tr)...
                [repmat({ones(obj.binWidth,1)},t,1);zeros(tr,1)],...
                Tdown,Trest,'UniformOutput',false);

            A = cellfun(@(thisblk)blkdiag(thisblk{:}),blk,...
                'UniformOutput',false);

            obj.P_ = cellfun(@(x,a) x * sparse(a),...
                obj.P_,A(:),...
                'UniformOutput',false);
            obj.subsampling = obj.binWidth;

            tMaskSub_ = cellfun(@(tm,a)logical(round(tm' * a./obj.subsampling)),...
                obj.tMask_,A,...
                'UniformOutput',false);
             obj.tMaskSub = tMaskSub_(:);
        end

        function smoothData(obj)
            %% SMOOTHEDDATA Smooth the preprocessed data using a Gaussian kernel.
            %
            % The data is expected to be binned. If the useGpu property is true, the
            % smoothing is performed using a GPU. The smoothed data is stored in the
            % S_ property.

            obj.S_ = cellfun(@(x)NeuralEmbedding.smoother(x,...
                obj.prekern,obj.causalSmoothing,obj.subsampling,obj.useGpu),...
                obj.P_,'UniformOutput',false);

        end

        function zscoreData(obj)
            %% ZSCOREDATA Zscore the data using the mean and std calculated from all the trials.
            %
            % The mean and std are calculated from all the trials and stored in the
            % mu and ss properties. The data is then zscored using the following formula:
            % data = (data - mu) ./ ss;
            %
            % The zscored data is stored in the S_ property.

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
           %% INITMSTRUCT Initialize a metrics structure.
            %
            % obj.initMstruct(data,type) initializes a metrics structure with the
            % data and type inputs. The resulting structure is stored in the str
            % output.
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
        function plot3(obj)
            reducedE = cellfun(@(x)[x nan(3,1)], ...
                obj.E, ...
                'UniformOutput',false);
            nT = sum(obj.cMask);
            MaxLines = min(nT,80);
            reducedE = [reducedE{randperm(nT,MaxLines)}];
            plot3(reducedE(1,:),reducedE(2,:),reducedE(3,:))
        end
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

            sz1 = size(varargin{1});

            % Check if first dimension is the number of samples
            if sz1(1) < sz1(2)
                varargin{1} = varargin{1}';
                sz1 = fliplr(sz1);
            end %fi

            % Check if all inputs have the same size
            for ii = 2:nargin
                if ~all(diag(...
                        sz1 == size(varargin{ii})' | fliplr(sz1) == size(varargin{ii})' ...
                        ))
                    error('Input sizes must be consistent');
                elseif all(diag(fliplr(sz1) == size(varargin{ii})'))
                    varargin{ii} = varargin{ii}';
                end
            end %ii

            % Initialize R2 matrix
            R2 = zeros(nargin);

            % Compute explained variances
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

    end
end


