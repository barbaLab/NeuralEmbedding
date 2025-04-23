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

        % Metadata
        Meta                    = struct("AnName","","ExpGroup","");
    end

    properties (GetAccess = public,SetAccess = private)
        ProjMatrix                                                          % (Double) Projection Matrix to embedded data
        VarExplained                                                        % (Double) Variance explained by each embedded dimension

        Area    string                                                      % String array with area classification of each unit/channel
        Conditions string                                                   % String array with area classification of each unit/channel

        nUnits double
        nTrial double
        nArea  double
        nCondition double
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
                
                opts.pars           (1,:) struct = struct.empty
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
            %   - binWidth: the width of the bins in samples
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
    end 
    %% Preprocessing methods
    methods (Access = private)
        function setPars(obj,pars)
            %SETPARS takes care of assigning custom parameters
            
            pNames = fieldnames(pars);
            for pp = 1:numel(pNames)
                pn = pNames{pp};
                set(obj,pn,pars.(pn));
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
        %   The bin width is set by the binWidth property. The resulting
        %   subsampling property reflects the new bin width.

            T     = cellfun(@length,obj.TrialTime);
            Tdown = floor(T./obj.binWidth);
            Trest = T - Tdown*obj.binWidth;

            % Binning as block diagonal matrix multiplication
            blk = arrayfun(@(t)...
                repmat({ones(obj.binWidth,1)},t,1),...
                Tdown,'UniformOutput',false);

            A = arrayfun(@(thisblk,tr,t)[blkdiag(thisblk{1}{:});sparse(tr,t)],...
                blk,Trest,Tdown,...
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
                obj.prekern,obj.causalSmoothing,obj.subsampling,obj.useGpu),...
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
            figure;
            plot3(reducedE(1,:),reducedE(2,:),reducedE(3,:))
        end

        function peth(obj)
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
                    error('Input sizes must be consistent');
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

    end
end


