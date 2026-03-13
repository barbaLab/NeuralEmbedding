function [E,W,VarExplained] = findEmbedding(obj,type, projectOnly)
% This function computes the embedding of the data using the specified
% algorithm. Supported algorithms are PCA, GPFA, CCA, UMAP, t-SNE, and
% Identity. If the algorithm is not recognized, it throws an error.
%
% The function takes as input the data to be embedded, the type of
% embedding to be used, and the parameters for the embedding algorithm.
%
% The output is the embedded data, the projection matrix, and the
% explained variance ratio.

% flag keeps track of whether the algorithm succeeded or not
flag = true;

if nargin<3
    projectOnly = false;
end

% switch through the different algorithms
switch deblank(type)
    case {'SmoothPCA','PCA','pca'}
        type = "PCA";
        % Get the names of the parameters for this algorithm
        parNames = ["numPC"];
        % Get the parameters for this algorithm
        pars = obj.assignEPars(parNames,type);
        pars.numPC = min(sum(obj.aMask),30);
        pars.projectOnly = projectOnly;
        % try
            % Compute the embedding using PCA
            [E,W,Winv,VarExplained] = ...
                embedding.PCA(obj.S,pars,obj.W,{obj.VarExplained});
        % catch er
        %     % If the algorithm fails, set flag to false and rethrow the error
        %     flag = false;
        %     rethrow(er);
        %     return;
        % end

    case {'GPFA','gpfa'}
        type = "GPFA";
        % Get the names of the parameters for this algorithm
        parNames = ["subsampling","numPC","TrialL"];
        % Get the parameters for this algorithm
        pars = obj.assignEPars(parNames,type);
        pars.numPC = obj.NumPC;
        pars.projectOnly = projectOnly;
        try
            % GPFA does not need presmoothing. It gets as input P instead of S
            [E, W, VarExplained] = ...
                embedding.GPFA(obj.P, pars, obj.W, {obj.VarExplained});       
            Winv = {pinv(W{1})};
        catch er
            % If the algorithm fails, set flag to false and rethrow the error
            flag = false;
            rethrow(er);
            return;
        end
    case {'CCA','cca'}
        type = "CCA";
        % Get the names of the parameters for this algorithm
        parNames = ["numPC","nArea","nTrial","TrialL"];
        % Get the parameters for this algorithm
        pars = obj.assignEPars(parNames,type);
        try
            % Initialize the data for CCA
            D = cell(obj.nTrial,obj.nArea);
            for aa = 1:obj.nArea
                % Set the area mask to the current area
                obj.aMask = obj.UArea(aa);
                % Get the data for the current area
                D(:,aa) = obj.S;
                % Increment the area counter
            end
            % Set the area mask to all areas
            obj.aMask = obj.UArea(1:end-1);
            % Compute the CCA
            [E,W,CanonCorrelation] = ...
                embedding.CCA(D,pars);
            Winv = cellfun(@pinv,W,'UniformOutput',false);
        catch er
            % If the algorithm fails, set flag to false and rethrow the error
            flag = false;
            rethrow(er);
            return;
        end

    case {'MCCA','mcca'}
        if isscalar(obj)
            [E,W,VarExplained] = findEmbedding(obj,'CCA');
            return;
        else
        type = "MCCA";
        % Get the names of the parameters for this algorithm
        parNames = ["nTrial","TrialL","mcca_k"];
        % Get the parameters for this algorithm
        pars = obj.assignEPars(parNames,type);
        try
            thisNumPc = {obj.NumPC};
            [obj.NumPC] = deal(30);
            % Run MCCA
            [E,W,Winv,Corr]  = embedding.MCCA({obj.E},pars);
            [obj.NumPC] = deal(thisNumPc{:});

            for ss = 1:numel(obj)
                amask = ismember(obj(ss).UArea,obj(ss).aMask_);
                cmask = obj(ss).cMask;
                % Smooth the data using the smoother
                obj(ss).E_(cmask,amask) = E{ss};
                obj(ss).W_(amask) = {W{1} * obj(ss).W_{amask}};
                obj(ss).Winv_(amask) = {obj(ss).Winv_{amask} * Winv{1}};
            end
            return;
        catch er
            flag = false;
            rethrow(er);
            return;
        end
        end
    case {'umap','UMAP'}
        type = "UMAP";
        % UMAP is not yet supported. Print a message and return
        fprintf("%s support is WIP. Stay tuned!%s",type,newline);
        % [E,W,VarExplained] = ...
        %     smoothUMAP(D,pars_);

    case {'t-SNE','tsne','tSNE','t-sne'}
        type = "tSNE";
        % tSNE is not yet supported. Print a message and return
        fprintf("%s support is WIP. Stay tuned!%s",type,newline);
        % [E,W,VarExplained] = ...
        %     smoothTSNE(D,pars_);

    case {'I','identity','noproject','noProject'}
        type = "I";
        % For the identity embedding, just return the original data and
        % matrix
        [E,W,VarExplained] = ...
            deal(obj.S,...
            eye(obj.nUnits),...
            NeuralEmbedding.explainedVar([obj.S{:}], [obj.S{:}]));

    otherwise
        % If the algorithm is not recognized, return NaNs
        [E,W,VarExplained] = deal(nan);
end

obj.currentEmbeddingMethod = type;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Common operations on embedded data                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if obj.Reproject
    flag = obj.useTMask;
    obj.useTMask = false;
    % Reproject the data using the projection matrix
    E = arrayfun(@(a_idx)embedding.(type).project(obj.S(:,a_idx), W(a_idx)),1:size(obj.S,2),'UniformOutput',false);
    E = cat(2,E{:});
    % Reset the mask to its original value
    obj.useTMask = flag;
end

% standardize data
E_s = cellfun(@(x)(x - mean(x,2)./std(x,[],2)),...
            E,'UniformOutput',false);

% smooth data
amask = ismember(obj.UArea,obj.aMask_);
cmask = obj.cMask;
% Smooth the data using the smoother
obj.E_(cmask,amask) = cellfun(@(x)NeuralEmbedding.smoother(x,...
            obj.postkern,obj.causalSmoothing,obj.useGpu),...
            E_s,'UniformOutput',false);
obj.W_(amask) = W;
obj.Winv_(amask) = Winv;
if exist("VarExplained","var")
    obj.VarExplained_(amask) = VarExplained{:};
end
if exist("CanonCorrelation","var")
    obj.CanonCorrelation_(1:obj.nArea-1) = CanonCorrelation;
end

end

