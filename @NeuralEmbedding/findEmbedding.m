function [E,ProjMatrix,VarExplained] = findEmbedding(obj,type)
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

% switch through the different algorithms
switch deblank(type)
    case {'SmoothPCA','PCA','pca'}
        type = "PCA";
        % Get the names of the parameters for this algorithm
        parNames = ["numPC"];
        % Get the parameters for this algorithm
        pars = obj.assignEPars(parNames,type);
        try
            % Compute the embedding using PCA
            [E,obj.ProjMatrix,obj.VarExplained] = ...
                embedding.PCA(obj.S,pars);
        catch er
            % If the algorithm fails, set flag to false and rethrow the error
            flag = false;
            rethrow(er);
            return;
        end

    case {'GPFA','gpfa'}
        type = "GPFA";
        % Get the names of the parameters for this algorithm
        parNames = ["subsampling","numPC","TrialL"];
        % Get the parameters for this algorithm
        pars = obj.assignEPars(parNames,type);
        try
            % GPFA does not need presmoothing. It gets as input P instead of S
            [E, obj.ProjMatrix, obj.VarExplained] = ...
                embedding.GPFA(obj.P, pars);                                             
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
            a = 1;
            for aa = obj.UArea(1:end-1)
                % Set the area mask to the current area
                obj.aMask = aa;
                % Get the data for the current area
                D(:,a) = obj.S;
                % Increment the area counter
                a = a + 1;
            end
            % Set the area mask to all areas
            obj.aMask = obj.UArea(1:end-1);
            % Compute the CCA
            [E,obj.ProjMatrix,obj.VarExplained] = ...
                embedding.CCA(D,pars);
        catch er
            % If the algorithm fails, set flag to false and rethrow the error
            flag = false;
            rethrow(er);
            return;
        end
    case {'umap','UMAP'}
        type = "UMAP";
        % UMAP is not yet supported. Print a message and return
        fprintf("%s support is WIP. Stay tuned!%s",type,newline);
        % [E,obj.ProjMatrix,obj.VarExplained] = ...
        %     smoothUMAP(D,pars_);

    case {'t-SNE','tsne','tSNE','t-sne'}
        type = "tSNE";
        % tSNE is not yet supported. Print a message and return
        fprintf("%s support is WIP. Stay tuned!%s",type,newline);
        % [E,obj.ProjMatrix,obj.VarExplained] = ...
        %     smoothTSNE(D,pars_);

    case {'I','identity','noproject','noProject'}
        type = "I";
        % For the identity embedding, just return the original data and
        % matrix
        [E,obj.ProjMatrix,obj.VarExplained] = ...
            deal(obj.S,...
            eye(obj.nUnits),...
            NeuralEmbedding.explainedVar([obj.S{:}], [obj.S{:}]));

    otherwise
        % If the algorithm is not recognized, return NaNs
        [E,obj.ProjMatrix,obj.VarExplained] = deal(nan);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Common operations on embedded data                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if obj.Reproject
    flag = obj.useTMask;
    obj.useTMask = false;
    % Reproject the data using the projection matrix
    E = arrayfun(@(a_idx)embedding.(type).project(obj.S(:,a_idx), obj.ProjMatrix(:,a_idx)),1:size(obj.S,2),'UniformOutput',false);
    E = cat(2,E{:});
    % E = embedding.(type).project(obj.S, obj.ProjMatrix);
    % Reset the mask to its original value
    obj.useTMask = flag;
end

% smooth data
amask = ismember(obj.UArea,obj.aMask_);
cmask = obj.cMask;
% Smooth the data using the smoother
obj.E_(cmask,amask) = cellfun(@(x)NeuralEmbedding.smoother(x,...
            obj.prekern,obj.causalSmoothing,obj.subsampling,obj.useGpu),...
            E,'UniformOutput',false);

end

