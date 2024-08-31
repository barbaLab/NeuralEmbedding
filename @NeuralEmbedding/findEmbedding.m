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

flag = true;

switch deblank(type)
    case {'SmoothPCA','PCA','pca'}
        type = "PCA";
        parNames = ["numPC"];
        pars = obj.assignEPars(parNames,type);
        try
            [E,obj.ProjMatrix,obj.VarExplained] = ...
                embedding.PCA(obj.S,pars);
        catch er
            flag = false;
            return;
        end

    case {'GPFA','gpfa'}
        type = "GPFA";
        parNames = ["subsampling","numPC","TrialL"];
        pars = obj.assignEPars(parNames,type);
        try
        % GPFA does not need presmoothing. It gets as input P instead of S
            [E, obj.ProjMatrix, obj.VarExplained] = ...
                embedding.GPFA(obj.P, pars);                                             
        catch er
            flag = false;
            return;
        end
    case {'CCA','cca'}
        type = "CCA";
        parNames = ["numPC","nArea","nTrial","TrialL"];
        pars = obj.assignEPars(parNames,type);
        try
            D = cell(obj.nTrial,obj.nArea);
            a = 1;
            for aa = obj.UArea(1:end-1)
                obj.aMask = aa;
                D(:,a) = obj.S;
                a = a + 1;
            end
            obj.aMask = obj.UArea(1:end-1);
            [E,obj.ProjMatrix,obj.VarExplained] = ...
                embedding.CCA(D,pars);
        catch er
            flag = false;
            return;
        end
    case {'umap','UMAP'}
        type = "UMAP";
        fprintf("%s support is WIP. Stay tuned!%s",type,newline);
        % [E,obj.ProjMatrix,obj.VarExplained] = ...
        %     smoothUMAP(D,pars_);

    case {'t-SNE','tsne','tSNE','t-sne'}
        type = "tSNE";
        fprintf("%s support is WIP. Stay tuned!%s",type,newline);
        % [E,obj.ProjMatrix,obj.VarExplained] = ...
        %     smoothTSNE(D,pars_);

    case {'I','identity','noproject','noProject'}
        type = "I";
        [E,obj.ProjMatrix,obj.VarExplained] = ...
            deal(obj.S,...
            eye(obj.nUnits),...
            NeuralEmbedding.explainedVar([obj.S{:}], [obj.S{:}]));

    otherwise
        [E,obj.ProjMatrix,obj.VarExplained] = deal(nan);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Common operations on embedded data                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if obj.Reproject
    flag = obj.useTMask;
    obj.useTMask = false;
    E = embedding.(type).project(obj.S, obj.ProjMatrix);
    obj.useTMask = flag;
end

% smooth data
amask = ismember(obj.UArea,obj.aMask_);
cmask = obj.cMask;
obj.E_(cmask,amask) = cellfun(@(x)NeuralEmbedding.smoother(x,...
            obj.prekern,obj.causalSmoothing,obj.subsampling,obj.useGpu),...
            E,'UniformOutput',false);

end
