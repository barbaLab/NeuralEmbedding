function flag = findEmbedding(obj,type)
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
        [E,obj.ProjMatrix,obj.VarExplained] = ...
            smoothCCA(D,pars_);
    case {'justCCA','justcca'}
        [E,obj.ProjMatrix,obj.VarExplained] = ...
            smoothjustCCA(D,pars_);
    case {'umap','UMAP'}
        [E,obj.ProjMatrix,obj.VarExplained] = ...
            smoothUMAP(D,pars_);

    case {'t-SNE','tsne','tSNE','t-sne'}
        [E,obj.ProjMatrix,obj.VarExplained] = ...
            smoothTSNE(D,pars_);

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
amask = strcmp(obj.aMask_,obj.UArea);
obj.E_(:,amask) = cellfun(@(x)NeuralEmbedding.smoother(x,...
            obj.prekern,obj.causalSmoothing,obj.subsampling,obj.useGpu),...
            E,'UniformOutput',false);

end