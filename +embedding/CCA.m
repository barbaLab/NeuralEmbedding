function [E,ProjMatrix,VarExplained] = CCA(D,pars)
    D_ = arrayfun(@(aidx)cat(2,D{:,aidx})',1:size(D,2),'UniformOutput',false);

    [E,ProjMatrix,VarExplained{1}] = embedding.CCA.reduce(D_,pars);
end