function [E,ProjMatrix,ProjMatrixInv,VarExplained] = PCA(D,pars,W)

if nargin < 3
    [W, VarExplained] = deal([]);
end

[nT,nA] = size(D);

D_ = repmat(struct(),size(D,1),1);
for id=1:size(D,1)
    D_(id).data = full(D{id});
end
if pars.projectOnly
    E = embedding.PCA.project(D,W);
    Winv = W{1}';
    data = [D{:}];
    VarExplained = NeuralEmbedding.explainedVar(Winv * [E{:}], data);
    VarExplained = {VarExplained(2,1)};

    ProjMatrix = W;
    ProjMatrixInv = {Winv};
else
    [E,ProjMatrix,VarExplained] = embedding.PCA.reduce(D_,pars.numPC);
    E = cellfun(@(e)e(1:pars.numPC,:),E,'UniformOutput',false);
    ProjMatrixInv = cellfun(@(w)w(:,1:pars.numPC),ProjMatrix,'UniformOutput',false);
    ProjMatrix = cellfun(@(w)w(:,1:pars.numPC)',ProjMatrix,'UniformOutput',false);
end



end