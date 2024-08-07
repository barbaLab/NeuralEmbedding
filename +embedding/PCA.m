function [E,ProjMatrix,VarExplained] = PCA(D,pars)

D_ = repmat(struct(),numel(D),1);
for id=1:numel(D)
   D_(id).data = full(D{id});
end

[E,ProjMatrix,VarExplained] = embedding.PCA.reduce(D_,pars.numPC);
end