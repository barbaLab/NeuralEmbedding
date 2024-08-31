function [E,ProjMatrix,VarExplained] = PCA(D,pars)
[nT,nA] = size(D);
E = cell(nT,nA);
ProjMatrix = cell(1,nA);
VarExplained = cell(1,nA);

for am = 1:size(D,2)
    D_ = repmat(struct(),size(D,1),1);
    for id=1:size(D,1)
        D_(id).data = full(D{id,am});
    end
    [E(:,am),ProjMatrix(:,am),VarExplained(:,am)] = embedding.PCA.reduce(D_,pars.numPC);
end
end