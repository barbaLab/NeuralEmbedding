function [E,ProjMatrix,ProjMatrixInv,Corr] = MCCA(Dall,pars)
    nSets = numel(Dall);
    minTrials = min(cellfun(@(x)size(x,1),Dall));
    randIdx = arrayfun(@(idx)randi(pars(idx).nTrial,1,minTrials),1:nSets, ...
        'UniformOutput',false);
    Dall_ = cellfun(@(d,idx)d(idx),Dall,randIdx ...
        ,'UniformOutput',false);
    TrialL = arrayfun(@(p,idx)[p.TrialL{idx{1}}],pars,randIdx,'UniformOutput',false);

    [ProjMatrix,Corr] = embedding.MCCA.reduce(Dall_,pars);
    ProjMatrixInv = cellfun(@(w)pinv(w),ProjMatrix,'UniformOutput',false);
    E_ = cellfun(@(d,c) c * cat(2,d{:}),Dall,ProjMatrix,'UniformOutput',false);


    % For each dataset, store the coregistered version
    E = Dall;
    for ii = 1:nSets
        index = 0;
        for jj = 1:pars(ii).nTrial
            E{ii}{jj} = E_{ii}(:,index + (1:pars(ii).TrialL{jj}),:);
            index = index + pars(ii).TrialL{jj};
        end
    end
end