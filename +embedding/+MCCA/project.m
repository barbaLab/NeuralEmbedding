function E = project(D,C)

[nTrial,nArea] = size(D);
E = cell(nTrial,nArea);
alldata = arrayfun(@(Aidx)cat(2,D{:,Aidx}),1:size(D,2),'UniformOutput',false);

E_ = cellfun(@(d,c) c * d,alldata,C,'UniformOutput',false);

% For each condition, store the reduced version of each data vector
for jj = 1:nArea
    index = 0;
    for ii = 1:nTrial
        E{ii,jj} = E_{jj}(index + (1:pars.TrialL),1:dims)';
        index = index + pars.TrialL;
    end
end