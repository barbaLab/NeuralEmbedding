function [E,ProjMatrix,VarExplained]= GPFA(D,pars)

seqTest = pars.seqTest;
nTrial = length(D);
if pars.doTest
    if isempty(seqTest)
        % if test sequence is not defined, randomly extract 5% of all trials to
        % be test trials
        seqTest = false(nTrial,1);
        seqTest(randperm(nTrial,floor(nTrial * .05))) = true;
    end
else
    seqTest = false(nTrial,1);
end
pars = rmfield(pars,"seqTest");

D_ = repmat(struct(),numel(D),1);
for id=1:numel(D)
    D_(id).y = full(D{id});
    D_(id).trialId = id;
end
[D_.T] = deal(pars.TrialL{:});
otherArgs = [fieldnames(pars) struct2cell(pars)]';
[E,ProjMatrix,VarExplained] = embedding.GPFA.reduce(D_(~seqTest),D_(seqTest),...
    'xDim',pars.numPC,'verbose',false,'binWidth',pars.subsampling,...
    otherArgs{:});
    
end