function [Dout,TrialTime,nUnits,nTrial,Condition,Area] = Double(Din,opts)

optsDefault = structfun(@isempty,opts,'UniformOutput',false);


% TODO check for fs

[nUnits,TrialL,nTrial] = size(Din);
Dout = arrayfun(@(idx) sparse(Din(:,:,idx)),...
    1:nTrial,'UniformOutput',false);
Dout = Dout(:);
if ~optsDefault.time
    TrialTime = opts.time;
else
    TrialTime   = (1:TrialL)./opts.fs;
end

Condition   = opts.condition;
Area        = opts.area;
end

