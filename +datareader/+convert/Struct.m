function [Dout,TrialTime,nUnits,nTrial,Condition,Area] = Struct(Din,opts)
optsDefault = structfun(@isempty,opts,'UniformOutput',false);

%% nTrial
nTrial = length(Din);

%% nUnits
nUnits = arrayfun(@(idx) size(Din(idx).data,1),...
    1:nTrial,'UniformOutput',true);
nUnits = unique(nUnits);

%% Dout
Dout = arrayfun(@(idx) Din(idx).data,...
    1:nTrial,'UniformOutput',false);
Dout = Dout(:);

%% TrialL
TrialL = arrayfun(@(idx) size(Din(idx).data,2),...
    1:nTrial,'UniformOutput',true);
TrialL = unique(TrialL);

%% TrialTime
if optsDefault.time
    tName = datareader.is.Struct(Din,"time");
    TrialTime = arrayfun(@(idx) Din(idx).(tName),...
        1:nTrial,'UniformOutput',false);
    TrialTime = cat(1,TrialTime{:});
    if any(...
            diff(TrialTime),...
            "all")
        error('Mismatch detected in times throughout trials.');
    elseif size(TrialTime,2) ~= TrialL
        error('Mismatch detected between times trial samples.');
    else
        TrialTime = TrialTime(1,:);
    end
else
    TrialTime = opts.time;
end
%% Condition
if optsDefault.condition
cName = datareader.is.Struct(Din,"condition");
Condition = arrayfun(@(idx) string(Din(idx).(cName)),...
    1:nTrial,'UniformOutput',true);
else
    Condition = opts.condition;
end
%% Area
if optsDefault.area
aName = datareader.is.Struct(Din,"area");
Area = arrayfun(@(idx) string(Din(idx).(aName)(:)),...
    1:nTrial,'UniformOutput',false);
Area = unique(cat(2,Area{:})','rows');
if size(Area,1)~=1
    error('Mismatch detected in Area labels throughout trials.');
end
else
    TrialTime = opts.area;
end

end

