% [Dout,TrialTime,nUnits,nTrial,Condition,Area] = Struct(Din,opts)
%
% Converts a data structure Din into the standard format for the
% NeuralEmbedding class.
%
% Parameters:
%   Din (struct array): Input data structure with fields
%       data (double): neural data, nUnits x TrialL x nTrials
%       time (double): time vector, TrialL x 1
%       condition (string): condition labels, 1 x nTrials
%       area (string): area labels, 1 x nUnits
%   opts (struct): Options structure with the following fields
%       time (double): Time vector ( TrialL x 1 )
%       condition (string): Condition labels ( 1 x nTrials )
%       area (string): Area labels ( 1 x nUnits )
%
% Returns:
%   Dout (cell): Standard format data, one cell per trial
%   TrialTime (double): Time vector ( TrialL x 1 )
%   nUnits (int): Number of units
%   nTrial (int): Number of trials
%   Condition (string): Condition labels ( 1 x nTrials )
%   Area (string): Area labels ( 1 x nUnits )
function [Dout,TrialTime,nUnits,nTrial,Condition,Area] = Struct(Din,opts)
    % Get the default values of the options
    optsDefault = structfun(@isempty,opts,'UniformOutput',false);

    %% Get the number of trials
    nTrial = length(Din);

    %% Get the number of units
    nUnits = arrayfun(@(idx) size(Din(idx).data,1),...
        1:nTrial,'UniformOutput',true);
    nUnits = unique(nUnits);

    %% Create the output data structure
    Dout = arrayfun(@(idx) Din(idx).data,...
        1:nTrial,'UniformOutput',false);
    Dout = Dout(:);

    %% Get the trial lengths
    TrialL = arrayfun(@(idx) size(Din(idx).data,2),...
        1:nTrial,'UniformOutput',true);
    TrialL = unique(TrialL);

    %% Get the time vector
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

    %% Get the condition labels
    if optsDefault.condition
        cName = datareader.is.Struct(Din,"condition");
        Condition = arrayfun(@(idx) string(Din(idx).(cName)),...
            1:nTrial,'UniformOutput',true);
    else
        Condition = opts.condition;
    end

    %% Get the area labels
    if optsDefault.area
        aName = datareader.is.Struct(Din,"area");
        Area = arrayfun(@(idx) string(Din(idx).(aName)(:)),...
            1:nTrial,'UniformOutput',false);
        Area = unique(cat(2,Area{:})','rows');
        if size(Area,1)~=1
            error('Mismatch detected in Area labels throughout trials.');
        end
    else
        Area = opts.area;
    end
end

