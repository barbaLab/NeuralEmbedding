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
%   Dishomogeneous (logical): Whether the trial lengths are homogeneous or not
function [Dout,TrialTime,nUnits,nTrial,Condition,Area,Dishomogeneous] = Struct(Din,opts)
    %% Get the default values of the options
    optsDefault = structfun(@isempty,opts,'UniformOutput',false);

    %% Validate the input data structure
    Din = datareader.is.Struct(Din,opts,true);

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

    %% Check if the trial lengths are homogeneous
    Dishomogeneous = numel(TrialL) > 1;

    %% Get the time vector
    if optsDefault.time
        TrialTime = {Din.time};
    elseif Dishomogeneous
        TrialTime = opts.time;
    else
        TrialTime = repmat(opts.time(1),nTrial,1);
    end
    % ensure proper orientation
    TrialTime = cellfun(@(t)t(:),TrialTime(:),'UniformOutput',false);

    %% Get the condition labels
    if optsDefault.condition
        Condition = cat(1,Din.condition);
    else
        Condition = opts.condition(:);
    end

    %% Get the area labels
    if optsDefault.area
        Area = Din(1).area(:);
    else
        Area = opts.area(:);
    end
end

