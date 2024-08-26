function [Dout,TrialTime,nUnits,nTrial,Condition,Area] = Double(Din,opts)
% function [Dout,TrialTime,nUnits,nTrial,Condition,Area] = Double(Din,opts)
%
% Convert a double array to the standard format for the NeuralEmbedding
% class.
%
% Parameters:
%   Din (double): Input data in the format nUnits x TrialL x nTrials
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

