function Dout = Double(Din,opts)
% Double checks if the input data is in the standard double format for
% the NeuralEmbedding class. If no options are provided, only checks if
% the input data is numeric.
%
% Parameters:
%   Din (double): Input data in numeric form nUnits x Time x nTrials.
%   opts (struct): Options structure with the following fields
%       time (double): Time vector ( Time x 1 )
%       area (string): Area labels ( 1 x nUnits )
%       condition (string): Condition labels ( 1 x nTrials )
%
% Returns:
%   Dout (logical): True if the input data matches the standard format
%       for the NeuralEmbedding class, false otherwise.
%

% Checks if the input data is numeric
if nargin == 1
    Dout = isnumeric(Din);
    if Dout && size(Din,3) == 1
        warning(sprintf("Single trial detected.\nInput data, when in numeric form, has the form nUnits X Time X nTrials."))
    end
    return;
    
% Checks if the input data matches the options structure
elseif nargin == 2
    Dout = datareader.is.Double(Din);

    if Dout
        ValidateArgs(Din,opts);
    end
    return;
end
end

function ValidateArgs(Din,opts)
% ValidateArgs checks if the options structure matches the dimensions
% of the input data.
%
% Parameters:
%   Din (double): Input data in numeric form nUnits x Time x nTrials.
%   opts (struct): Options structure with the following fields
%       time (double): Time vector ( Time x 1 )
%       area (string): Area labels ( 1 x nUnits )
%       condition (string): Condition labels ( 1 x nTrials )
%

[nUnits,Time,nTrials] = size(Din);

% Checks if the time vector matches the input data second dimension
if ~optsDefault.time
    assert(length(opts.time)==Time,...
        'Provided time and input data second dimension mismatch.');
else
    % If no time vector is provided, it is assumed to be linearly spaced
    ...
end

% Checks if the area labels match the input data first dimension
if ~optsDefault.area
    assert(length(opts.area)==nUnits,...
        'Provided area label and input data first dimension mismatch.');
else
    % If no area labels are provided, they are assumed to be string numbers
    error(sprintf('Area labels not provided.\nThey are required for input data in numeric form [string array, 1xnUnits].'));
end

% Checks if the condition labels match the input data third dimension
if ~optsDefault.condition
    assert(length(opts.condition)==nTrials,...
        'Provided conidtion labels and input data third dimension mismatch.');
else
    % If no condition labels are provided, they are assumed to be string numbers
    error(sprintf('Condition labels not provided.\nThey are required for input data in numeric form [string array, 1xnTrials'));
end

end

