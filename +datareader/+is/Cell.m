% @NeuralEmbedding\+datareader\+is\Cell.m
%
% Checks if the input data is in cell format
%
% Parameters:
%   Din (cell): Input data in cell format
%   opts (struct): Options structure with the following fields
%       time (double): Time vector ( TrialL x 1 )
%       condition (string): Condition labels ( 1 x nTrials )
%       area (string): Area labels ( 1 x nUnits )
%
% Returns:
%   Dout (logical): True if the input data is in the correct format, false
%   otherwise
function Dout = Cell(Din,opts)
    % Check if the input data is a cell array
    if nargin == 1
        % If only the data is provided, check that it is a cell array and
        % all the elements are numeric
        Dout = iscell(Din) && all(cellfun(@isnumeric,Din));
        % If the data is a single trial, issue a warning
        if length(Din) == 1
            warning(sprintf("Single trial detected.\nInput data, when in numeric form, has the form nUnits x Time x nTrials."))
        end
        return;
    elseif nargin == 2

        % Check if the input data is in the correct format
        Dout = datareader.is.Cell(Din);
        if Dout
            % Validate the options and the input data
            dishomogeneous = ValidateArgs(Din,opts);

            if ~dishomogeneous
                % Convert the cell array to a numeric array
                Din_ = cat(3,Din{:});

                opts.time = opts.time{1};
                % Check if the numeric array is in the correct format
                Dout = NeuralEmbedding.datareader.is.Double(Din_,opts);
            end
        end
        
        return;
    end

end

% @NeuralEmbedding\+datareader\+is\Cell.m
%
% Validates the input options and the input data
%
% Parameters:
%   Din (cell): Input data in cell format
%   opts (struct): Options structure with the following fields
%       time (double): Time vector ( TrialL x 1 )
%       condition (string): Condition labels ( 1 x nTrials )
%       area (string): Area labels ( 1 x nUnits )
%
% Returns:
%   dishomogeneous (logical): True if the input data is homogeneous, false
%   otherwise
function dishomogeneous = ValidateArgs(Din,opts)
    % Get the default values of the options
    optsDefault = structfun(@isempty,opts,'UniformOutput',false);

    % Get the number of trials
    nTrials = length(Din);
    % Get the number of units
    nUnits  = unique(cellfun(@(x)size(x,1),Din));
    % Get the trial lengths
    Time    = cellfun(@(x)size(x,2),Din);

    % Check if the trial lengths are homogeneous
    % If not, the time vector should be a cell array with the same
    % length as the number of trials
    if length(unique(Time)) ~= 1
        dishomogeneous = true;

        % Check if the time vector is a cell array and has the same
        % length as the number of trials
        if ~optsDefault.time
            assert(iscell(opts.time),...
                'Cell input detected. Please provide a cell array of time vectors, one per trial.');

            assert(length(Din) == length(Time),...
                'Provided time and input data dimension mismatch. Please provide a cell array of time vectors and neural data, one per trial.');

            % Check if each time vector has the same length as the trial
            % lengths
            assert(all(cellfun(@(x)size(x,2),Din) == Time),...
                'Provided time and input data dimension mismatch. Please provide a cell array of time vectors matching trial lenghts.');
        else
            error('Cell input detected. Please provide a cell array of time vectors, one per trial.');
        end
    else
        dishomogeneous = false;
    end

    % Check if the area labels are provided
    if ~optsDefault.area
        assert(length(opts.area)==nUnits,...
            'Provided area label and input data first dimension mismatch.');
    else
        error(sprintf('Area labels not provided.\nThey are required for input data in numeric form [string array, 1xnUnits].'));
    end
    % Check if the condition labels are provided
    if ~optsDefault.condition
        assert(length(opts.condition)==nTrials,...
            'Provided conidtion labels and input data third dimension mismatch.');
    else
        error(sprintf('Condition labels not provided.\nThey are required for input data in numeric form [string array, 1xnTrials.'));
    end
end


