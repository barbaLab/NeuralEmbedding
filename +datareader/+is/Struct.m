% STRUCT   Checks if the input data is a struct with required fields
%
% Checks if the input data is a struct with required fields
%
% Parameters:
%   Din (struct): Input data in struct format
%   opts (struct): Options structure with the following fields
%       time (double): Time vector ( TrialL x 1 )
%       condition (string): Condition labels ( 1 x nTrials )
%       area (string): Area labels ( 1 x nUnits )
%
% Returns:
%   Dout (logical): True if the input data is a struct with required fields,
%       false otherwise

function Dout = Struct(Din,opts,returnDinLabel)

    % Check if the input is a string
    if nargin == 1
        if isstring(Din)
            switch Din
                case 'time'
                    Dout = ["T","t","Time","time","Times","times"];
                case 'data'
                    Dout = ["Data","data","X","x","Train","train"];
                case 'condition'
                    Dout = ["Condition","condition","Cond","cond",...
                        "Trial","trial","Trials","trials"];
                case 'area'
                    Dout = ["Area","area"];
            end
            return;
        elseif isstruct(Din)
            Dout = isstruct(Din) && ...
                all(arrayfun(@(x)isnumeric(x.data),Din));
            % If the data is a single trial, issue a warning
            if length(Din) == 1
                warning(sprintf("Single trial detected.\nInput data, when in numeric form, has the form nUnits x Time x nTrials."))
            end
            return;
        else
            error("Unrecognized call with only one non-string, non-struct argument");
        end
    end

    % Checks if Din is struct with required fields
    if nargin == 2
        returnDinLabel = false;
        Dout = datareader.is.Struct(Din);
        if Dout
            [Dout,dishomogeneous] = ValidateArgs(Din,opts);
            return;
        end
    elseif nargin == 3 && returnDinLabel

        Dout = datareader.is.Struct(Din);
        if Dout

            [Dout,dishomogeneous] = ValidateArgs(Din,opts);
            return;
        end
    end
end %nargin

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
function [Din,dishomogeneous] = ValidateArgs(Din,opts)
    % Get the default values of the options
    optsDefault = structfun(@isempty,opts,'UniformOutput',false);

    % Validate Din and standardize naming
    fnames = fieldnames(Din);
    fnames = string(fnames);
    fields2check = ["data","time","condition","area"];
    fieldType    = ["numeric","numeric","string","string"];

    Din_ = repmat(struct(),size(Din));

    % Checks for the existance of required fields
    % Checks for data type of the required fields
    % If required field is not present, checks in opts 
    for ff = [fields2check; fieldType]
        % retrieve the correct fieldname between those accepted
        thisFieldName = fnames(ismember(fnames,datareader.is.Struct(ff(1))));
        if isempty(thisFieldName) && optsDefault.(ff(1))
            error("No field %s provided in opts struct.\n",ff(1));
        elseif ~isempty(thisFieldName) &&...
                all(arrayfun(@(d) isa(d.(thisFieldName),ff(2)),Din))
            [Din_.(ff(1))] = deal(Din.(thisFieldName));
        elseif ~optsDefault.(ff(1)) &&...
                all(arrayfun(@(d) isa(opts.(ff(1)),ff(2)),Din))
            [Din_.(ff(1))] = deal(Din.(thisFieldName));
        else
            isWrongType = find(...
                ~arrayfun(@(d) isa(d.(thisFieldName),ff(2)),Din),...
                1,"first");

            wrongType = class(Din(isWrongType).(thisFieldName));
            error("Unexpected type for field %s."+ newline +...
                "Expected %s, provided %s.\n",ff(1),ff(2),...
                wrongType);
        end
    end
    
    Din = Din_; clear("Din_");

    % Get the number of trials
    nTrials = length(Din);
    % Get the number of units
    nUnits  = unique(arrayfun(@(x)size(x.data,1),Din));
    assert(numel(nUnits)==1,...
        "Detected mismatch in units number throughout trials");

    % Get the trial lengths
    Time    = arrayfun(@(x)size(x.data,2),Din);

    % Check if the trial lengths are homogeneous
    % If not, the time vector should be a cell array with the same
    % length as the number of trials
    if numel(unique(Time)) ~= 1
        dishomogeneous = true;

        % Check if the time vector is a cell array and has the same
        % length as the number of trials
        if ~optsDefault.time
            assert(iscell(opts.time),...
                'Struct input detected. Please provide a cell array of time vectors, one per trial.');

            assert(nTrials == length(opts.time),...
                'Provided time and input data dimension mismatch. Please provide a cell array of time vectors, one per trial,or set a time field in input struct.');

            % Check if each time vector has the same length as the trial
            % lengths
            assert( all( Time == cellfun(@length,opts.time) ),...
                'Provided time and input data dimension mismatch. Please provide a cell array of time vectors matching trial lenghts.');
        else % we already checked above for time in Din struct
            assert( all( Time == arrayfun(@(d)length(d.time),Din) ),...
                'Provided time and input data dimension mismatch. Please provide,for each trial, a time struct field matching data second dimension.');
        end
    else
        assert(iscell(opts.time),...
            'Struct input detected. Please provide a cell array of time vectors, one per trial.');

        assert(length(opts.time{1}) == unique(Time),...
                'Provided time and input data dimension mismatch. Please provide a cell array of time vectors matching trial lenghts.');

        dishomogeneous = false;
    end

    % Check if the area labels are provided
    if ~optsDefault.area
        assert(length(opts.area)==nUnits,...
            'Provided area label and input data first dimension mismatch.');
    else % we already checked above for area in Din struct
        assert(all(arrayfun(@(d)size(d.data,1) == length(d.area),Din)),...
                'Provided area labels do not match the number of provided units. Please provide,for each trial, a area field matching data first dimension.');
    end

    % Check if the condition labels are provided
    if ~optsDefault.condition
        assert(length(opts.condition)==nTrials,...
            'Provided conidtion labels and input data third dimension mismatch.');
    else

        assert(all(arrayfun(@(d)~isempty(d.condition),Din)),...
                'Empty condition detected! Please, provide one condition label per trial');
    end
end



