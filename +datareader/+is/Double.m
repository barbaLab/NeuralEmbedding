function Dout = Double(Din,opts)
optsDefault = structfun(@isempty,opts,'UniformOutput',false);
Dout = true;
if nargin == 1
    Dout = Dout && isnumeric(Din);
    if size(Din,3) == 1
        warning(sprintf("Single trial detected.\nInput data, when in numeric form, has the form nUnits X Time X nTrials."))
    end
    return;
    % Checks if Din domensions matche opts
elseif nargin == 2
    [nUnits,Time,nTrials] = size(Din);
    if ~optsDefault.time
        assert(length(opts.time)==Time,...
            'Provided time and input data second dimension mismatch.');
    else
        ... this is fine;
    end
    if ~optsDefault.area
        assert(length(opts.area)==nUnits,...
            'Provided area label and input data first dimension mismatch.');
    else
        error(sprintf('Area labels not provided.\nThey are required for input data in numeric form [string array, 1xnUnits].'));
    end
    if ~optsDefault.condition
        assert(length(opts.condition)==nTrials,...
            'Provided conidtion labels and input data third dimension mismatch.');
    else
        error(sprintf('Condition labels not provided.\nThey are required for input data in numeric form [string array, 1xnTrials'))
    end
    
    Dout = Dout &&  isnumeric(Din);
    if size(Din,3) == 1
        warning(sprintf("Single trial detected.\nInput data, when in numeric form, has the form nUnits X Time X nTrials."))
    end
    return;
end
end

