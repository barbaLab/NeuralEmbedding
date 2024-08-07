function Out = Struct(Din,opts)
%DATAHIGH Summary of this function goes here
%   Detailed explanation goes here

if nargin == 1
    if isstring(Din)
        switch Din
            case 'time'
                Out = ["T","t","Time","time","Times","times"];
            case 'data'
                Out = ["Data","data","X","x","Train","train"];
            case 'condition'
                Out = ["Condition","condition","Cond","cond",...
                    "Trial","trial","Trials","trials"];
            case 'area'
                Out = ["Area","area"];
        end
        return;
    end
    Out = false;
    error("Unrecognized call with only one non-string argument");



    % Checks if Din is struct with required fields
elseif nargin == 2
    Out = false;
    if isstruct(Din)
        fnames = fieldnames(Din);
        % if opts is struct, check for presence of non defaults fields. If
        % not there check for presence of suitable field in D
        if isstruct(opts)
            OptsDefault = structfun(@isempty,opts,'UniformOutput',false);

            Out = any(ismember(datareader.is.Struct("data"),fnames))    &&...        % check for data field
                ~OptsDefault.time ||...
                    any(ismember(datareader.is.Struct("time"),fnames)) &&...    % check for time field
                ~OptsDefault.condition ||...
                    any(ismember(datareader.is.Struct("condition"),fnames))  &&...    % and so on
                ~OptsDefault.area ||...
                    any(ismember(datareader.is.Struct("area"),fnames));
        
            % retrieve the correct fieldname
        elseif isstring(opts)
            AcceptedNames = datareader.is.Struct(opts);
            Out = AcceptedNames(ismember(AcceptedNames,fnames));
        end
    end

end
end

