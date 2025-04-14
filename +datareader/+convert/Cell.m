function [Dout,TrialTime,nUnits,nTrial,Condition,Area,Dishomogeneous] = Cell(Din,opts)
    optsDefault = structfun(@isempty,opts,'UniformOutput',false);

    % TODO check for fs
    if optsDefault.fs
        error('fs field is missing in opts structure.\nIt is required when converting data from double format.\nPlease input the sampling frequency (in Hz) in opts.fs.');
    end
    % Get the number of trials
    nTrial = length(Din);
    % Get the number of units
    nUnits  = unique(cellfun(@(x)size(x,1),Din));
    % Get the trial lengths
    TrialL    = cellfun(@(x)size(x,2),Din);


    if length(unique(TrialL)) == 1

        Dishomogeneous = false;
        dataclass = cellfun(@(x)issparse(x),Din);
        if not(all(dataclass))
            tmp = cellfun(@sparse,Din(not(dataclass)),'UniformOutput',false);
            Dout = Din;
            Dout(not(dataclass)) = tmp;
        else
            Dout = Din;
        end
        TrialTime = opts.time;
        if not(iscell(TrialTime))
            TrialTime = repmat({TrialTime},nTrial,1);
        else
            TrialTime = repmat(TrialTime(1),nTrial,1);
        end
        Condition   = opts.condition;
        Area        = opts.area;
        return;
    end

    Dishomogeneous = true;
        Dout = cellfun(@(x) sparse(x),...
            Din,'UniformOutput',false);
    Dout = Dout(:);
    
    TrialTime = opts.time;
    
    Condition   = opts.condition;
    Area        = opts.area;
    
    
    end
    
    