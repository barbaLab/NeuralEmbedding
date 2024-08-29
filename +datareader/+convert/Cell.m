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

        % Convert the cell array to a numeric array
        Din_ = cat(3,Din{:});
        % Check if the numeric array is in the correct format
        [Dout,TrialTime,nUnits,nTrial,Condition,Area] = ...
            NeuralEmbedding.datareader.convert.Double(Din_,opts);
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
    
    