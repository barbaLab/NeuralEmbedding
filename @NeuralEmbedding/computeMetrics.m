function flag = computeMetrics(obj,type)
    % Compute neural embedding quality metrics
    % 
    % This function is a part of NeuralEmbedding toolbox. It computes various
    % quality metrics for neural embeddings. The metrics are computed on the
    % smoothed data and the results are stored in the Metrics property of the
    % NeuralEmbedding object. The metrics are computed only once and then
    % stored in the NeuralEmbedding object. If the metrics are requested again,
    % the stored results are returned instead of recomputing them.
    % 

    flag = true;

    switch deblank(type)
        case {'arclength','Arclength','arc','Arc','ArcLength','length','Length','len','Len'}
            type = "Arclength";
            parNames = [""];

        case {'Alignment','alignment','align','Align'}
            type = "Alignment";
            parNames = [""];
        otherwise
            error("Specific metric not found.")
    end

    % compute metrics
    try
        fprintf(1,"\nCopmuting %s for %s.%s",type,obj.Animal,obj.Session);
        pars = obj.assignMPars(parNames,type);
        M = ...
            metrics.(type)(obj.E,pars);
    catch er
        flag = false;
        return;
    end

    % Common operations on all metrics
    Mstr = obj.initMstruct(M,type);

    if isempty(obj.M)
        obj.M_ = Mstr;
    elseif obj.appendM
        obj.M = [obj.M_ Mstr];
    else
        idx = [obj.M_.type] == Mstr.type & ...
            [obj.M_.condition] == Mstr.condition & ...
            [obj.M_.Area] == Mstr.Area & ...
            [obj.M_.date] < Mstr.date;
        if any(idx)
            obj.M_(idx) = Mstr;
        else
            obj.M_ = [obj.M_ Mstr];
        end
    end

end

