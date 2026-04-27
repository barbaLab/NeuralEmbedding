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
    % When called on a non-scalar NeuralEmbedding array, the behaviour
    % depends on the requested metric type:
    %   - Alignment : trajectories from all objects are pooled together so
    %                 that cross-session alignment is computed. The single
    %                 resulting value is stored in every object in the array.
    %   - All other metrics : the metric is computed independently for each
    %                         object in the array.
    %

    flag = true;

    switch deblank(type)
        case {'arclength','Arclength','arc','Arc','ArcLength','length','Length','len','Len'}
            type = "Arclength";
            parNames = [""];

        case {'Alignment','alignment','align','Align'}
            type = "Alignment";
            parNames = [""];

        case {'DPrime','dprime','dPrime','d_prime','d-prime'}
            type = "DPrime";
            parNames = ["baseline_idx","signal_idx"];
        otherwise
            error("Specific metric not found.")
    end

    % Handle non-scalar object arrays
    if ~isscalar(obj)
        switch type
            case 'Alignment'
                % Pool trajectories from all objects for cross-session alignment
                allE = arrayfun(@(o) o.E, obj, 'UniformOutput', false);
                E_combined = vertcat(allE{:});
                try
                    fprintf(1, "\nComputing cross-session %s", type);
                    pars = obj(1).assignMPars(parNames, type);
                    M = metrics.(type)(E_combined, pars);
                catch er
                    flag = false;
                    return;
                end
                % Store the shared result in every object
                for ii = 1:numel(obj)
                    Mstr = obj(ii).initMstruct(M, type);
                    if isempty(obj(ii).M)
                        obj(ii).M_ = Mstr;
                    elseif obj(ii).appendM
                        obj(ii).M_ = [obj(ii).M_ Mstr];
                    else
                        idx = [obj(ii).M_.type] == Mstr.type & ...
                            [obj(ii).M_.condition] == Mstr.condition & ...
                            [obj(ii).M_.Area] == Mstr.Area & ...
                            [obj(ii).M_.date] < Mstr.date;
                        if any(idx)
                            obj(ii).M_(idx) = Mstr;
                        else
                            obj(ii).M_ = [obj(ii).M_ Mstr];
                        end
                    end
                end
            otherwise
                % For metrics that are meaningful per-session, compute individually
                flag = arrayfun(@(o) o.computeMetrics(type), obj);
        end
        return;
    end

    % compute metrics (single object)
    try
        fprintf(1,"\nComputing %s for %s.%s",type,obj.Animal,obj.Session);
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
        obj.M_ = [obj.M_ Mstr];
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

