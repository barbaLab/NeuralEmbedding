function flag = computeMetrics(obj,type)
flag = true;

switch deblank(type)
    case {'arclength','Arclength','arc','Arc'}
        type = "Arclength";
        parNames = [""];
        pars = obj.assignMPars(parNames,type);
        try
            M = ...
                metrics.(type)(obj.E,pars);
        catch er
            flag = false;
            return;
        end
    case {'Alignment','alignment','align','Align'}
        type = "Alignment";
        parNames = [""];
        pars = obj.assignMPars(parNames,type);
        try
            M = ...
                metrics.(type)(obj.E,pars);
        catch er
            flag = false;
            return;
        end
    otherwise
        M = deal(nan);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Common operations on all metrics                                 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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