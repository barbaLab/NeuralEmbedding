function TA = Alignment(E,pars)
%ALIGNMENT computes trajectory alignment and returns it's standard
%deviation

pars =  metrics.pars.alignment();

theta = metrics.compute.alignment(cat(E{:},3),pars.type,pars.nsec);

TA = mean(...                           % mean across points
    std(theta,[],1)...                  % for each point, compute std across trials
    );

end

