function pars = loadParams()
%% GPFA specific parameters
pars = struct();

pars.endLeg_range = @(t)getNormRange(t,fraction);
pars.interest_range = @(t)getInterestRange(t,fraction,alignment);
pars.ccaRefSig = [];

end

function T = getNormRange(t,fraction)
    
    Tmax = length(t);
    T = false(size(t));
    T(1:Tmax/fraction/2) = true;
    T(end-Tmax/fraction/2:end) = true;
    T = t(T);
end

function T = getInterestRange(t,fraction,alignment)

    Tmax = length(t);
    T = false(size(t));
    Pre = alignment-Tmax/fraction/2;
    Post = alignment+Tmax/fraction/2;
    T(Pre:Post) = true;
    T = t(T);
end