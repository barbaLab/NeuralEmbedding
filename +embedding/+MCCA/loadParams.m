function pars = loadParams()
%% GPFA specific parameters
pars = struct();

% pars.endLeg_range = @(t)getNormRange(t,fraction);
% pars.interest_range = @(t)getInterestRange(t,fraction,alignment);
% pars.ccaRefSig = [];
pars.mcca_k = 0.9; 

end
