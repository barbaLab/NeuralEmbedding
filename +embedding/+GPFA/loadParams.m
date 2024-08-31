function pars = loadParams()
%% GPFA specific parameters
pars = struct();

pars.startTau      = 100; % in msec
pars.startEps      = 1e-3;

pars.seqTest = [];
pars.doTest = false;
end

