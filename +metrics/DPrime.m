function dp = DPrime(E, pars)
%DPRIME computes d-prime modulation index between baseline and signal windows
%
% input
%                       E             1xnTrials cell array, each with nDims×nTime matrix
%                       pars          parameter struct with fields:
%                                       .baseline_idx  indices of baseline time points
%                                       .signal_idx    indices of signal time points
% output
%                       dp            1×nDims vector of d' values

if nargin < 2 || isempty(pars)
    pars = metrics.pars.DPrime();
end

dp = metrics.compute.dprime(cat(3, E{:}), pars.baseline_idx, pars.signal_idx);

end
