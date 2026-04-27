function pars = ParallelAnalysis()
%PARALLELANALYSIS Default parameters for shuffle-based dimension selection.
%
%   PARS = diagnostics.pars.ParallelAnalysis() returns a struct with the
%   default parameters used by diagnostics.compute.dim_parallel_analysis.
%
%   Fields
%   ------
%   nShuffle : positive integer (default 200)
%       Number of shuffle replicates for the null distribution.
%
%   alpha : scalar in (0,1) (default 0.05)
%       Significance level. The null quantile is computed at 1-alpha.
%
%   rngSeed : non-negative integer or [] (default 0)
%       Random-number generator seed. Set to [] to use the current RNG
%       state without resetting.
%
%   mode : 'neuronwise' | 'rowperm' (default 'neuronwise')
%       Shuffle strategy.
%       'neuronwise' - independently permute each neuron's time series,
%                      breaking cross-neuron covariance while preserving
%                      each neuron's marginal distribution.
%       'rowperm'    - permute entire rows (time points), a weaker null
%                      that also breaks temporal correlations within each
%                      neuron.
%
%   See also diagnostics.compute.dim_parallel_analysis

pars.nShuffle = 200;
pars.alpha    = 0.05;
pars.rngSeed  = 0;
pars.mode     = 'neuronwise';
end
