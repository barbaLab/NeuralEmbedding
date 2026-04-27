function pars = CVReconstruction()
%CVRECONSTRUCTION Default parameters for cross-validated reconstruction.
%
%   PARS = diagnostics.pars.CVReconstruction() returns a struct with the
%   default parameters used by diagnostics.compute.cv_reconstruction.
%
%   Fields
%   ------
%   kfold : positive integer >= 2 (default 5)
%       Number of cross-validation folds.
%
%   rngSeed : non-negative integer or [] (default 0)
%       Random-number generator seed. Set to [] to skip RNG reset.
%
%   zscore : logical (default true)
%       If true, z-score each neuron using statistics computed from the
%       training fold only (no leakage).
%
%   See also diagnostics.compute.cv_reconstruction

pars.kfold   = 5;
pars.rngSeed = 0;
pars.zscore  = true;
end
