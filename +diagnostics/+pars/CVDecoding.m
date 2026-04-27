function pars = CVDecoding()
%CVDECODING Default parameters for cross-validated decoding and permutation test.
%
%   PARS = diagnostics.pars.CVDecoding() returns a struct with the default
%   parameters used by diagnostics.compute.cv_decoding and
%   diagnostics.compute.permutation_test.
%
%   Fields
%   ------
%   kfold : positive integer >= 2 (default 5)
%       Number of cross-validation folds for decoding.
%
%   nPerm : positive integer (default 500)
%       Number of permutation replicates for the null distribution.
%
%   rngSeed : non-negative integer or [] (default 0)
%       Random-number generator seed. Set to [] to skip RNG reset.
%
%   decoder : 'nearestCentroid' (default 'nearestCentroid')
%       Decoder type. Currently only nearest-centroid is supported
%       (no additional toolboxes required).
%
%   permMode : 'global' | 'blocked' | 'timeshift' (default 'global')
%       Permutation strategy used when building the null distribution.
%       'global'    - permute all labels randomly.
%       'blocked'   - permute labels within user-supplied session blocks.
%       'timeshift' - circular time-shift of the neural data (preserves
%                     auto-correlation).
%
%   See also diagnostics.compute.cv_decoding,
%            diagnostics.compute.permutation_test

pars.kfold    = 5;
pars.nPerm    = 500;
pars.rngSeed  = 0;
pars.decoder  = 'nearestCentroid';
pars.permMode = 'global';
end
