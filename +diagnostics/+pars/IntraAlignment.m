function pars = IntraAlignment()
%INTRAALIGNMENT Default parameters for within-session latent-space stability.
%
%   PARS = diagnostics.pars.IntraAlignment() returns a struct with the
%   default parameters used by diagnostics.compute.intra_alignment and
%   NeuralEmbedding.crossValAlignment.
%
%   Fields
%   ------
%   nSplit : positive integer (default 100)
%       Number of random half-trial splits used to build the null
%       distribution of Procrustes disparities.  Increase to 500 for
%       publication-quality estimates.
%
%   dim : non-negative integer (default 0)
%       If > 0, PCA is refit on each trial-half independently (using
%       economy SVD) and the result is projected to the first DIM
%       principal components before alignment.
%       If 0 (default), the input embeddings are aligned directly without
%       refitting (appropriate when latent coordinates from the full-session
%       PCA are passed in).
%
%   allowScale : logical (default false)
%       If false, alignment uses rotation/reflection only (orthogonal
%       Procrustes).  If true, an isotropic scale factor is also optimised.
%
%   rngSeed : non-negative integer or [] (default 0)
%       Random-number generator seed.  Set to [] to use the current RNG
%       state.
%
%   verbose : logical (default true)
%       Print progress (split counter) during computation.
%
%   See also diagnostics.compute.intra_alignment,
%            NeuralEmbedding.crossValAlignment

pars.nSplit     = 100;
pars.dim        = 0;
pars.allowScale = false;
pars.rngSeed    = 0;
pars.verbose    = true;
end
