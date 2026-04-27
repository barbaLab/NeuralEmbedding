function [Xout, yout] = global_permute(X, y)
%GLOBAL_PERMUTE Global (unrestricted) label permutation.
%
%   [XOUT, YOUT] = diagnostics.shufflers.global_permute(X, Y) returns X
%   unchanged and Y with its elements randomly permuted across all samples.
%
%   This is the standard label-shuffle null. It breaks any association
%   between neural activity and the labels while preserving:
%     - The marginal distribution of each neuron's activity.
%     - Cross-neuron covariance structure.
%     - The marginal distribution of labels.
%
%   Use this as the default null for decoding permutation tests when there
%   is no block structure to preserve.
%
%   Inputs
%   ------
%   X : T x N data matrix (returned unchanged).
%   y : T x 1 label vector.
%
%   Outputs
%   -------
%   Xout : same as X.
%   yout : y with elements randomly permuted.
%
%   See also diagnostics.shufflers.blocked_permute,
%            diagnostics.shufflers.circular_shift,
%            diagnostics.compute.permutation_test

Xout = X;
yout = y(randperm(numel(y)));
end
