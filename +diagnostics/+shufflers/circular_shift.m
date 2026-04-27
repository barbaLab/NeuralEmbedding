function [Xout, yout] = circular_shift(X, y)
%CIRCULAR_SHIFT Circular time-shift null for decoding permutation tests.
%
%   [XOUT, YOUT] = diagnostics.shufflers.circular_shift(X, Y) returns Y
%   unchanged and X with each column (neuron) independently circularly
%   shifted by a random offset in 1 .. T-1. This misaligns the neural
%   data with the labels without destroying each neuron's autocorrelation.
%
%   This preserves:
%     - Each neuron's temporal autocorrelation.
%     - Each neuron's marginal distribution.
%   While breaking:
%     - Alignment between neural activity and labels.
%     - Cross-neuron synchrony at zero lag.
%
%   Use this as a temporal null when the data are continuous time-series
%   (not independent shuffled trials) and temporal structure must be
%   preserved.
%
%   Inputs
%   ------
%   X : T x N matrix (rows are time-ordered samples).
%   y : T x 1 label vector (returned unchanged).
%
%   Outputs
%   -------
%   Xout : T x N matrix with independently circularly shifted columns.
%   yout : same as y.
%
%   See also diagnostics.shufflers.global_permute,
%            diagnostics.shufflers.blocked_permute,
%            diagnostics.compute.circular_shift,
%            diagnostics.compute.permutation_test

Xout = diagnostics.compute.circular_shift(X);
yout = y;
end
