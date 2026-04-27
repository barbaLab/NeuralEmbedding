function Xout = circular_shift(X)
%CIRCULAR_SHIFT Circular time-shift null for neural time-series data.
%
%   XOUT = diagnostics.compute.circular_shift(X) returns a copy of X where
%   each column (neuron) has been independently shifted by a uniformly
%   random time offset in 1 .. T-1 (wrapping at the boundaries).
%
%   This shuffle:
%     - Preserves each neuron's autocorrelation structure.
%     - Preserves each neuron's marginal distribution.
%     - Destroys the alignment between the neural data and any external
%       labels (behaviour, task variables).
%     - Partially preserves instantaneous cross-neuron correlations at
%       non-zero lags (unlike neuronwise shuffle).
%
%   Use this as a temporal null for decoding permutation tests when the
%   data are sampled continuously in time (not independent trials).
%
%   Input
%   -----
%   X : T x N matrix, where rows are time-ordered samples.
%
%   Output
%   ------
%   Xout : T x N matrix with independently circularly shifted columns.
%
%   See also diagnostics.compute.shuffle_neuronwise,
%            diagnostics.shufflers.circular_shift

[T, N] = size(X);
Xout   = zeros(T, N);
shifts = randi([1, T-1], 1, N);   % independent shift per neuron
for nn = 1:N
    Xout(:, nn) = circshift(X(:, nn), shifts(nn));
end
end
