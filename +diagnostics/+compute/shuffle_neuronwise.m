function Xshuf = shuffle_neuronwise(X)
%SHUFFLE_NEURONWISE Independently permute each neuron's time series.
%
%   XSHUF = diagnostics.compute.shuffle_neuronwise(X) returns a copy of X
%   where each column (neuron) has been independently randomly permuted.
%
%   This shuffle:
%     - Preserves each neuron's marginal firing-rate distribution.
%     - Destroys cross-neuron covariance (the structure the embedding uses).
%     - Destroys temporal autocorrelations within each neuron.
%
%   Use this as the null for parallel analysis (dimension selection).
%
%   Input
%   -----
%   X : T x N matrix.
%
%   Output
%   ------
%   Xshuf : T x N matrix with independently permuted columns.
%
%   See also diagnostics.compute.dim_parallel_analysis,
%            diagnostics.compute.circular_shift

[T, N] = size(X);
Xshuf  = zeros(T, N);
for nn = 1:N
    Xshuf(:, nn) = X(randperm(T), nn);
end
end
