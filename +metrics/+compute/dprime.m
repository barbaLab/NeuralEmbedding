function dp = dprime(data, baseline_idx, signal_idx)
%DPRIME computes the d-prime modulation index between two time windows
%
% Inputs:
%   data          - nDims x nTime x nTrials array of embedded signals
%   baseline_idx  - indices of baseline time points
%   signal_idx    - indices of signal time points
%
% Output:
%   dp            - 1 x nDims vector of d' values

if isempty(data)
    dp = nan;
    return;
end

[nDims, ~, nTrials] = size(data);

if nTrials < 2
    dp = nan(1, nDims);
    return;
end

% Mean activity per trial within each window
baseline = squeeze(mean(data(:, baseline_idx, :), 2));  % nDims x nTrials
signal   = squeeze(mean(data(:, signal_idx,   :), 2));  % nDims x nTrials

% When nDims == 1, squeeze reduces 1x1xnTrials to a nTrials×1 column;
% reshape to 1×nTrials so that trailing dimension is always trials.
if nDims == 1
    baseline = baseline(:)';
    signal   = signal(:)';
end

% Compute d' per dimension (statistics across trials)
mu_b  = mean(baseline, 2);
mu_s  = mean(signal,   2);
std_b = std(baseline,  0, 2);
std_s = std(signal,    0, 2);

% Pooled standard deviation: sqrt( (sigma_b^2 + sigma_s^2) / 2 )
pooled_std = sqrt(0.5 .* (std_b.^2 + std_s.^2));

dp = (mu_s - mu_b) ./ pooled_std;

% Return as 1 x nDims row vector
dp = dp';

end
