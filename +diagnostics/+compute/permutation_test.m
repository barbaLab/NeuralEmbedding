function results = permutation_test(statFcn, permuterFcn, X, y, nPerm, rngSeed)
%PERMUTATION_TEST Generic permutation test for decoding statistics.
%
%   RESULTS = diagnostics.compute.permutation_test(STATFCN, PERMUTERFCN,
%   X, Y, NPERM, RNGSEED) computes a one-sided permutation test.
%
%   The test evaluates whether the real statistic (computed on the original
%   data) is significantly larger than expected by chance.
%
%   Inputs
%   ------
%   statFcn    : function handle @(X, y) -> scalar statistic.
%                Called with the original data to obtain accReal, then
%                called with permuted (X or y) for each null replicate.
%   permuterFcn: function handle @(X, y) -> [Xp, yp].
%                Returns permuted copies of X and/or y for one replicate.
%                Use diagnostics.shufflers.global_permute,
%                    diagnostics.shufflers.blocked_permute, or
%                    diagnostics.shufflers.circular_shift.
%   X          : T x N data matrix.
%   y          : T x 1 label vector.
%   nPerm      : positive integer, number of permutation replicates.
%   rngSeed    : non-negative integer or [] (default 0).
%
%   Outputs
%   -------
%   results : struct with fields
%       .statReal  - observed statistic.
%       .statNull  - (1 x nPerm) null distribution.
%       .pValue    - one-sided p-value: (1 + sum(null >= real)) / (nPerm+1).
%       .effectZ   - effect-size z-score: (real - mean(null)) / std(null).
%       .nPerm     - nPerm used.
%       .rngSeed   - rngSeed used.
%
%   Notes
%   -----
%   * The p-value formula (1 + sum(null >= real)) / (nPerm + 1) is the
%     standard permutation-test estimator (Phipson & Smyth, 2010). It is
%     bounded below by 1/(nPerm+1) and is conservative under the null.
%   * The z-score provides an effect size regardless of nPerm.
%
%   Example
%   -------
%   statFcn     = @(X,y) diagnostics.compute.cv_decoding(X, y, 5).accMean;
%   permuterFcn = @(X,y) diagnostics.shufflers.global_permute(X, y);
%   res = diagnostics.compute.permutation_test(statFcn, permuterFcn, X, y, 500, 0);
%   fprintf('p = %.4f, z = %.2f\n', res.pValue, res.effectZ);
%
%   See also diagnostics.shufflers.global_permute,
%            diagnostics.shufflers.blocked_permute,
%            diagnostics.shufflers.circular_shift

if nargin < 6 || isempty(rngSeed)
    rngSeed = 0;
end
if ~isempty(rngSeed)
    rng(rngSeed, 'twister');
end

% Real statistic
statReal = statFcn(X, y);

% Null distribution
statNull = zeros(1, nPerm);
for pp = 1:nPerm
    [Xp, yp]    = permuterFcn(X, y);
    statNull(pp) = statFcn(Xp, yp);
end

% p-value (one-sided, Phipson & Smyth 2010)
pValue  = (1 + sum(statNull >= statReal)) / (nPerm + 1);

% Effect size z-score
mn = mean(statNull);
sd = std(statNull);
if sd == 0
    effectZ = 0;
else
    effectZ = (statReal - mn) / sd;
end

results.statReal = statReal;
results.statNull = statNull;
results.pValue   = pValue;
results.effectZ  = effectZ;
results.nPerm    = nPerm;
results.rngSeed  = rngSeed;
end
