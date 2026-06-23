function results = alignment_metrics(Z1, Z2_aligned)
%ALIGNMENT_METRICS Quantitative metrics for latent-space alignment quality.
%
%   RESULTS = diagnostics.compute.alignment_metrics(Z1, Z2_ALIGNED)
%   computes several metrics that quantify how well two latent spaces are
%   aligned, assuming Z2_ALIGNED has already been transformed by
%   diagnostics.compute.align_procrustes.
%
%   Inputs
%   ------
%   Z1         : T x d reference latent space (centred).
%   Z2_aligned : T x d aligned latent space.
%
%   Outputs
%   -------
%   results : struct with fields
%       .disparity       - Procrustes disparity ||Z1-Z2||_F^2 / ||Z1||_F^2.
%       .principalAngles - (1 x d) principal angles (radians) between the
%                          column spaces of Z1 and Z2_aligned, in ascending
%                          order.
%       .meanPrincipalAngle - mean of principalAngles (radians).
%       .distCorr        - Pearson correlation between vectorised pairwise
%                          Euclidean distance matrices of Z1 and Z2_aligned
%                          (Mantel-style; 1 = identical structure).
%
%   Notes
%   -----
%   * Principal angles: let Q1, Q2 be column-space orthonormal bases (from
%     QR decomposition). Angles = acos(svd(Q1'*Q2)).
%   * Distance correlation is computed on the upper-triangular part of the
%     pairwise distance matrix to avoid double-counting.
%   * For large T, distance-correlation computation is O(T^2); set
%     distCorr to NaN if T > 2000 to avoid excessive runtime.
%   * No toolboxes required.
%
%   Example
%   -------
%   proc = diagnostics.compute.align_procrustes(Z1, Z2);
%   met  = diagnostics.compute.alignment_metrics(Z1, proc.Z2_aligned);
%   fprintf('Mean principal angle = %.2f deg\n', ...
%       rad2deg(met.meanPrincipalAngle));
%
%   See also diagnostics.compute.align_procrustes

% --- Validation ---
if size(Z1, 2) ~= size(Z2_aligned, 2)
    error('diagnostics:alignment_metrics:dimMismatch', ...
        'Z1 and Z2_aligned must have the same number of columns.');
end

% Centre
Z1 = Z1 - mean(Z1, 1);
Z2 = Z2_aligned - mean(Z2_aligned, 1);

d = size(Z1, 2);
T = size(Z1, 1);

% --- Disparity ---
nZ1 = norm(Z1, 'fro');
if nZ1 == 0
    disparity = 0;
else
    disparity = norm(Z1 - Z2, 'fro')^2 / nZ1^2;
end

% --- Principal angles ---
[Q1, ~] = qr(Z1, 0);   % economy QR
[Q2, ~] = qr(Z2, 0);
svs     = svd(Q1' * Q2);
svs     = min(max(svs, -1), 1);   % clamp for numerical safety
principalAngles = acos(svs(:)');

% --- Distance correlation (Mantel-style) ---
if T <= 2000
    D1 = i_pdist_upper(Z1);
    D2 = i_pdist_upper(Z2);
    distCorr = i_pearson(D1, D2);
else
    distCorr = NaN;  % too expensive for large T
end

results.disparity           = disparity;
results.principalAngles     = principalAngles;
results.meanPrincipalAngle  = mean(principalAngles);
results.distCorr            = distCorr;
end

% =========================================================================
%  Local helpers
% =========================================================================

function v = i_pdist_upper(Z)
% Upper-triangular pairwise Euclidean distances as a vector.
T = size(Z, 1);
v = zeros(1, T*(T-1)/2);
k = 0;
for ii = 1:T-1
    diff = Z(ii+1:end, :) - Z(ii, :);
    n    = T - ii;
    v(k+1 : k+n) = sqrt(sum(diff.^2, 2));
    k = k + n;
end
end

function r = i_pearson(a, b)
a = a(:); b = b(:);
a = a - mean(a); b = b - mean(b);
na = norm(a); nb = norm(b);
if na == 0 || nb == 0
    r = 0;
    return;
end
r = (a' * b) / (na * nb);
end
