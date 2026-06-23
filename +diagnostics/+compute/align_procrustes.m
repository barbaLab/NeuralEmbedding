function results = align_procrustes(Z1, Z2, allowScale)
%ALIGN_PROCRUSTES Orthogonal Procrustes alignment of two latent spaces.
%
%   RESULTS = diagnostics.compute.align_procrustes(Z1, Z2) finds the
%   orthogonal matrix R that minimises ||Z1 - Z2*R||_F, and returns the
%   aligned Z2 and related metrics.
%
%   RESULTS = diagnostics.compute.align_procrustes(Z1, Z2, ALLOWSCALE)
%   also optimises an isotropic scaling factor when ALLOWSCALE is true.
%
%   Both matrices are centred (mean-subtracted) before alignment.
%
%   Inputs
%   ------
%   Z1         : T x d latent space (reference session, e.g. trials x dims).
%   Z2         : T x d latent space (session to be aligned). Must have the
%                same number of columns d as Z1.
%   allowScale : logical (default false). Allow isotropic scaling.
%
%   Outputs
%   -------
%   results : struct with fields
%       .Z2_aligned  - (T x d) Z2 after applying R (and scale if enabled).
%       .R           - (d x d) orthogonal rotation matrix.
%       .scale       - isotropic scale factor (1.0 when allowScale=false).
%       .disparity   - Procrustes disparity = ||Z1 - Z2_aligned||_F^2 /
%                      ||Z1||_F^2 (normalised; 0 = perfect alignment).
%
%   Notes
%   -----
%   * Implements the classic SVD solution: [U,~,V] = svd(Z1'*Z2);
%     R = V * U'.
%   * No toolboxes required.
%   * For multi-session alignment see NeuralEmbedding.alignSessions.
%
%   Example
%   -------
%   res = diagnostics.compute.align_procrustes(Z1, Z2);
%   fprintf('Disparity = %.4f\n', res.disparity);
%
%   See also diagnostics.compute.alignment_metrics,
%            diagnostics.pars.ProcrustesAlignment

if nargin < 3
    allowScale = false;
end

% --- Validation ---
if ~ismatrix(Z1) || ~isnumeric(Z1) || ~ismatrix(Z2) || ~isnumeric(Z2)
    error('diagnostics:align_procrustes:badInput', ...
        'Z1 and Z2 must be 2-D numeric matrices.');
end
if size(Z1, 2) ~= size(Z2, 2)
    error('diagnostics:align_procrustes:dimMismatch', ...
        'Z1 and Z2 must have the same number of columns (latent dims).');
end

% --- Centre ---
Z1c = Z1 - mean(Z1, 1);
Z2c = Z2 - mean(Z2, 1);

% --- SVD solution ---
M       = Z1c' * Z2c;             % d x d
[U, S, V] = svd(M);
R       = V * U';                  % orthogonal rotation

% --- Optional scale ---
if allowScale
    scale = trace(S) / (norm(Z2c, 'fro')^2);
else
    scale = 1.0;
end

Z2_aligned = scale * Z2c * R;

% --- Disparity ---
norm_Z1 = norm(Z1c, 'fro');
if norm_Z1 == 0
    disparity = 0;
else
    disparity = norm(Z1c - Z2_aligned, 'fro')^2 / norm_Z1^2;
end

results.Z2_aligned = Z2_aligned;
results.R          = R;
results.scale      = scale;
results.disparity  = disparity;
end
