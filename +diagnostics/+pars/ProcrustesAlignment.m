function pars = ProcrustesAlignment()
%PROCRUSTESALIGNMENT Default parameters for cross-session Procrustes alignment.
%
%   PARS = diagnostics.pars.ProcrustesAlignment() returns a struct with
%   the default parameters used by diagnostics.compute.align_procrustes
%   and diagnostics.compute.alignment_metrics.
%
%   Fields
%   ------
%   allowScale : logical (default false)
%       If false, the alignment is restricted to rotation and reflection
%       (orthogonal Procrustes). If true, an isotropic scaling factor is
%       also optimised.
%
%   refSession : positive integer (default 1)
%       Index of the session (within an object array) used as the
%       alignment reference.
%
%   See also diagnostics.compute.align_procrustes,
%            diagnostics.compute.alignment_metrics

pars.allowScale = false;
pars.refSession = 1;
end
