function [Xout, yout] = blocked_permute(X, y, blockId)
%BLOCKED_PERMUTE Block-restricted label permutation.
%
%   [XOUT, YOUT] = diagnostics.shufflers.blocked_permute(X, Y, BLOCKID)
%   returns X unchanged and Y with its elements randomly permuted
%   WITHIN each block defined by BLOCKID. Labels are never swapped across
%   blocks.
%
%   This preserves:
%     - Block-level mean and distributional properties of labels (e.g.
%       session-level biases, condition imbalances).
%     - Cross-neuron covariance structure.
%   While breaking:
%     - Within-block alignment between neural activity and labels.
%
%   Use this when sessions/blocks have different overall label distributions
%   and you want a null that respects that structure.
%
%   Inputs
%   ------
%   X       : T x N data matrix (returned unchanged).
%   y       : T x 1 label vector.
%   blockId : T x 1 integer or categorical vector with block/session IDs.
%             Samples with the same blockId are treated as one block.
%
%   Outputs
%   -------
%   Xout : same as X.
%   yout : y with elements permuted within each block.
%
%   See also diagnostics.shufflers.global_permute,
%            diagnostics.shufflers.circular_shift,
%            diagnostics.compute.permutation_test

Xout = X;
yout = y;
blocks = unique(blockId);
for bb = 1:numel(blocks)
    idx = find(blockId == blocks(bb));
    yout(idx) = y(idx(randperm(numel(idx))));
end
end
