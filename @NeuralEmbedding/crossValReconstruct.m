function results = crossValReconstruct(obj, dim, pars)
%CROSSVALRECONSTRUCT Cross-validated PCA reconstruction of neural activity.
%
%   RESULTS = CROSSVALRECONSTRUCT(OBJ, DIM) performs k-fold
%   cross-validation to estimate how well a DIM-dimensional PCA embedding
%   reconstructs held-out neural activity. Embedding and z-scoring are
%   fitted on the training folds only (no data leakage).
%
%   RESULTS = CROSSVALRECONSTRUCT(OBJ, DIM, PARS) overrides default
%   parameters via the struct PARS.  See diagnostics.pars.CVReconstruction.
%
%   Multi-session usage
%   -------------------
%   RESULTS = CROSSVALRECONSTRUCT(OBJS, DIM, ...) where OBJS is a vector /
%   array of NeuralEmbedding objects returns a (1 x nSessions) struct array.
%
%   Inputs
%   ------
%   obj : scalar or array of NeuralEmbedding objects.
%   dim : positive integer, number of latent dimensions.
%   pars: struct overriding diagnostics.pars.CVReconstruction defaults.
%         Fields: kfold (5), rngSeed (0), zscore (true).
%
%   Outputs
%   -------
%   results : struct (or struct array) with fields
%       .reconCorrMean    - mean Pearson r across folds.
%       .reconCorrPerFold - per-fold Pearson r.
%       .R2Mean           - mean R^2 across folds.
%       .R2PerFold        - per-fold R^2.
%       .corrPerNeuron    - mean per-neuron Pearson r across folds.
%       .dim, .kfold, .rngSeed - parameters used.
%       .animal, .session       - metadata.
%
%   Example
%   -------
%   res = NE.crossValReconstruct(5);
%   fprintf('Mean Pearson r = %.3f\n', res.reconCorrMean);
%
%   See also diagnostics.compute.cv_reconstruction,
%            diagnostics.pars.CVReconstruction,
%            NeuralEmbedding.selectDimension

% --- Multi-session dispatch ---
if ~isscalar(obj)
    nSess = numel(obj);
    results = cell(1, nSess);
    for ss = 1:nSess
        if nargin < 3
            results{ss} = crossValReconstruct(obj(ss), dim);
        else
            results{ss} = crossValReconstruct(obj(ss), dim, pars);
        end
    end
    results = [results{:}];
    return;
end

% --- Defaults ---
defaultPars = diagnostics.pars.CVReconstruction();
if nargin < 3 || isempty(pars)
    pars = defaultPars;
else
    pars = NeuralEmbedding.mergestructs(defaultPars, pars);
end

% --- Extract data ---
X = i_get_data(obj);

% --- Build label for progress output ---
label = sprintf('%s.%s', obj.Animal, obj.Session);
if pars.verbose
    fprintf(1, '\nCrossValReconstruct [%s]  dim=%d', label, dim);
end

% --- Run CV reconstruction ---
results = diagnostics.compute.cv_reconstruction(X, dim, pars);

% --- Attach metadata ---
results.animal  = obj.Animal;
results.session = obj.Session;

% --- Store in M_ ---
obj.i_storeM(results, 'CVReconstruction');
end

% =========================================================================
function X = i_get_data(obj)
S = obj.S;
S = S(:, 1);  % first area column
X = cell2mat(cellfun(@(s) s', S, 'UniformOutput', false));
end
