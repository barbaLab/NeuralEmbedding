# Dimensionality Reduction Techniques

NeuralEmbedding exposes a common interface to several dimensionality reduction algorithms through [`findEmbedding`](./@NeuralEmbedding/findEmbedding.m). Each method consumes the preprocessed neural activity stored in the `NeuralEmbedding` object and returns embedded trajectories along with the projection matrices that produced them.

## Typical workflow

```matlab
% D is a struct array, numeric array, or cell array of trials (see constructor for formats)
NE = NeuralEmbedding(D, ...
    "fs", 1000, ...           % sampling frequency in Hz
    "time", tvec, ...         % optional trial time vector
    "area", areaLabels, ...   % optional area labels per unit
    "condition", condLabels); % optional condition label per trial

% Choose an embedding
[E, W, VarExplained] = NE.findEmbedding("PCA");   % returns embedded data, projection, variance explained
```

`findEmbedding(type, projectOnly)` accepts the method name (case-insensitive) and an optional `projectOnly` flag to bypass model fitting when a projection already exists.

## Available methods

### Principal Component Analysis (PCA)
- **Call:** `NE.findEmbedding("PCA");`
- **Purpose:** Orthogonal linear projection capturing maximal variance.
- **Parameters:** `NumPC` property (automatically capped at 30 and the available units in the selected area mask); inherits smoothing/z-scoring from preprocessing.
- **Inputs:** Smoothed data `obj.S` (cells of `nDims × T` matrices per trial).
- **Outputs:** Embedded trajectories `E`, projection matrices `W`, variance explained per component `VarExplained`.

### Gaussian Process Factor Analysis (GPFA)
- **Call:** `NE.findEmbedding("GPFA");`
- **Purpose:** Captures smooth latent trajectories with temporal structure.
- **Parameters:** `NumPC` (set on the object), `TrialL` (trial length in bins), `subsampling` (inherited from preprocessing).
- **Inputs:** Preprocessed binned data `obj.P`.
- **Outputs:** `E`, `W`, and `VarExplained`; inverse projection is computed automatically.

### Canonical Correlation Analysis (CCA)
- **Call:** `NE.findEmbedding("CCA");`
- **Purpose:** Finds correlated projections across multiple recorded areas.
- **Parameters:** `NumPC`, `nArea`, `nTrial`, `TrialL`.
- **Inputs:** Cell array `D` containing smoothed activity per area (`obj.S` is assembled internally).
- **Outputs:** `E`, area-specific projection matrices `W`, and canonical correlations.

### Multi-set Canonical Correlation Analysis (MCCA)
- **Call:** `NE.findEmbedding("MCCA");`
- **Purpose:** Aligns multiple `NeuralEmbedding` objects (e.g., multiple sessions) into a shared latent space.
- **Parameters:** `mcca_k` (regularization, default `0.9`), `nTrial`, `TrialL`.
- **Notes:** If invoked on a single object, it falls back to the CCA workflow.
- **Outputs:** Shared embeddings and projections applied back to each object in the array.

### Identity
- **Call:** `NE.findEmbedding("identity");`
- **Purpose:** Bypasses dimensionality reduction; returns smoothed data with an identity projection.
- **Outputs:** Embedded data identical to the smoothed input, identity projection matrix, and explained variance of the original space.

### Experimental / WIP
- **UMAP / t-SNE:** Stubs exist, but calls currently print “support is WIP. Stay tuned!” and return without computation.

## Tips for effective use
- The constructor automatically performs preprocessing (`removeInactiveNeurons`, `binData`, `smoothData`, `zscoreData`) using the parameters passed in `opts` (e.g., `binwidth`, `prekern`, `useGpu`). If you change these properties after construction, rerun `performPrePro` before calling `findEmbedding` to refresh the processed data.
- Use `projectOnly=true` when projecting new trials with an existing `W` matrix.
- Access masks (`aMask`, `cMask`, `tMask`) to limit areas, conditions, or time bins before running an embedding.
