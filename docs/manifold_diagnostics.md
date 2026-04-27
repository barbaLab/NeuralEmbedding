# Manifold Diagnostics

This page describes the **manifold diagnostic toolkit** added to NeuralEmbedding.
It provides robust best-practice diagnostics for reconstructed neural manifolds
and latent spaces, covering four areas:

| Method | Class method | Underlying compute |
|--------|-------------|-------------------|
| A – Dimension selection | `selectDimension` | `diagnostics.compute.dim_parallel_analysis` |
| B – CV reconstruction | `crossValReconstruct` | `diagnostics.compute.cv_reconstruction` |
| C – CV decoding + permutation test | `crossValDecode` | `diagnostics.compute.cv_decoding`, `diagnostics.compute.permutation_test` |
| D – Multi-session alignment | `alignSessions` | `diagnostics.compute.align_procrustes`, `diagnostics.compute.alignment_metrics` |

---

## Quick-start

```matlab
% Single session
NE = NeuralEmbedding(data, 'condition', labels, 'session', 'S1', 'animal', 'Rat1');

% A) Select dimensionality
resA = NE.selectDimension(1:15);
fprintf('dStar = %d\n', resA.dStar);

% B) Cross-validated reconstruction
resB = NE.crossValReconstruct(resA.dStar);
fprintf('CV Pearson r = %.3f\n', resB.reconCorrMean);

% C) CV decoding + permutation test
y_trial = ...; % trial-level integer labels
resC = NE.crossValDecode(y_trial, resA.dStar);
fprintf('Acc = %.1f%%  p = %.4f\n', resC.accMean*100, resC.pValue);

% D) Align two sessions
NE1.findEmbedding('PCA'); NE2.findEmbedding('PCA');
resD = alignSessions([NE1, NE2]);
fprintf('Disparity = %.4f\n', resD.disparity(2));
```

Multi-session methods accept an array of NeuralEmbedding objects and return a
struct array (one entry per session):

```matlab
NEobjs = [NE1, NE2, NE3];  % vector of NeuralEmbedding objects

% All four methods work with object arrays
resA = NEobjs.selectDimension(1:15);
resD = NEobjs.alignSessions();
```

---

## A – Dimension selection (`selectDimension`)

### What it tests
Whether the dimensionality of the real neural covariance exceeds that expected
by chance given the marginal statistics of each neuron.

### Method: parallel analysis / shuffle null
1. Compute PCA eigenspectrum of the real (z-scored) data.
2. For each of `nShuffle` replicates, independently permute each neuron's
   time series (mode `'neuronwise'`) or shuffle entire rows (mode `'rowperm'`).
3. Select `d*` = largest component k where the real eigenvalue exceeds the
   (1-α) quantile of null eigenvalues.

### What each null preserves / breaks

| Null mode | Preserves | Breaks |
|-----------|-----------|--------|
| `neuronwise` | Marginal distribution of each neuron | Cross-neuron covariance, temporal autocorrelation within each neuron |
| `rowperm` | Instantaneous cross-neuron patterns | Temporal structure of every neuron |

**Recommendation**: use `'neuronwise'` (default). It produces a sharper null
because it destroys the specific structure the embedding exploits.

### Recommended defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `nShuffle` | 200 | 500 for publication-quality results |
| `alpha` | 0.05 | |
| `mode` | `'neuronwise'` | |
| `rngSeed` | 0 | Set to `[]` to use current RNG state |

### Interpreting results
- `dStar = 0` means no component exceeded the null → data may be too noisy or
  the embedding method is not capturing meaningful variance.
- `dStar` near the true latent dimensionality is expected for well-separated
  synthetic data.

---

## B – Cross-validated reconstruction (`crossValReconstruct`)

### What it tests
How well the latent embedding reconstructs held-out neural activity (no data
leakage).

### Method: k-fold CV PCA reconstruction
1. Split time bins into k folds.
2. For each fold: fit PCA (and optional z-score) on training fold only; project
   test fold into latent space; reconstruct by projecting back.
3. Score: Pearson r and R² between vectorised `X_test` and `Xhat_test`.

### No-leakage guarantee
All preprocessing (z-score mean/std, PCA loadings) is estimated from the
training fold and applied to the test fold.

### Recommended defaults

| Parameter | Default |
|-----------|---------|
| `kfold` | 5 |
| `zscore` | `true` |

### Interpreting results
- `reconCorrMean` increases with `dim` up to the true latent dimensionality and
  plateaus (or decreases slightly due to overfitting) beyond.
- Use the elbow in the reconstruction-vs-dim curve to corroborate `dStar` from
  parallel analysis.

---

## C – CV decoding + permutation test (`crossValDecode`)

### What it tests
Whether the latent representation contains statistically significant information
about task labels / behaviour variables, using a permutation-based null.

### Method
1. **Decoding**: k-fold CV with a nearest-centroid classifier in the PCA latent
   space (no toolbox required).
2. **Permutation test**: re-run the same CV pipeline `nPerm` times under
   permuted data/labels. p-value:

   ```
   p = (1 + sum(null_acc >= real_acc)) / (nPerm + 1)
   ```

   Effect size: `z = (real - mean(null)) / std(null)`.

### Permutation modes

| `permMode` | What is permuted | Preserves | Breaks |
|------------|-----------------|-----------|--------|
| `'global'` | All labels randomly permuted | Label marginal distribution | Label–activity alignment |
| `'blocked'` | Labels permuted within each trial block | Block-level label distribution | Within-block alignment |
| `'timeshift'` | Neural data circularly shifted per neuron | Neural autocorrelation | Neural–label alignment |

**Recommendation**:
- Default: `'global'` – the strictest null for classification tasks.
- Use `'blocked'` when sessions / conditions have different overall label
  frequencies that must be preserved.
- Use `'timeshift'` when temporal structure in the neural data is important and
  must be preserved in the null (e.g., continuous recordings).

### Recommended defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kfold` | 5 | |
| `nPerm` | 500 | 1000 for publication |
| `permMode` | `'global'` | |

### Interpreting p-values
- `p < alpha` (e.g. 0.05): the decoded labels are significantly better than
  chance.
- `p >= alpha`: decoding is not significant at the chosen level.
- `effectZ > 2`: strong effect regardless of `nPerm`.
- The permutation p-value is bounded below by `1/(nPerm+1)`.

---

## D – Multi-session alignment (`alignSessions`)

### What it tests
Whether the latent geometry is consistent across sessions / animals.

### Method: orthogonal Procrustes
For each non-reference session:
1. Centre both latent matrices.
2. Solve `min ||Z_ref - Z_sess * R||_F` subject to `R'R = I` (orthogonal
   Procrustes) via SVD: `[U,S,V] = svd(Z_ref' * Z_sess); R = V * U'`.
3. Optionally optimise an isotropic scale factor (`allowScale = true`).

### Alignment metrics

| Metric | Meaning | Good value |
|--------|---------|------------|
| `disparity` | Normalised Frobenius distance | Closer to 0 |
| `principalAngles` | Subspace angles (radians) between column spaces | Closer to 0 |
| `distCorr` | Mantel correlation between pairwise distance matrices | Closer to 1 |

### Recommended defaults

| Parameter | Default |
|-----------|---------|
| `refSession` | 1 (first session in array) |
| `allowScale` | `false` (orthogonal only) |

### Multi-session usage pattern

```matlab
NEobjs = [NE1, NE2, NE3];
% Must run findEmbedding first on all sessions
for ss = 1:numel(NEobjs)
    NEobjs(ss).numPC = 5;         % set to dStar
    NEobjs(ss).findEmbedding('PCA');
end
pars = diagnostics.pars.ProcrustesAlignment();
pars.refSession = 1;
res = NEobjs.alignSessions(pars);

% Per-session alignment summary
for ss = 2:numel(NEobjs)
    fprintf('Session %d: disparity=%.4f  meanAngle=%.2f°  distCorr=%.3f\n', ...
        ss, res.disparity(ss), rad2deg(res.meanPrincipalAngle(ss)), ...
        res.distCorr(ss));
end
```

---

## Null-model summary table

| Null | Preserves | Breaks | Use for |
|------|-----------|--------|---------|
| Neuron-wise shuffle | Marginal distribution per neuron | Cross-neuron covariance, autocorrelation | Dimension selection |
| Row permutation | Instantaneous covariance | Temporal order | Weaker dimension-selection null |
| Global label permutation | Label marginal distribution | Label–activity alignment | Decoding permutation test (default) |
| Blocked label permutation | Block-level label structure | Within-block alignment | Multi-condition / multi-session decoding |
| Circular time-shift | Autocorrelation per neuron | Neural–label alignment | Temporal decoding null |

---

## Toolbox dependencies

All implementations use **base MATLAB only** (SVD, basic linear algebra).
No Statistics, Machine Learning, or Signal Processing Toolbox functions are
required.

The nearest-centroid decoder is self-contained. If you wish to use a richer
decoder (e.g. SVM), you can supply a custom `statFcn` to
`diagnostics.compute.permutation_test` directly.

---

## File locations

```
+diagnostics/
    +compute/
        dim_parallel_analysis.m   % parallel analysis core
        cv_reconstruction.m       % CV reconstruction core
        cv_decoding.m             % CV decoding core
        permutation_test.m        % generic permutation test
        shuffle_neuronwise.m      % neuron-wise shuffle utility
        circular_shift.m          % circular time-shift utility
        align_procrustes.m        % orthogonal Procrustes alignment
        alignment_metrics.m       % alignment quality metrics
    +pars/
        ParallelAnalysis.m        % default parameters
        CVReconstruction.m
        CVDecoding.m
        ProcrustesAlignment.m
    +shufflers/
        global_permute.m          % global label permuter
        blocked_permute.m         % blocked label permuter
        circular_shift.m          % circular-shift permuter wrapper

@NeuralEmbedding/
    selectDimension.m             % class method – dimension selection
    crossValReconstruct.m         % class method – CV reconstruction
    crossValDecode.m              % class method – CV decoding
    alignSessions.m               % class method – Procrustes alignment

examples/
    demo_manifold_diagnostics.m   % end-to-end demo script

tests/
    smoke_test_diagnostics.m      % automated smoke test
```
