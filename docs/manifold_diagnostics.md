# Manifold Diagnostics

This page describes the **manifold diagnostic toolkit** added to NeuralEmbedding.
It provides robust best-practice diagnostics for reconstructed neural manifolds
and latent spaces, covering six areas:

| Method | Class method | Underlying compute |
|--------|-------------|-------------------|
| A – Dimension selection | `selectDimension` | `diagnostics.compute.dim_parallel_analysis` |
| B – CV reconstruction | `crossValReconstruct` | `diagnostics.compute.cv_reconstruction` |
| C – CV decoding + permutation test | `crossValDecode` | `diagnostics.compute.cv_decoding`, `diagnostics.compute.permutation_test` |
| D – Multi-session alignment | `alignSessions` | `diagnostics.compute.align_procrustes`, `diagnostics.compute.alignment_metrics` |
| E – Within-session stability | `crossValAlignment` | `diagnostics.compute.intra_alignment` |
| F – Event-based labels | `labelsFromEvents` | — (utility method) |

All long-running methods print **progress** to the console in the format
`[Animal.Session]: step N/N done.` Suppress with `pars.verbose = false`.

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
% Option 1: trial-level integer labels
resC = NE.crossValDecode(y_trial, resA.dStar);
% Option 2: build labels from stored events
y_evts = NE.labelsFromEvents({'Cue','Go'});   % per-time-bin categorical
resC2 = NE.crossValDecode(y_evts, resA.dStar);
fprintf('Acc = %.1f%%  p = %.4f\n', resC.accMean*100, resC.pValue);

% D) Align two sessions and activate the aligned subspace
NE1.findEmbedding('PCA'); NE2.findEmbedding('PCA');
resD = alignSessions([NE1, NE2]);
NE2.useAlignment = true;          % get.E / get.W now return aligned data
fprintf('Disparity = %.4f\n', resD.disparity(end, 2));  % last area, session 2

% E) Within-session stability (intra-session split-half)
NE.findEmbedding('PCA');
resE = NE.crossValAlignment();
fprintf('Intra-session disparity = %.4f ± %.4f\n', ...
    resE.disparityMean, resE.disparityStd);

% Inspect stored results (all methods auto-save to M_)
NE.M    % table with ParallelAnalysis, CVReconstruction, CVDecoding, IntraAlignment
```

Multi-session methods accept an array of NeuralEmbedding objects and return a
struct array (one entry per session):

```matlab
NEobjs = [NE1, NE2, NE3];  % vector of NeuralEmbedding objects

% All methods work with object arrays
resA  = NEobjs.selectDimension(1:15);
resD  = NEobjs.alignSessions();
resIA = NEobjs.crossValAlignment();
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
For each non-reference session **and for every area** present in the reference:
1. Centre both latent matrices.
2. Solve `min ||Z_ref - Z_sess * R||_F` subject to `R'R = I` (orthogonal
   Procrustes) via SVD: `[U,S,V] = svd(Z_ref' * Z_sess); R = V * U'`.
3. Optionally optimise an isotropic scale factor (`allowScale = true`).

### Object-level side effects
`alignSessions` stores results inside each object so it becomes **self-sufficient**:

| What is stored | Where |
|----------------|-------|
| Rotated per-trial embeddings | Private `E_aligned_` (per area, per trial) |
| Rotated projection matrix | Private `W_aligned_` (per area) |
| Alignment quality metrics | `M_` via `i_storeM` (type = `'Alignment'`) |

Once `alignSessions` has been called, toggling the public flag
`OBJ.useAlignment = true` makes `OBJ.E` and `OBJ.W` return the **aligned**
subspace instead of the original one, for all areas:

```matlab
NEobjs = [NE1, NE2, NE3];
for ss = 1:3, NEobjs(ss).findEmbedding('PCA'); end
res = alignSessions(NEobjs);       % stores aligned subspaces in each object

NE2.useAlignment = true;           % activate for session 2
E_aligned = NE2.E;                 % returns rotation-transformed embedding

NE2.useAlignment = false;          % restore original
E_original = NE2.E;
```

- Setting `useAlignment = true` before `alignSessions` has been called silently
  falls back to the original embedding (no error, no data corruption).
- `W_aligned_` is `[]` if no embedding was fitted yet.
- The flag is **independent per object**, so you can activate it for a subset of
  sessions (e.g. all non-reference sessions).

### Alignment metrics

| Metric | Meaning | Good value |
|--------|---------|------------|
| `disparity` | Normalised Frobenius distance | Closer to 0 |
| `principalAngles` | Subspace angles (radians) between column spaces | Closer to 0 |
| `distCorr` | Mantel correlation between pairwise distance matrices | Closer to 1 |

Results are now `(nAreas × nSessions)` arrays/cell arrays, one row per area.

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

% Per-session, per-area alignment summary
for ss = 2:numel(NEobjs)
    for aa = 1:numel(res.areas)
        fprintf('Session %d, Area %s: disparity=%.4f  meanAngle=%.2f°\n', ...
            ss, res.areas(aa), res.disparity(aa,ss), ...
            rad2deg(res.meanPrincipalAngle(aa,ss)));
    end
end

% Activate aligned subspace for all non-reference sessions
for ss = 2:numel(NEobjs)
    NEobjs(ss).useAlignment = true;
end
```

---

## Result storage in M_

All diagnostic methods automatically store their output in the object's `M_`
property using the same replace-or-append logic as `computeMetrics`:

| Method | M_ type field |
|--------|--------------|
| `selectDimension` | `'ParallelAnalysis'` |
| `crossValReconstruct` | `'CVReconstruction'` |
| `crossValDecode` | `'CVDecoding'` |
| `alignSessions` (per session) | `'SessionAlignment'` |
| `crossValAlignment` | `'IntraAlignment'` |

```matlab
NE.selectDimension(1:15);
NE.crossValReconstruct(5);
NE.crossValAlignment();
M = NE.M;    % returns table with all stored metrics
M.type       % ["ParallelAnalysis"; "CVReconstruction"; "IntraAlignment"]
M.data{1}    % the full results struct for ParallelAnalysis
```

Re-running any diagnostic with the same condition/area mask **replaces** the
previous entry (no duplicates). Set `NE.appendM = true` to keep all runs.

> **Note**: The cross-session alignment type is `'SessionAlignment'` (not
> `'Alignment'`) to avoid confusion with the pre-existing `alignment` metric
> computed by `computeMetrics`.

---

## E – Within-session stability (`crossValAlignment`)

### What it tests
Whether the latent manifold geometry is **reproducible within the session** —
i.e. whether the manifold learned from a random half of trials matches the
manifold from the other half.

### Method: random split-half Procrustes
For each of `nSplit` replicates:
1. Randomly split all trials into two equal-size groups.
2. Optionally refit PCA on each group independently (when `dim > 0`);
   otherwise use the existing latent coordinates.
3. Align the two latent matrices with orthogonal Procrustes.
4. Record disparity, principal angles, and distance correlation.

### Interpreting results
- Low median disparity (close to 0) → manifold is stable within the session.
- Compare `disparityMean` with the cross-session disparity from `alignSessions`
  to distinguish genuine session-to-session change from within-session
  noise/variability.
- A high within-session disparity relative to the cross-session disparity
  may indicate that the session itself is non-stationary (e.g. learning, drift).

### Recommended defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `nSplit` | 100 | 500 for publication |
| `dim` | 0 (use current E) | Set to `dStar` to refit PCA on each half |
| `allowScale` | `false` | |

```matlab
NE.findEmbedding('PCA');

% Use current embedding directly (no refit)
res = NE.crossValAlignment();
fprintf('Intra-session disparity = %.4f ± %.4f\n', ...
    res.disparityMean, res.disparityStd);

% Compare with cross-session alignment
resX = alignSessions([NE1, NE2]);
fprintf('Cross-session disparity = %.4f\n', resX.disparity(1,2));
```

---

## F – Event-based time-bin labels (`labelsFromEvents`)

### What it does
`labelsFromEvents` converts stored behavioral events into a **per-time-bin
categorical label vector** that can be passed directly to `crossValDecode` or
`crossValReconstruct`.

Each time bin in each trial is labelled with the name of the *most recent*
event (from the requested list) that has occurred up to that bin. Bins before
the first requested event are labelled `'0'`.

### Usage

```matlab
% Add events first
NE.addEvents(evts);   % evts: struct with fields Ts, Name, Trial

% Decode which event epoch each time bin belongs to
y = NE.labelsFromEvents({'Cue','Go','Reward'});
% y is a T x 1 categorical with categories '0','Cue','Go','Reward'

% Use directly with crossValDecode
res = NE.crossValDecode(y, dStar);

% Use all unique event names
y_all = NE.labelsFromEvents("all");
```

### Priority tie-breaking
If two events from the requested list share the same timestamp, the one that
appears **earlier** in the `eventNames` input wins (i.e. input order
determines priority).

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
        permutation_test.m        % generic permutation test (with progress)
        shuffle_neuronwise.m      % neuron-wise shuffle utility
        circular_shift.m          % circular time-shift utility
        align_procrustes.m        % orthogonal Procrustes alignment
        alignment_metrics.m       % alignment quality metrics
        intra_alignment.m         % within-session split-half stability
    +pars/
        ParallelAnalysis.m        % default parameters (includes verbose)
        CVReconstruction.m
        CVDecoding.m
        ProcrustesAlignment.m
        IntraAlignment.m          % parameters for within-session stability
    +shufflers/
        global_permute.m          % global label permuter
        blocked_permute.m         % blocked label permuter
        circular_shift.m          % circular-shift permuter wrapper

@NeuralEmbedding/
    selectDimension.m             % class method – dimension selection (auto-saves to M_)
    crossValReconstruct.m         % class method – CV reconstruction (auto-saves to M_)
    crossValDecode.m              % class method – CV decoding (auto-saves to M_)
    alignSessions.m               % class method – Procrustes alignment (stores E_aligned_, W_aligned_, saves to M_ as 'SessionAlignment')
    crossValAlignment.m           % class method – within-session stability (auto-saves to M_ as 'IntraAlignment')
    labelsFromEvents.m            % utility – build per-time-bin labels from stored events
    i_storeM.m                    % private helper – update M_ with a diagnostic result

examples/
    demo_manifold_diagnostics.m   % end-to-end demo script

tests/
    smoke_test_diagnostics.m      % automated smoke test
```
