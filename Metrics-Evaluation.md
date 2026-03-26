# Metrics Evaluation

NeuralEmbedding computes quality metrics on the embedded trajectories stored in `obj.E`. Use [`computeMetrics`](./@NeuralEmbedding/computeMetrics.m) to run a metric and store the result on the object.

## Quick start

```matlab
% After computing an embedding
NE.findEmbedding("PCA");

% Compute an embedding quality metric
flag = NE.computeMetrics("arclength");  % returns true on success

% Results live in NE.M (with metadata), plus NE.M_ internally
disp(NE.M);
```

`computeMetrics(type)` accepts several aliases (e.g., `"arc"` → arclength). Parameters are pulled from object properties when present (e.g., `baseline_idx`, `signal_idx` for DPrime).

## Supported metrics (via computeMetrics)

| Metric | Call | What it measures | Key parameters |
| --- | --- | --- | --- |
| **Arclength** | `NE.computeMetrics("arc")` | Median trajectory length across trials using linear/spline/pchip interpolation. | `method` (defaults to spline). |
| **Alignment** | `NE.computeMetrics("alignment")` | Dispersion of trajectories after time normalization. | `type` (default / `equidist` / `index`), `nsec` (segments). |
| **DPrime** | `NE.computeMetrics("dprime")` | Modulation index between baseline and signal windows per dimension. | `baseline_idx`, `signal_idx` (defaults inside metric). |

All metrics expect a cell array of size `nTrials × nAreas` where each cell holds an `nDims × T` matrix (the format produced by `findEmbedding`).

## Additional metric utilities

The `+metrics` package also includes standalone functions that operate on embedded trajectories: `Deviation`, `Tangling`, `Frechet`, and `Poincare`. They accept the same cell-array input (`{nDims × T}` per trial) and return scalars or per-dimension values, enabling custom analyses even when they are not wired into `computeMetrics` yet.

## Best practices
- Run metrics **after** computing an embedding so `obj.E` is populated.
- Set `NE.appendM = true;` before calling `computeMetrics` if you want to store several metric results instead of replacing the most recent one.
- The metrics operate on smoothed, standardized data; adjust preprocessing parameters on the object before embedding if you need different smoothing or binning for the evaluation.
