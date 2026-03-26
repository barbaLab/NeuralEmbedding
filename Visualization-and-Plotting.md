# Visualization and Plotting

NeuralEmbedding provides ready-to-use plotting helpers for inspecting embeddings and neural activity. All plotting methods are instance methods on the `NeuralEmbedding` object and assume that embeddings (`obj.E`) and masks (`aMask`, `cMask`) are already configured.

## Available plots

### 3D trajectory plot
- **Call:** `NE.plot3(maxT);` (default `maxT = 40`)
- **What it shows:** The first three embedding dimensions as colored 3D trajectories. A subset of trials closest to the median trajectory is plotted for clarity.
- **Events:** Add behavioral events before plotting with `NE.addEvents(evtsStruct);` (fields: `Ts`, `name`, `trial`, optional `data` for metadata). Events appear as scatter markers with a legend.

### 3D animation
- **Call:** `NE.animate3(maxT);` (default `maxT = 40`)
- **What it shows:** Animated 3D trajectories with a sliding trail (up to 50 time points). Useful for visualizing temporal evolution frame by frame.

### Peri-event time histograms (PETH)
- **Call:** `NE.peth();`
- **What it shows:** Heatmaps of smoothed activity per area and condition. Requires homogeneous trial lengths; uses the original binned/smoothed data rather than the embedding.

## Typical visualization workflow

```matlab
% After embedding
NE.findEmbedding("PCA");

% Limit to a subset of conditions or areas if desired
NE.cMask = "ConditionA";
NE.aMask = "Motor";

% Plot trajectories
NE.plot3(25);           % static 3D plot
NE.animate3(25);        % animation

% Add events and replot
evts = struct( ...
    "Ts",    {0, 150}, ...           % one timestamp per struct element
    "name",  {"GoCue", "GoCue"}, ... % event labels
    "trial", {1, 2}, ...             % matching trial indices
    "data",  {[], []});              % optional metadata payloads
NE.addEvents(evts);
NE.plot3();             % events now overlaid
```

## Tips
- `plot3` and `animate3` use the first three embedding dimensions; ensure the object `NumPC` property (or equivalent method parameter) is at least 3.
- Use masks to focus on specific conditions or areas before plotting to avoid clutter.
- `peth` is a quick diagnostic for preprocessing quality; run it prior to embedding to inspect firing patterns.
