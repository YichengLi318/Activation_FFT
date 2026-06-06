# Results

This folder keeps only the public-facing output of the project: a short summary and a few selected figures.

## Completed Experiments

### 1. Frequency analysis of internal activations

The main pipeline extracts hidden activations from a chosen layer, applies DFT or rFFT along the token axis, and studies layer-wise spectral structure.

The Chinese poem experiment is kept here as a compact visual example. Two representative layers are shown below:

- `poem_spectrum_layer_0.png`: an early-layer spectrum snapshot
- `poem_spectrum_layer_20.png`: a deeper-layer spectrum snapshot

These plots show that spectral energy is not distributed uniformly across frequencies, and that the shape changes substantially with depth.

### 2. Locality experiments under spectral editing

The locality experiments compare the effect of removing low-frequency bands versus high-frequency bands on local and global tasks.

- `locality_local_task_curve.png`: the local-task accuracy-drop curve
- `locality_global_task_curve.png`: the global-task accuracy-drop curve

The retained figures show a clear difference between low-band removal and high-band removal, and that the effect is strongly layer-dependent.

### 3. Archived agreement-style comparisons

Agreement-style comparisons were also run during the project, but the large raw result archives have been removed from this public folder. Only the main conclusion is retained here: frequency-domain edits can be compared directly against activation-domain edits, and the resulting layer-wise trends are not identical.

## Selected Figures

### Early-layer poem spectrum

![Early-layer poem spectrum](poem_spectrum_layer_0.png)

### Deep-layer poem spectrum

![Deep-layer poem spectrum](poem_spectrum_layer_20.png)

### Local-task locality curve

![Local-task locality curve](locality_local_task_curve.png)

### Global-task locality curve

![Global-task locality curve](locality_global_task_curve.png)