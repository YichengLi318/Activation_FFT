# LLM DFT

<<<<<<< HEAD
This repository studies frequency-domain structure in LLM hidden activations.

## Research Idea

The central question of this project is whether hidden activations in a large language model admit an interpretable frequency-domain description when viewed along the token axis.

The starting point is simple: for a fixed layer, the hidden state of each token can be treated as a sequence over token position. Once that sequence is available, a discrete Fourier transform can be applied dimension-wise. This produces a spectrum that describes how much of the activation energy is concentrated in different token-scale periodic patterns.

This perspective is useful for two reasons.

First, it gives a compact descriptive view of model behavior. Instead of inspecting only raw hidden vectors, we can inspect dominant spectral peaks, their amplitudes, and how these patterns change from shallow to deep layers. In structured text, such as Chinese regulated verse, this creates a direct bridge between observable textual rhythm and internal periodic structure in the model.

Second, it gives an intervention space. After transforming activations into the frequency domain, we can selectively perturb low-frequency or high-frequency bands, invert the transform, and inject the modified activations back into the model. This lets us ask a mechanistic question: which kinds of tasks depend more strongly on global, slowly varying components, and which kinds depend more strongly on local, rapidly varying components?

The full workflow is therefore:

1. Extract hidden states from an internal layer.
2. Apply DFT or rFFT along the token axis.
3. Inspect amplitude and power-spectrum patterns across layers.
4. Edit selected frequency bands and inject the signal back into the model.
5. Compare frequency-domain perturbations with activation-domain perturbations on locality and agreement-style tasks.

## Main Experiments

### Chinese poem experiment

The poem experiment studies classical Chinese verse and asks whether dominant spectral periods align with the number of characters per line. The key observation is that strong peaks appear at periods exactly matching line length: period 5 for five-character poems and period 7 for seven-character poems. This is the most direct evidence in the repository that the frequency representation is not merely a visualization trick, but captures an interpretable textual regularity that the model internally tracks.

### Locality experiment

The locality experiment studies how different tasks respond to perturbations in different frequency bands. The main qualitative result is asymmetric:

- for detail-oriented tasks, interfering with high-frequency components causes larger degradation
- for global tasks, interfering with low-frequency components causes larger degradation

This supports the interpretation that high-frequency components carry fine-grained local information, while low-frequency components carry broader contextual structure.

## What Is Kept

- `main.py`: unified entry point for activation extraction, DFT analysis, and spectral perturbation.
- `function/activation.py`: model loading and hidden-state extraction.
- `function/analysis.py`: DFT analysis and spectrum export.
- `function/perturb.py`: frequency editing and round-trip injection.
- `function/visual.py`: static spectrum plotting.
- `scripts/run_agreement_compare.py`: main locality and agreement experiment driver.
- `scripts/plot_locality_results.py`: plot utility for locality experiments.
- `configs/`: experiment configuration files.
- `results/`: selected figures and a short public-facing results summary.

## Repository Layout

=======
This repository studies frequency-domain structure in LLM hidden activations. The core workflow is intentionally narrow:

1. Extract hidden states from an internal layer.
2. Apply DFT or rFFT along the token axis.
3. Inspect amplitude and power-spectrum patterns across layers.
4. Edit selected frequency bands and inject the signal back into the model.
5. Compare frequency-domain perturbations with activation-domain perturbations on locality and agreement-style tasks.

## What Is Kept

- `main.py`: unified entry point for activation extraction, DFT analysis, and spectral perturbation.
- `function/activation.py`: model loading and hidden-state extraction.
- `function/analysis.py`: DFT analysis and spectrum export.
- `function/perturb.py`: frequency editing and round-trip injection.
- `function/visual.py`: static spectrum plotting.
- `scripts/run_agreement_compare.py`: main locality and agreement experiment driver.
- `scripts/plot_locality_results.py`: plot utility for locality experiments.
- `configs/`: experiment configuration files.
- `results/`: curated public-facing figures and a short results summary.

## Repository Layout

>>>>>>> 22216ed345cc05dfa9b66897e4e6896da6857841
```text
LLM_DFT/
├─ main.py
├─ function/
├─ scripts/
├─ configs/
├─ dataset/
├─ results/
└─ model/
```

<<<<<<< HEAD
`results/` contains a short summary and a small set of selected figures that are suitable for GitHub.
=======
`results/` is now intentionally minimal. It only contains a short summary and a small set of selected figures that are suitable for GitHub.
>>>>>>> 22216ed345cc05dfa9b66897e4e6896da6857841

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the model under `model/Qwen3-1.7B/`.

Run extraction on any dataset JSON available in the workspace:

```bash
python main.py --mode extract --dataset_file path/to/your_dataset.json --model_dir Qwen3-1.7B --fast
```

Run DFT analysis and plotting:

```bash
python main.py --mode dft --dataset_file path/to/your_dataset.json
```

Run one spectral perturbation experiment:

```bash
python main.py --mode perturb --dataset_file path/to/your_dataset.json --model_dir Qwen3-1.7B --layer -1 --operation noise --noise_std 0.05 --topk 5 --max_new_tokens 10 --fast
```

`dataset_file` can point either to a file under `dataset/` or to any existing workspace-relative JSON path.

## Published Results

<<<<<<< HEAD
The public `results/` folder includes a short summary of the main findings together with a few representative figures:
=======
The repository does not keep full raw result archives anymore. Instead, the public `results/` folder only contains:
>>>>>>> 22216ed345cc05dfa9b66897e4e6896da6857841

- a compact English summary of completed experiments
- a few selected spectrum figures for Chinese poem data
- a few selected locality figures

<<<<<<< HEAD
This keeps the repository easy to browse while preserving the main empirical story.
=======
This keeps the repository small and easy to browse while preserving the main empirical story.
>>>>>>> 22216ed345cc05dfa9b66897e4e6896da6857841

## Notes

- `model/` is local-only and should not be committed.
- large datasets under `dataset/` are local-only by default
<<<<<<< HEAD
- plotting scripts remain in the repository and can still be used with locally generated outputs
=======
- plotting scripts remain in the repository, but the public tree no longer ships all raw intermediate outputs they were originally generated from

## Current Scope

This repository is now positioned as a compact research archive and reproduction entry point rather than a dumping ground for one-off scripts and full raw outputs.
>>>>>>> 22216ed345cc05dfa9b66897e4e6896da6857841
