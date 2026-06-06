# LLM DFT

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

`results/` is now intentionally minimal. It only contains a short summary and a small set of selected figures that are suitable for GitHub.

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

The repository does not keep full raw result archives anymore. Instead, the public `results/` folder only contains:

- a compact English summary of completed experiments
- a few selected spectrum figures for Chinese poem data
- a few selected locality figures

This keeps the repository small and easy to browse while preserving the main empirical story.

## Notes

- `model/` is local-only and should not be committed.
- large datasets under `dataset/` are local-only by default
- plotting scripts remain in the repository, but the public tree no longer ships all raw intermediate outputs they were originally generated from

## Current Scope

This repository is now positioned as a compact research archive and reproduction entry point rather than a dumping ground for one-off scripts and full raw outputs.
