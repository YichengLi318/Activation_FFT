import os
import glob
import argparse
from tqdm import tqdm

# Import the refactored modules
import activation
import analysis
import visual

# --- Constants ---
DATASET_DIR = 'dataset'


def run_extract(args):
    print(f"--- Extracting Activations using model dir: '{args.model_dir}' ---")
    tokenizer, model, device = activation.load_model(args.model_dir)
    if not model:
        print("Model loading failed. Exiting.")
        return

    num_layers = getattr(model.config, 'num_hidden_layers', getattr(model.config, 'n_layer', None))
    hidden_size = getattr(model.config, 'hidden_size', getattr(model.config, 'n_embd', None))
    num_heads = getattr(model.config, 'num_attention_heads', getattr(model.config, 'n_head', None))
    print(f"Model layers: {num_layers}, hidden size: {hidden_size}, attention heads: {num_heads}.")

    # Find corpus files to process
    if args.dataset_file:
        target_path = os.path.join(DATASET_DIR, args.dataset_file)
        if not os.path.exists(target_path):
            print(f"Error: dataset file not found: {target_path}")
            return
        corpus_files = [target_path]
        print(f"Using specified dataset file: {target_path}")
    else:
        corpus_files = glob.glob(os.path.join(DATASET_DIR, '*.json'))
        if args.fast:
            print("--- FAST MODE: Using only the first corpus file. ---")
            corpus_files = corpus_files[:1]

    for corpus_path in tqdm(corpus_files, desc="Extracting Activations"):
        print(f"Processing corpus: {corpus_path}")
        activation.process_corpus(model, tokenizer, corpus_path, num_layers, device, args.fast)
    print("--- Activation extraction complete. ---")


def run_dft_and_plot(args):
    if not args.dataset_file:
        print("Error: --dataset_file is required for DFT and plotting mode.")
        return

    dataset_name = os.path.splitext(os.path.basename(args.dataset_file))[0]
    selected_dims = None
    if args.dims:
        parsed = [int(x.strip()) for x in args.dims.split(',') if x.strip().isdigit()]
        selected_dims = parsed if parsed else None

    input_dir = os.path.join('output', 'data', dataset_name)
    output_dir = os.path.join('output', 'analysis', dataset_name)
    image_dir = os.path.join('output', 'image', dataset_name)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    print(f"--- Performing DFT Analysis for dataset '{dataset_name}' ---")
    analysis.analyze_dft_difference(input_dir, output_dir, selected_dims)
    print("--- DFT analysis complete. ---")

    baseline_dir = None
    if args.baseline_file:
        baseline_name = os.path.splitext(os.path.basename(args.baseline_file))[0]
        baseline_dir = os.path.join('output', 'analysis', baseline_name)
        print(f"Using baseline dataset analysis from '{baseline_dir}' if available.")

    print(f"--- Generating PNG Visualizations for dataset '{dataset_name}' ---")
    visual.generate_dft_plots(output_dir, image_dir, baseline_analysis_dir=baseline_dir)
    print("--- Visualization complete. ---")


def main(args):
    if args.mode == 'extract':
        run_extract(args)
    elif args.mode == 'dft':
        run_dft_and_plot(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM DFT pipeline: extract activations or run DFT+plot.")
    parser.add_argument('--mode', type=str, choices=['extract', 'dft'], default='extract', help="Run mode: 'extract' or 'dft'.")
    parser.add_argument('--model_dir', type=str, default='Qwen3-1.7B', help="Local folder under 'model/' or a HF model id (e.g., 'Qwen/Qwen3-1.7B').")
    parser.add_argument('--dataset_file', type=str, default=None, help="Dataset JSON filename under 'dataset/' (e.g., cn_peoms.json).")
    parser.add_argument('--baseline_file', type=str, default=None, help="Optional baseline dataset JSON filename under 'dataset/'.")
    parser.add_argument('--fast', action='store_true', help="Enable fast mode in extraction: only process first corpus file and few texts.")
    parser.add_argument('--dims', type=str, default=None, help="Comma-separated dimensions to analyze (optional; defaults to random 5).")

    args = parser.parse_args()
    main(args)
