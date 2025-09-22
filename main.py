import os
import glob
import argparse
from tqdm import tqdm

# Import the refactored modules
import activation
import analysis
import visual

# --- Constants ---
# Automatically find all json files in the dataset directory
DATASET_DIR = 'dataset'
ACTIVATION_DIR = 'output/data'
ANALYSIS_DIR = 'output/analysis'
IMAGE_DIR = 'output/images'

def main(args):
    # --- Setup ---
    os.makedirs(ACTIVATION_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # --- Stage 1: Extract Activations ---
    print(f"--- Stage 1: Extracting Activations using model: '{args.model}' ---")
    
    tokenizer, model, device = activation.load_model(args.model)
    if not model:
        print("Model loading failed. Exiting.")
        return

    num_layers = model.config.n_layer
    print(f"Model has {num_layers} hidden layers and hidden size {model.config.n_embd}.")

    # Find all corpus files to process
    corpus_files = glob.glob(os.path.join(DATASET_DIR, '*.json'))
    if args.fast:
        print("--- Running in FAST MODE: Using only the first corpus file. ---")
        corpus_files = corpus_files[:1]

    for corpus_path in tqdm(corpus_files, desc="1/2 Extracting Activations"):
        print(f"Processing corpus: {corpus_path}")
        activation.process_corpus(model, tokenizer, corpus_path, num_layers, device)
        
    print("--- Activation extraction complete. ---")

    # --- Stage 2: Performing DFT Difference Analysis ---
    print("\n--- Stage 2: Performing DFT Difference Analysis ---")
    analysis.analyze_dft_difference(ACTIVATION_DIR, ANALYSIS_DIR)
    print("--- DFT difference analysis complete. ---")

    # --- Stage 3: Generating Visualizations ---
    print("\n--- Stage 3: Generating DFT Visualizations ---")
    visual.generate_dft_plots(ANALYSIS_DIR, IMAGE_DIR)
    print("--- Visualization complete. ---")
    
    print("\n--- Pipeline finished successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM Topic Specificity analysis pipeline.")
    parser.add_argument(
        '--model', 
        type=str, 
        default='small', 
        choices=['small', 'large'],
        help="Model to use: 'small' for gpt2-small, 'large' for gpt2-large."
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help="Enable fast mode. Processes only the first corpus file found."
    )
    
    args = parser.parse_args()
    main(args)