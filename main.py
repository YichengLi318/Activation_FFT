import os
import json
import shutil
from tqdm import tqdm
from util import load_model, analyze_and_save_activations
from visual import generate_all_plots

# --- Configuration ---
MODEL_NAME = 'large' # Options: 'small', 'large'
FAST_MODE = False    # If True, runs a quick test on 10% of the data.


# --- Constants ---
DATA_FILES = [
    'dataset/syntax_pairs.json',
    'dataset/math_pairs.json'
]
OUTPUT_DIR = 'output/data'
IMAGES_DIR = 'output/images'


def load_data():
    """
    Loads all sentence pairs from the specified data files, ensuring key consistency.
    """
    all_pairs = []
    for file_path in DATA_FILES:
        source_file = os.path.basename(file_path).replace('_pairs.json', '')
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for pair in data:
                # Normalize sentence keys to 'A' and 'B' for consistency.
                if 'sentence_A' in pair:
                    pair['A'] = pair.pop('sentence_A')
                if 'sentence_B' in pair:
                    pair['B'] = pair.pop('sentence_B')
                
                # Add/overwrite 'catalog' from the filename to ensure it always exists and is correct.
                # This is a defensive measure against inconsistent data sources.
                pair['catalog'] = source_file
                all_pairs.append(pair)
    print(f"Loaded {len(all_pairs)} total pairs from data files.")
    return all_pairs


def clear_old_data(directory):
    """
    Clears old data from the output directory.
    """
    if os.path.exists(directory):
        print(f"Clearing old data from {directory}...")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    print("Old data cleared.")


def main():
    """
    Main function to run the analysis and generate plots.
    """
    # clear_old_data(OUTPUT_DIR) # Disabled for checkpointing
    # clear_old_data(IMAGES_DIR) # Disabled for checkpointing
    pairs = load_data()
    tokenizer, model = load_model(MODEL_NAME)
    num_layers = model.config.num_hidden_layers
    
    print(f"Model has {num_layers} hidden layers.")
    print(f"Activation vector length (hidden size): {model.config.hidden_size}")
    print(f"\n--- Starting Analysis with model: '{MODEL_NAME}' ---")

    pairs_to_process = pairs
    if FAST_MODE:
        print("--- Running in FAST MODE: Processing 1 in every 10 pairs. ---")
        pairs_to_process = [pair for i, pair in enumerate(pairs) if i % 10 == 0]
        print(f"Reduced number of pairs from {len(pairs)} to {len(pairs_to_process)}.")

    for pair in tqdm(pairs_to_process, desc="Analyzing Pairs"):
        analyze_and_save_activations(model, tokenizer, pair, num_layers)

    generate_all_plots(OUTPUT_DIR, IMAGES_DIR)


if __name__ == "__main__":
    main()