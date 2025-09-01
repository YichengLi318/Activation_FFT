import os
import json
import glob
from tqdm import tqdm

from util import (
    load_model,
    get_model_attributes,
    create_test_set,
    analyze_pair_activations
)
from visual import generate_all_plots

def main():
    # Configuration
    MODEL_NAME = 'gpt2-large'
    DATA_FILES = [
        os.path.join('dataset', 'syntax_pairs.json'),
        os.path.join('dataset', 'math_pairs.json')
    ]
    BASE_OUTPUT_DIR = 'output'
    OUTPUT_DIR_DATA = os.path.join(BASE_OUTPUT_DIR, 'data')
    OUTPUT_DIR_IMAGES = os.path.join(BASE_OUTPUT_DIR, 'images')
    os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_DATA, exist_ok=True)

    # Create and load the test set
    test_set = create_test_set(DATA_FILES)
    print(f"Test set with {len(test_set)} pairs created and saved to dataset/test_set.json")

    # Load model
    tokenizer, model = load_model(MODEL_NAME)
    if model is None:
        return

    num_layers, hidden_size = get_model_attributes(model)
    layers_to_analyze = range(num_layers)

    # Analysis
    print("\n--- Starting Analysis ---")
    for layer_to_extract in tqdm(layers_to_analyze, desc="Analyzing Layers"):
        layer_results = {
            "layer": layer_to_extract,
            "pairs": []
        }
        json_filename = os.path.join(OUTPUT_DIR_DATA, f"layer_{layer_to_extract}.json")

        if os.path.exists(json_filename):
            continue # Skip if already processed

        for pair in test_set:
            pair_result = analyze_pair_activations(model, tokenizer, pair, layer_to_extract)
            if pair_result is not None:
                layer_results["pairs"].append(pair_result)

        with open(json_filename, 'w') as f:
            json.dump(layer_results, f, indent=4)

    # Visualization
    print("\n--- Analysis complete. Loading data for plotting... ---")

    # Load all layer data for plotting
    all_layer_data = []
    json_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_DATA, 'layer_*.json')), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    for f_path in tqdm(json_files, desc="Loading JSON files"):
        with open(f_path, 'r') as f:
            all_layer_data.append(json.load(f))

    # Generate all plots
    generate_all_plots(all_layer_data, hidden_size, num_layers, OUTPUT_DIR_IMAGES)

if __name__ == "__main__":
    main()