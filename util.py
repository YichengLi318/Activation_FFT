import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import json

def load_model(model_name='large'):
    """
    Loads a model and tokenizer from a local directory.
    Only 'small' and 'large' are supported, which map to 'gpt2-small' and 'gpt2-large'.
    """
    # Map short names to full model names
    if model_name == 'small':
        full_model_name = 'gpt2-small'
    elif model_name == 'large':
        full_model_name = 'gpt2-large'
    else:
        print(f"Error: Invalid model name '{model_name}'. Only 'small' or 'large' are supported.")
        return None, None

    model_path = os.path.join(os.path.dirname(__file__), 'model', full_model_name)
    print(f"Attempting to load model from '{model_path}'...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at '{model_path}'.")
        print(f"Please ensure the model '{full_model_name}' is available locally.")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model from '{model_path}': {e}")
        return None, None

def get_model_attributes(model):
    """Gets the number of layers and hidden size from the model config."""
    num_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    print(f"Model has {num_layers} hidden layers.")
    print(f"Activation vector length (hidden size): {hidden_size}")
    return num_layers, hidden_size

def get_activations(model, tokenizer, text, layer_idx):
    """Extracts activations for a given text and layer index."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # The first element of the hidden_states tuple is the word embedding output, so the layer number needs to be +1
    activations = outputs.hidden_states[layer_idx + 1]
    return activations.squeeze(0)  # Remove batch dimension

def perform_dft(activation_vector):
    """Performs DFT on an activation vector and returns the magnitude."""
    return np.abs(np.fft.fft(activation_vector))

def analyze_and_save_activations(model, tokenizer, pair, num_layers):
    """
    Analyzes average activations for a given pair for all layers and saves the results.
    Includes checkpointing to skip already processed pairs.
    """
    pair_name = f"{pair['catalog']}_{pair['type']}_{pair['id']}"
    sentence_A, sentence_B = pair["A"], pair["B"]

    for layer_idx in range(num_layers):
        output_dir = os.path.join('output', 'data', f"layer_{layer_idx}")
        os.makedirs(output_dir, exist_ok=True)
        type_file_path = os.path.join(output_dir, f"{pair['type']}.json")
        
        # --- Checkpoint Logic ---
        existing_data = []
        if os.path.exists(type_file_path):
            with open(type_file_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                    if any(item['pair_name'] == pair_name for item in existing_data):
                        continue  # Skip if already processed
                except json.JSONDecodeError:
                    existing_data = []  # File is corrupt or empty, treat as new
        
        # --- Analysis ---
        activations_A = get_activations(model, tokenizer, sentence_A, layer_idx)
        activations_B = get_activations(model, tokenizer, sentence_B, layer_idx)

        avg_activation_A = activations_A.mean(dim=0).numpy()
        avg_activation_B = activations_B.mean(dim=0).numpy()
        dft_avg_A = perform_dft(avg_activation_A)
        dft_avg_B = perform_dft(avg_activation_B)
        
        N = dft_avg_A.shape[-1]
        freq = np.fft.fftfreq(N, d=1.0)[:N // 2]

        result = {
            "id": pair['id'],
            "pair_name": pair_name,
            "layer": layer_idx,
            "frequencies": freq.tolist(),
            "dft_avg_A": dft_avg_A[:N // 2].tolist(),
            "dft_avg_B": dft_avg_B[:N // 2].tolist(),
        }

        # --- Save Result ---
        existing_data.append(result)
        with open(type_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)