import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import json
from tqdm import tqdm

def load_model(model_name='large'):
    """
    Loads a model and tokenizer from a local directory.
    Only 'small' and 'large' are supported, which map to 'gpt2-small' and 'gpt2-large'.
    """
    if model_name == 'small':
        full_model_name = 'gpt2-small'
    elif model_name == 'large':
        full_model_name = 'gpt2-large'
    else:
        print(f"Error: Invalid model name '{model_name}'. Only 'small' or 'large' are supported.")
        return None, None, None

    model_path = os.path.join(os.path.dirname(__file__), 'model', full_model_name)
    print(f"Attempting to load model from '{model_path}'...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"========================================")
    print(f"           USING DEVICE: {device.upper()}       ")
    print(f"========================================")

    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at '{model_path}'.")
        print(f"Please ensure the model '{full_model_name}' is available locally.")
        return None, None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return tokenizer, model, device
    except Exception as e:
        print(f"Failed to load model from '{model_path}': {e}")
        return None, None, None

def get_model_attributes(model):
    """Gets the number of layers and hidden size from the model config."""
    num_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    print(f"Model has {num_layers} hidden layers.")
    print(f"Activation vector length (hidden size): {hidden_size}")
    return num_layers, hidden_size

def get_activations(model, tokenizer, text, layer_idx, device):
    """Extracts activations for a given text and layer index."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=model.config.n_positions)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    except Exception as e:
        print(f"Error during model inference: {e}")
        if 'CUDA out of memory' in str(e):
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")
        return None

    activations = outputs.hidden_states[layer_idx + 1]
    return activations.squeeze(0)

def process_corpus(model, tokenizer, corpus_path, num_layers, device):
    """
    Extracts and saves the raw token activations for a given corpus for all layers.
    Saves each item's activations to a separate file for scalability.
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    category = os.path.basename(corpus_path).replace('.json', '')

    for layer_idx in range(num_layers):
        print(f"Processing Layer {layer_idx} for {category}...")
        
        for item in tqdm(corpus, desc=f"Layer {layer_idx}"):
            item_id = item['id']
            output_dir = os.path.join('output', 'data', f"layer_{layer_idx}", category)
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, f"{item_id}.json")

            if os.path.exists(output_file_path):
                continue

            activations = get_activations(model, tokenizer, item['content'], layer_idx, device)
            if activations is None:
                print(f"Skipping item {item_id} due to inference error.")
                continue

            result = {
                "id": item_id,
                "category": item['category'],
                "layer": layer_idx,
                "activations": activations.cpu().numpy().tolist(),
            }

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f)